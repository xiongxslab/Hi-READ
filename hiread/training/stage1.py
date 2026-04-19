import os
from typing import Iterable, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.functional.image import structural_similarity_index_measure

from hiread.data.genome_dataset import GenomeDataset
from hiread.model import hiread_models

try:
    from pytorch_msssim import ssim as _msssim_ssim
except ImportError:  # pragma: no cover - optional runtime dependency
    _msssim_ssim = None


def build_stage1_feature_dict(feature_specs):
    feature_dict = {}
    for name, file_name, norm in feature_specs:
        feature_dict[name] = {"file_name": file_name, "norm": norm}
    return feature_dict


def _build_single_dataset(
    dataset_root,
    assembly,
    celltype,
    feature_dict,
    mode,
    train_chromosomes=None,
    val_chromosomes=None,
    test_chromosomes=None,
    include_sequence=True,
    include_genomic_features=True,
    use_aug=True,
):
    celltype_root = os.path.join(dataset_root, celltype)
    centrotelo_path = os.path.join(dataset_root, "centrotelo.bed")
    return GenomeDataset(
        celltype_root=celltype_root,
        genome_assembly=assembly,
        feat_dicts=feature_dict,
        mode=mode,
        include_sequence=include_sequence,
        include_genomic_features=include_genomic_features,
        use_aug=use_aug,
        train_chromosomes=train_chromosomes,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        centrotelo_path=centrotelo_path,
    )


def build_stage1_dataset(
    dataset_roots: Iterable[str],
    assembly: str,
    celltype: str,
    feature_dict,
    mode: str,
    train_chromosomes=None,
    val_chromosomes=None,
    test_chromosomes=None,
    include_sequence=True,
    include_genomic_features=True,
    use_aug=True,
):
    datasets = [
        _build_single_dataset(
            dataset_root=root,
            assembly=assembly,
            celltype=celltype,
            feature_dict=feature_dict,
            mode=mode,
            train_chromosomes=train_chromosomes,
            val_chromosomes=val_chromosomes,
            test_chromosomes=test_chromosomes,
            include_sequence=include_sequence,
            include_genomic_features=include_genomic_features,
            use_aug=use_aug,
        )
        for root in dataset_roots
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


class Stage1LightningModule(pl.LightningModule):
    def __init__(
        self,
        model_type="ConvTransModel",
        num_genomic_features=1,
        mid_hidden=256,
        pos_embedding="EPEG",
        learning_rate=1e-3,
        weight_decay=0.1,
        max_epochs=100,
        ssim_start_epoch=10,
        ssim_max_epoch=30,
        ssim_max_weight=0.15,
        warmup_epochs=10,
        min_learning_rate=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        model_cls = getattr(hiread_models, model_type)
        self.model = model_cls(
            num_genomic_features=num_genomic_features,
            mid_hidden=mid_hidden,
            pos_embedding=pos_embedding,
        )

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim=1)
        inputs = torch.cat([seq, features], dim=2).float()
        return inputs, mat.float()

    def _align_outputs_and_targets(self, output, target):
        if output.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        elif output.dim() == 3 and target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        if output.shape[-2:] != target.shape[-2:]:
            min_height = min(output.shape[-2], target.shape[-2])
            min_width = min(output.shape[-1], target.shape[-1])
            output = output[..., :min_height, :min_width]
            target = target[..., :min_height, :min_width]

        return output, target

    def _compute_ssim_loss(self, output, target):
        if output.dim() == 3:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)

        data_range = (target.max() - target.min()).detach()
        if not torch.isfinite(data_range) or data_range <= 0:
            data_range = target.new_tensor(1.0)

        if _msssim_ssim is not None:
            ssim_value = _msssim_ssim(
                output,
                target,
                data_range=float(data_range.item()),
                size_average=True,
            )
        else:
            ssim_value = structural_similarity_index_measure(
                output.float(),
                target.float(),
                data_range=float(data_range.item()),
            )
        return 1 - ssim_value

    def _compute_ssim_weight(self):
        current_epoch = self.current_epoch
        if current_epoch < self.hparams.ssim_start_epoch:
            return 0.0
        if current_epoch >= self.hparams.ssim_max_epoch:
            return float(self.hparams.ssim_max_weight)
        ramp_span = max(self.hparams.ssim_max_epoch - self.hparams.ssim_start_epoch, 1)
        return float(self.hparams.ssim_max_weight) * (current_epoch - self.hparams.ssim_start_epoch) / ramp_span

    def training_step(self, batch, batch_idx):
        inputs, target = self.proc_batch(batch)
        output = self(inputs)
        output, target = self._align_outputs_and_targets(output, target)

        mse_loss = F.mse_loss(output, target)
        ssim_weight = self._compute_ssim_weight()
        if ssim_weight > 0:
            ssim_loss = self._compute_ssim_loss(output, target)
        else:
            ssim_loss = output.new_tensor(0.0)

        total_loss = mse_loss + ssim_weight * ssim_loss
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=inputs.shape[0])
        self.log("train_ssim_loss", ssim_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=inputs.shape[0])
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.shape[0])
        self.log("ssim_weight", ssim_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=inputs.shape[0])
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, target = self.proc_batch(batch)
        output = self(inputs)
        output, target = self._align_outputs_and_targets(output, target)
        mse_loss = F.mse_loss(output, target)
        ssim_loss = self._compute_ssim_loss(output, target)
        self.log("val_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.shape[0], sync_dist=True)
        self.log("val_ssim_loss", ssim_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=inputs.shape[0], sync_dist=True)
        return mse_loss

    def test_step(self, batch, batch_idx):
        inputs, target = self.proc_batch(batch)
        output = self(inputs)
        output, target = self._align_outputs_and_targets(output, target)
        mse_loss = F.mse_loss(output, target)
        ssim_loss = self._compute_ssim_loss(output, target)
        self.log("test_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.shape[0], sync_dist=True)
        self.log("test_ssim_loss", ssim_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=inputs.shape[0], sync_dist=True)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        warmup_epochs = min(max(int(self.hparams.warmup_epochs), 0), max(int(self.hparams.max_epochs), 1))
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_epochs = max(int(self.hparams.max_epochs) - warmup_epochs, 1)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=self.hparams.min_learning_rate,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(int(self.hparams.max_epochs), 1),
                eta_min=self.hparams.min_learning_rate,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def _make_dataloader(dataset, batch_size, shuffle, num_workers):
    persistent_workers = num_workers > 0
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": persistent_workers,
    }
    if persistent_workers:
        loader_kwargs["prefetch_factor"] = 1
    return DataLoader(**loader_kwargs)


def train_stage1(
    dataset_roots,
    assembly,
    celltype,
    feature_specs,
    output_dir,
    train_chromosomes=None,
    val_chromosomes=None,
    test_chromosomes=None,
    batch_size=8,
    num_workers=16,
    max_epochs=80,
    patience=80,
    seed=2077,
    model_type="ConvTransModel",
    num_genomic_features=1,
    mid_hidden=256,
    pos_embedding="EPEG",
    learning_rate=2e-4,
    weight_decay=0.1,
    ssim_start_epoch=10,
    ssim_max_epoch=30,
    ssim_max_weight=0.15,
    warmup_epochs=10,
    min_learning_rate=1e-6,
    accelerator="auto",
    devices="auto",
    precision="bf16-mixed",
    checkpoint_path: Optional[str] = None,
):
    pl.seed_everything(seed, workers=True)
    feature_dict = build_stage1_feature_dict(feature_specs)

    train_dataset = build_stage1_dataset(
        dataset_roots=dataset_roots,
        assembly=assembly,
        celltype=celltype,
        feature_dict=feature_dict,
        mode="train",
        train_chromosomes=train_chromosomes,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        use_aug=True,
    )
    val_dataset = build_stage1_dataset(
        dataset_roots=dataset_roots,
        assembly=assembly,
        celltype=celltype,
        feature_dict=feature_dict,
        mode="val",
        train_chromosomes=train_chromosomes,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        use_aug=False,
    )

    train_loader = _make_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _make_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = Stage1LightningModule(
        model_type=model_type,
        num_genomic_features=num_genomic_features,
        mid_hidden=mid_hidden,
        pos_embedding=pos_embedding,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        ssim_start_epoch=ssim_start_epoch,
        ssim_max_epoch=ssim_max_epoch,
        ssim_max_weight=ssim_max_weight,
        warmup_epochs=warmup_epochs,
        min_learning_rate=min_learning_rate,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="stage1-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger(save_dir=output_dir, name="logs", version="stage1")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        precision=precision,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=checkpoint_path or None)
    return trainer.checkpoint_callback.best_model_path or trainer.checkpoint_callback.last_model_path
