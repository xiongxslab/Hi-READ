# Hi-READ

Hi-READ is a two-stage framework for predicting local chromatin contact maps from DNA sequence and an epigenomic track, with downstream workflows for attribution, loop analysis, structural-variant simulation, and gene-disease summarization.

## Repository Layout

``` text
Hi-READ/
├── hiread/                    # Stage-1 model, data loaders, inference utilities
├── hiread_diffusion/          # Stage-2 diffusion model
├── scripts/                   # Public Python entry points and workflow wrappers
├── workflows/                 # Downstream analysis workflows
├── demo_data/                 # Small demo inputs only
├── demo_scripts/              # One-command demos
├── demo_outputs/              # Demo outputs generated locally
├── configs/
├── requirements.txt
├── environment.yml
├── setup.py
└── README.md
```

## Installation

Python-only installation:

``` bash
conda create -n hiread python=3.10
conda activate hiread
pip install -r requirements.txt
pip install -e .
```

Conda environment with R packages for the bundled workflow demos:

``` bash
conda env create -f environment.yml
conda activate hiread
pip install -e .
```

Some downstream analysis demo scripts require R packages including `data.table`, `dplyr`, `ggplot2`, `tidyr`, `GenomicRanges`, and `rtracklayer`.

## Core Model Data Layout

For training and prediction, arrange each dataset root as:

``` text
DATA_ROOT/
├── centrotelo.bed
├── dna_sequence/
│   ├── chr1.fa.gz
│   ├── chr2.fa.gz
│   └── ...
└── cell line/
    ├── genomic_features/
    │   └── chip.bw
    └── hic_matrix/
        ├── chr1.npz
        ├── chr2.npz
        └── ...
```

Default split:

-   validation chromosome: `chr10`
-   test chromosome: `chr15`
-   training chromosomes: remaining chromosomes unless overridden

## Core Commands

### Stage-1 Training

``` bash
python scripts/train_stage1.py \
  --data-root DATA_ROOT \
  --celltype cell line \
  --output-dir outputs/stage1_run
```

### Stage-2 Training

``` bash
python scripts/train_stage2.py \
  --stage1-checkpoint outputs/stage1_run/checkpoints/last.ckpt \
  --data-root DATA_ROOT \
  --celltype cell line \
  --output-dir outputs/stage2_run
```

### Two-Stage Training

``` bash
python scripts/train_two_stage.py \
  --stage1-data-root DATA_ROOT \
  --celltype cell line \
  --output-dir outputs/two_stage_run
```

### Stage-1 Prediction

``` bash
python scripts/predict_stage1.py \
  --model outputs/stage1_run/checkpoints/last.ckpt \
  --seq-dir DATA_ROOT/dna_sequence \
  --chip-bw DATA_ROOT/cell line/genomic_features/chip.bw \
  --celltype cell line \
  --chr chr15 \
  --start 30000000 \
  --out outputs/stage1_prediction
```

### Stage-2 Prediction

``` bash
python scripts/predict_stage2.py \
  --stage2-checkpoint outputs/stage2_run/checkpoints/last.ckpt \
  --stage1-checkpoint outputs/stage1_run/checkpoints/last.ckpt \
  --data-root DATA_ROOT \
  --celltype cell line \
  --chr chr15 \
  --start 30000000 \
  --out outputs/stage2_prediction
```

## **Download dataset files and pretrained model weights**

[Training datasets and Pretrained model weights](https://drive.google.com/file/d/1dJQ2To1zJ9CFGS2AHhCMqsOFtx8iXYSS/view?usp=drive_link)

## Resource

Resource heatmap and loops predicted from the Hi-READ model can be used at <https://xiongxslab.github.io/Hi-READ/>

### GRAM Attribution

``` bash
python scripts/analyze_gram.py \
  --chr chr15 \
  --start 30146560 \
  --model outputs/stage1_run/checkpoints/last.ckpt \
  --seq DATA_ROOT/dna_sequence \
  --chip DATA_ROOT/cell line/genomic_features/chip.bw \
  --celltype cell line
```

### Screening(Impact score calculation)

``` bash
python scripts/analyze_screening.py whole \
  --chr chr21 \
  --model outputs/stage1_run/checkpoints/last.ckpt \
  --seq DATA_ROOT/dna_sequence \
  --chip DATA_ROOT/cell line/genomic_features/chip.bw \
  --celltype cell line
```

``` bash
python scripts/analyze_screening.py fine \
  --model outputs/stage1_run/checkpoints/last.ckpt \
  --seq DATA_ROOT/dna_sequence \
  --chip DATA_ROOT/cell line/genomic_features/chip.bw \
  --celltype cell line \
  --regions-csv candidate_regions.csv
```

`candidate_regions.csv` must contain:

``` text
chr,start,end
```

## Citation
