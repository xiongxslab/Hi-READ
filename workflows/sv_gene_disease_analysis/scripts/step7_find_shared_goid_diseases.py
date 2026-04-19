#!/usr/bin/env python3

import argparse
import csv
import os
from typing import Dict, List, Tuple


def read_sig_terms(
    run_root: str, tf: str, bucket: str, padj_thr: float
) -> Dict[str, Dict[str, dict]]:
    """
    Returns: disease -> term_id -> {Description,p.adjust,Count,geneID}
    Keeps the best (lowest p.adjust) row per (disease, term_id).
    """
    p = os.path.join(
        run_root, "raw_go_results", "min_sv1", tf, bucket, "disease_go_enrichment.csv"
    )
    if not os.path.exists(p):
        raise FileNotFoundError(p)

    by: Dict[str, Dict[str, dict]] = {}
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            disease = row.get("group_label")
            tid = row.get("ID")
            if not disease or not tid:
                continue
            try:
                padj = float(row.get("p.adjust", "nan"))
            except Exception:
                continue
            if padj != padj or padj >= padj_thr:
                continue

            disease_map = by.setdefault(disease, {})
            prev = disease_map.get(tid)
            if prev is None or padj < float(prev["p.adjust"]):
                disease_map[tid] = {
                    "ID": tid,
                    "Description": row.get("Description", ""),
                    "p.adjust": padj,
                    "Count": row.get("Count", ""),
                    "geneID": row.get("geneID", ""),
                }
    return by


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find diseases whose significant GO term IDs intersect across all TFs in a TF-set (min_sv1)."
    )
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--bucket", default="brain", choices=["brain", "heart", "immune"])
    ap.add_argument("--padj", type=float, default=0.05)
    ap.add_argument("--tfs", required=True, help="Comma-separated TF list, e.g. CTCF,Rad21")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    if len(tfs) < 2:
        raise ValueError("--tfs must include at least 2 TFs")

    per_tf = {tf: read_sig_terms(args.run_root, tf, args.bucket, args.padj) for tf in tfs}
    diseases = set.intersection(*[set(per_tf[tf].keys()) for tf in tfs])

    out_rows: List[dict] = []
    for disease in sorted(diseases):
        term_sets = [set(per_tf[tf][disease].keys()) for tf in tfs]
        inter = set.intersection(*term_sets)
        if not inter:
            continue

        # Conservative shared p.adjust: max across TFs (for the same term).
        best_tid = None
        best_p = None
        best_desc = ""
        for tid in inter:
            p = max(float(per_tf[tf][disease][tid]["p.adjust"]) for tf in tfs)
            if best_p is None or p < best_p:
                best_p = p
                best_tid = tid
                best_desc = per_tf[tfs[0]][disease][tid]["Description"]

        out_rows.append(
            {
                "bucket": args.bucket,
                "disease": disease,
                "shared_term_count": len(inter),
                "best_shared_term_id": best_tid,
                "best_shared_term_desc": best_desc,
                "best_shared_padj_conservative": best_p,
                "shared_term_ids": ";".join(sorted(inter)),
            }
        )

    # Sort by shared term count desc, then best conservative padj asc.
    out_rows.sort(
        key=lambda r: (-int(r["shared_term_count"]), float(r["best_shared_padj_conservative"]), r["disease"])
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [
            "bucket",
            "disease",
            "shared_term_count",
            "best_shared_term_id",
            "best_shared_term_desc",
            "best_shared_padj_conservative",
            "shared_term_ids",
        ])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(args.out_csv)
    print("n_candidates", len(out_rows))


if __name__ == "__main__":
    main()

