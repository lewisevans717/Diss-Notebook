"""
Parse a SLURM .out log file from run_pipeline.py and reconstruct results.xlsx.

Usage:
    python parse_log.py <log_file> <model_name> [--output results.xlsx]

Example:
    python parse_log.py logs/imagenet_pipeline_23656.out resnet50_imagenet \
        --output results_imagenet_23656.xlsx
"""

import argparse
import re
import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline name → (compression, attack, quality, epsilon) metadata
# ---------------------------------------------------------------------------

def parse_pipeline_name(name: str):
    """Reverse-engineer compression/attack/quality/epsilon from a pipeline name."""
    name = name.strip()

    compression = None
    attack = None
    quality = None
    epsilon = None

    COMPRESSIONS = {"jpeg", "jpeg2000", "pca", "patchsvd", "lic_roi"}
    ATTACKS = {"fgsm", "pgd", "apgd"}

    if name == "clean":
        return compression, attack, quality, epsilon

    # attack-only:  e.g. "fgsm    (eps=0.03)"
    m = re.fullmatch(r'(\w+)\s+\(eps=([\d.]+)\)', name)
    if m and m.group(1) in ATTACKS:
        attack = m.group(1)
        epsilon = float(m.group(2))
        return compression, attack, quality, epsilon

    # compression-only:  e.g. "jpeg    (25%)"
    m = re.fullmatch(r'(\w+)\s+\((\d+)%\)', name)
    if m and m.group(1) in COMPRESSIONS:
        compression = m.group(1)
        quality = float(m.group(2))
        return compression, attack, quality, epsilon

    # compression -> attack:  e.g. "jpeg -> fgsm    (25%, eps=0.03)"
    m = re.fullmatch(r'(\w+)\s+->\s+(\w+)\s+\((\d+)%,\s*eps=([\d.]+)\)', name)
    if m:
        compression = m.group(1)
        attack = m.group(2)
        quality = float(m.group(3))
        epsilon = float(m.group(4))
        return compression, attack, quality, epsilon

    # attack -> compression:  e.g. "fgsm -> jpeg    (eps=0.03, 25%)"
    m = re.fullmatch(r'(\w+)\s+->\s+(\w+)\s+\(eps=([\d.]+),\s*(\d+)%\)', name)
    if m:
        attack = m.group(1)
        compression = m.group(2)
        epsilon = float(m.group(3))
        quality = float(m.group(4))
        return compression, attack, quality, epsilon

    raise ValueError(f"Could not parse pipeline name: '{name}'")


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_log(log_path: str, model_name: str) -> pd.DataFrame:
    with open(log_path) as f:
        content = f.read()

    # Split on pipeline blocks
    # Each block starts with "  Pipeline: <name> - started <time>"
    pipeline_pattern = re.compile(
        r'Pipeline:\s+(.+?)\s+-\s+started\s+[\d:.]+\s*\n'
        r'.*?- Accuracy:\s+([\d.]+)%\s*\n'
        r'.*?- PSNR:\s+([\d.eE+\-]+|None)\s*\n'
        r'.*?- MSE:\s+([\d.eE+\-]+|None)\s*\n'
        r'.*?- MAE:\s+([\d.eE+\-]+|None)\s*\n'
        r'.*?- Total:\s+(\d+)\s*\n'
        r'.*?- Correct:\s+(\d+)',
        re.DOTALL
    )

    records = []
    for m in pipeline_pattern.finditer(content):
        pipeline_name = m.group(1).strip()
        accuracy_pct = float(m.group(2))
        psnr = None if m.group(3) == "None" else float(m.group(3))
        mse  = None if m.group(4) == "None" else float(m.group(4))
        mae  = None if m.group(5) == "None" else float(m.group(5))
        total   = int(m.group(6))
        correct = int(m.group(7))

        compression, attack, quality, epsilon = parse_pipeline_name(pipeline_name)

        records.append({
            "model":       model_name,
            "pipeline":    pipeline_name,
            "compression": compression,
            "attack":      attack,
            "quality":     quality,
            "epsilon":     epsilon,
            "accuracy":    accuracy_pct / 100.0,
            "correct":     correct,
            "total":       total,
            "psnr_mean":   psnr,
            "mse_mean":    mse,
            "mae_mean":    mae,
        })

    print(f"Parsed {len(records)} pipelines from {log_path}")
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Parse run_pipeline.py SLURM log to xlsx.")
    p.add_argument("log_file", help="Path to the .out log file")
    p.add_argument("model_name", help="Model name (e.g. resnet50_imagenet)")
    p.add_argument("--output", default="results_parsed.xlsx",
                   help="Output .xlsx path (default: results_parsed.xlsx)")
    args = p.parse_args()

    df = parse_log(args.log_file, args.model_name)
    df.to_excel(args.output, sheet_name="Sheet1", index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
