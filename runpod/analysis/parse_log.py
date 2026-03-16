"""
Parse training.log and extract metrics for both baseline and film runs.
Saves results to parsed_metrics.json for use by other scripts.

Usage:
    python analysis/parse_log.py                        # uses training.log in parent dir
    python analysis/parse_log.py --log path/to/file.log
"""

import argparse
import json
import re
from pathlib import Path


def parse_log(log_path: str) -> dict:
    runs = {}

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Match eval lines: [run_name] step  10000/50000 | loss x | val x | ppl x | lr x
            m = re.search(
                r"\[(\S+)\] step\s+(\d+)/(\d+) \| loss ([\d.]+) \| val ([\d.]+) \| ppl ([\d.]+) \| lr ([\d.e+-]+)",
                line,
            )
            if not m:
                continue

            run_name  = m.group(1)
            step      = int(m.group(2))
            max_steps = int(m.group(3))
            train_loss = float(m.group(4))
            val_loss   = float(m.group(5))
            val_ppl    = float(m.group(6))
            lr         = float(m.group(7))

            if run_name not in runs:
                runs[run_name] = {
                    "max_steps":   max_steps,
                    "steps":       [],
                    "train_loss":  [],
                    "val_loss":    [],
                    "val_ppl":     [],
                    "lr":          [],
                }

            runs[run_name]["steps"].append(step)
            runs[run_name]["train_loss"].append(train_loss)
            runs[run_name]["val_loss"].append(val_loss)
            runs[run_name]["val_ppl"].append(val_ppl)
            runs[run_name]["lr"].append(lr)

    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="../training.log")
    p.add_argument("--out", default="analysis/parsed_metrics.json")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        # try relative to script location
        log_path = Path(__file__).parent.parent / "training.log"

    print(f"Parsing: {log_path}")
    runs = parse_log(log_path)

    for name, data in runs.items():
        steps = data["steps"]
        ppls  = data["val_ppl"]
        best  = min(ppls)
        print(f"  {name}: {len(steps)} eval points | best val ppl = {best:.2f} @ step {steps[ppls.index(best)]}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
