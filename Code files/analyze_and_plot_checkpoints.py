#!/usr/bin/env python3
"""
analyze_and_plot_checkpoints.py

- Inspect checkpoints to find plausible evaluation metrics (return_avg_mean).
- Parse logs for recent_avg_return and learning rate traces.
- Optionally evaluate checkpoints deterministically if metric is missing.
- Plot logs + checkpoint evals + lr overlay.

Usage (terminal):
    python analyze_and_plot_checkpoints.py --logs logs.txt --ckpt-dir checkpoints --out combined.png --eval-if-missing --eval-episodes 50

Notes:
- If you want automatic deterministic checkpoint evaluation, implement build_model() and act_deterministic().
- The script is defensive and will print diagnostics if it can't find metrics.
"""

import argparse
import re
from pathlib import Path
import math
import sys
import csv
import warnings

# plotting and torch imports
import matplotlib.pyplot as plt
import numpy as np

# optional heavy import, import inside functions to avoid memory issues when not needed
try:
    import torch
except Exception:
    torch = None

# ------------------------------
# Config / candidate keys
# ------------------------------
CHECKPOINT_KEY_CANDIDATES = [
    "return_avg_mean",
    "mean_return",
    "metrics.return_avg_mean",
    "metrics.mean_return",
    "eval.return_avg_mean",
    "eval.mean_return",
    "results.return_avg_mean",
    "evaluation.return_avg_mean",
    "eval/return_avg_mean",
]

LR_PATTERNS = [
    re.compile(r"lr[:=]\s*([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"learning_rate[:=]\s*([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"policy_lr[:=]\s*([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"current_lr[:=]\s*([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)", re.IGNORECASE),
]

STEP_RECENT_PATTERN = re.compile(
    r"Steps:\s*([0-9]+)\s*/\s*([0-9]+)[^\n]*recent[_ ]?avg[_ ]?return:\s*([+-]?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
ALTERNATE_STEP_PATTERN = re.compile(
    r"(?:Step|step|steps|update|iter)[:=]?\s*([0-9]+)[^\n]{0,60}recent[_ ]?avg[_ ]?return[:=]?\s*([+-]?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

# ------------------------------
# Utility functions
# ------------------------------
def extract_step_from_name(name: str):
    digits = re.findall(r"\d+", name)
    if not digits:
        return None
    return int("".join(digits))


def try_get_value_from_obj(obj, key_path):
    parts = key_path.split(".")
    cur = obj
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def find_metric_in_checkpoint(ckpt):
    """Try candidate keys and heuristics to find plausible 'mean return' metric in checkpoint dict."""
    # candidates first
    for k in CHECKPOINT_KEY_CANDIDATES:
        if "." in k:
            val = try_get_value_from_obj(ckpt, k)
            if val is not None:
                try:
                    return float(val)
                except:
                    pass
        else:
            if isinstance(ckpt, dict) and k in ckpt:
                try:
                    return float(ckpt[k])
                except:
                    pass

    # nested containers heuristics
    for rootk in ["metrics", "eval", "results", "logs", "info", "evaluation"]:
        if isinstance(ckpt, dict) and rootk in ckpt and isinstance(ckpt[rootk], dict):
            for subk in ["return_avg_mean", "mean_return", "avg_return", "eval_return"]:
                if subk in ckpt[rootk]:
                    try:
                        return float(ckpt[rootk][subk])
                    except:
                        pass

    # scan for numeric candidates and pick plausible ones
    candidates = []
    def scan(d, path=""):
        if isinstance(d, dict):
            for kk, vv in d.items():
                scan(vv, f"{path}.{kk}" if path else kk)
        elif isinstance(d, (float, int)):
            candidates.append((path, float(d)))
        elif isinstance(d, (list, tuple)) and len(d)>0 and all(isinstance(x,(int,float)) for x in d[:20]):
            candidates.append((path + " (list_mean)", float(sum(d)/len(d))))
    scan(ckpt)

    # filter plausible ranges for LunarLander-ish returns (heuristic)
    plausible = [(k,v) for (k,v) in candidates if -5000 < v < 5000]
    if not plausible:
        return None

    # choose candidate whose absolute value is large but plausible (often mean returns are > 0 later)
    plausible_sorted = sorted(plausible, key=lambda x: abs(x[1]), reverse=True)
    return float(plausible_sorted[0][1])


# ------------------------------
# Logging / parsing functions
# ------------------------------
def parse_logs(logfile: Path):
    text = logfile.read_text(encoding="utf-8", errors="ignore")
    recent_rows = []
    for m in STEP_RECENT_PATTERN.finditer(text):
        step = int(m.group(1))
        total = int(m.group(2))
        recent = float(m.group(3))
        recent_rows.append((step, recent))
    if not recent_rows:
        for m in ALTERNATE_STEP_PATTERN.finditer(text):
            step = int(m.group(1))
            recent = float(m.group(2))
            recent_rows.append((step, recent))

    # parse lr rows (try to associate with step)
    lr_rows = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        # try to find step in the line
        step_search = re.search(r"(?:Steps:|steps:|Step:|step:|update:|iter:)\s*([0-9]+)", line, re.IGNORECASE)
        lr_val = None
        for pat in LR_PATTERNS:
            mm = pat.search(line)
            if mm:
                try:
                    lr_val = float(mm.group(1))
                    break
                except:
                    pass
        if lr_val is not None:
            if step_search:
                step = int(step_search.group(1))
                lr_rows.append((step, lr_val))
            else:
                # search back a few lines for step
                found_step = None
                for back in range(1,4):
                    if idx-back < 0:
                        break
                    sline = lines[idx-back]
                    ssearch = re.search(r"(?:Steps:|steps:|Step:|step:|update:|iter:)\s*([0-9]+)", sline, re.IGNORECASE)
                    if ssearch:
                        found_step = int(ssearch.group(1))
                        break
                if found_step is not None:
                    lr_rows.append((found_step, lr_val))
                else:
                    # append with None step - will be ignored in plotting
                    lr_rows.append((None, lr_val))

    recent_rows = sorted(recent_rows, key=lambda x: x[0]) if recent_rows else []
    lr_rows = [r for r in lr_rows if r[0] is not None]
    lr_rows = sorted(lr_rows, key=lambda x: x[0]) if lr_rows else []
    return recent_rows, lr_rows


# ------------------------------
# Checkpoint reading
# ------------------------------
def read_checkpoints(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return []
    files = sorted([p for p in ckpt_dir.iterdir() if p.suffix in [".pt", ".pth", ".zip"]])
    ckpt_rows = []
    if not files:
        return ckpt_rows

    if torch is None:
        print("Warning: torch not available; can't inspect checkpoint files. Install PyTorch to inspect.")
        return ckpt_rows

    for f in files:
        try:
            data = torch.load(str(f), map_location="cpu")
        except Exception as e:
            print(f"Warning: failed to load {f.name}: {e}")
            continue
        metric = find_metric_in_checkpoint(data)
        if metric is not None:
            step = extract_step_from_name(f.name)
            if step is None:
                # try to read step inside checkpoint
                if isinstance(data, dict):
                    for k in ("step","update","num_updates","training_step","global_step"):
                        if k in data and isinstance(data[k], (int,float)):
                            step = int(data[k]); break
            if step is None:
                print(f"Skipping {f.name} — no step extracted")
                continue
            ckpt_rows.append((step, float(metric), f.name))
            print(f"Found metric in {f.name}: step={step}, metric={metric}")
        else:
            print(f"No metric found in {f.name}; will consider evaluating if requested.")
    ckpt_rows = sorted(ckpt_rows, key=lambda x: x[0])
    return ckpt_rows


# ------------------------------
# Optional deterministic evaluation (user must fill hooks)
# ------------------------------
def build_model():
    """
    USER ACTION REQUIRED if you want automatic deterministic evaluation.
    Replace with your model constructor and weight-loading logic.
    Must return a PyTorch model object (eval mode).
    """
    raise NotImplementedError("Implement build_model() for automatic checkpoint evaluation.")


def load_weights_into_model(model, ckpt_path: Path):
    """Try common load strategies; adapt to your checkpoint structure."""
    data = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(data, dict):
        if "model_state_dict" in data:
            model.load_state_dict(data["model_state_dict"])
        elif "state_dict" in data:
            model.load_state_dict(data["state_dict"])
        else:
            # try to load data as full state_dict
            try:
                model.load_state_dict(data)
            except Exception as e:
                print("Warning: couldn't load weights automatically:", e)
    else:
        # data is not dict; cannot load
        raise RuntimeError("Checkpoint format not recognized for automatic load.")


def make_eval_env():
    """USER ACTION: return a Gym/Gymnasium env configured for deterministic eval"""
    try:
        import gymnasium as gym
    except Exception:
        import gym
    return gym.make("LunarLander-v3")  # adjust if needed


def act_deterministic(model, obs):
    """
    USER ACTION REQUIRED: implement deterministic action selection for your model.
    Example: if model returns logits, do argmax; if actor returns distribution, pick mode.
    """
    raise NotImplementedError("Implement act_deterministic(model, obs) for auto-eval.")


def evaluate_checkpoints_deterministic(ckpt_dir: Path, out_csv: Path, n_episodes=50):
    """
    Iterate over ckpts, load model, evaluate n_episodes deterministically, save CSV results.
    Requires build_model() and act_deterministic() to be implemented by user.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for deterministic evaluation.")

    results = []
    for f in sorted(ckpt_dir.iterdir()):
        if f.suffix not in [".pt", ".pth", ".zip"]:
            continue
        try:
            model = build_model()
            load_weights_into_model(model, f)
            model.eval()
        except Exception as e:
            print(f"Failed to load {f.name}: {e}")
            continue

        env = make_eval_env()
        returns = []
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            done = False
            total = 0.0
            while True:
                a = act_deterministic(model, obs)
                obs, reward, terminated, truncated, info = env.step(a)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        env.close()
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        step = extract_step_from_name(f.name) or 0
        print(f"Evaluated {f.name}: step={step}, mean={mean:.2f}, std={std:.2f}")
        results.append((step, mean, std, f.name))

    # save csv
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step","mean_return","std_return","ckpt_name"])
        for row in sorted(results):
            writer.writerow(row)
    print("Saved deterministic eval results to", out_csv)
    return results


# ------------------------------
# Plotting
# ------------------------------
def plot_combined(recent_rows, ckpt_rows, lr_rows, out_path: Path, show=False):
    plt.figure(figsize=(12,5))
    ax = plt.gca()

    if recent_rows:
        steps_r, vals_r = zip(*recent_rows)
        ax.plot(steps_r, vals_r, marker="o", linestyle="-", color="orange", label="recent_avg_return (logs)")

    if ckpt_rows:
        steps_c = [r[0] for r in ckpt_rows]
        vals_c = [r[1] for r in ckpt_rows]
        ax.plot(steps_c, vals_c, marker="x", linestyle="", color="tab:blue", label="checkpoint return_avg_mean")

    if lr_rows:
        ax2 = ax.twinx()
        steps_lr, vals_lr = zip(*lr_rows)
        ax2.plot(steps_lr, vals_lr, linestyle="--", color="green", label="learning_rate")
        ax2.set_ylabel("Learning rate")
        # log scale often useful
        try:
            ax2.set_yscale("log")
        except Exception:
            pass

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Return")
    ax.set_title("Training returns (logs) and checkpoint evals (overlay).")
    ax.grid(True)

    lines, labels = ax.get_legend_handles_labels()
    if lr_rows:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2; labels += labels2
    ax.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved combined plot to", out_path.resolve())
    if show:
        plt.show()


# ------------------------------
# Quick inspector for a single checkpoint (helpful)
# ------------------------------
def inspect_checkpoint_one(ckpt_path: Path, top_n=20):
    if torch is None:
        print("PyTorch not available; cannot inspect checkpoint.")
        return
    if not ckpt_path.exists():
        print("Checkpoint not found:", ckpt_path)
        return
    data = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(data, dict):
        print("Checkpoint root is not dict; type:", type(data))
        return
    print("Top-level keys:", list(data.keys())[:80])
    # collect numeric entries
    nums = []
    def collect(d, path=""):
        if isinstance(d, dict):
            for k,v in d.items():
                collect(v, f"{path}.{k}" if path else k)
        elif isinstance(d, (int,float)):
            nums.append((path, float(d)))
        elif isinstance(d, (list,tuple)) and len(d)>0 and all(isinstance(x,(int,float)) for x in d[:20]):
            nums.append((path + " (list_mean)", float(sum(d)/len(d))))
    collect(data)
    nums_sorted = sorted(nums, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    print("Top numeric candidates (path = value):")
    for p,v in nums_sorted:
        print(p, "=", v)


# ------------------------------
# CLI main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs.txt", help="Path to training logs")
    parser.add_argument("--ckpt-dir", default="checkpoints_lunarlander", help="Checkpoint folder")
    parser.add_argument("--out", default="combined_training_plot.png", help="Output image")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("--eval-if-missing", action="store_true", help="If checkpoints lack metrics, run deterministic eval (requires implementing hooks)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Episodes per checkpoint for deterministic eval")
    parser.add_argument("--inspect-one", default=None, help="Inspect a single checkpoint and print numeric candidates (path to file)")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown args:", unknown)

    logf = Path(args.logs)
    ckptdir = Path(args.ckpt_dir)
    outpath = Path(args.out)

    # Step 1: parse logs
    recent_rows, lr_rows = ([], [])
    if logf.exists():
        recent_rows, lr_rows = parse_logs(logf)
        print(f"Parsed {len(recent_rows)} recent_avg_return points and {len(lr_rows)} LR points from logs.")
    else:
        print("Log file not found:", logf)

    # Optional inspect one
    if args.inspect_one:
        inspect_checkpoint_one(Path(args.inspect_one))
        return

    # Step 2: read checkpoints for embedded metrics
    ckpt_rows = read_checkpoints(ckptdir) if ckptdir.exists() else []
    print(f"Found {len(ckpt_rows)} checkpoint metrics embedded.")

    # Step 3: if none found and eval-if-missing enabled, evaluate deterministically
    if not ckpt_rows and args.eval_if_missing and ckptdir.exists():
        print("No embedded checkpoint metrics found. Running deterministic evaluation (you must implement build_model/act_deterministic).")
        try:
            results = evaluate_checkpoints_deterministic(ckptdir, out_csv=Path("ckpt_eval_results.csv"), n_episodes=args.eval_episodes)
            # convert to ckpt_rows format
            ckpt_rows = [(int(r[0]), float(r[1]), r[3]) for r in results]
        except NotImplementedError as nie:
            print("Auto-evaluation not implemented:", nie)
        except Exception as e:
            print("Error during deterministic evaluation:", e)

    # Final: plot combined
    if not recent_rows and not ckpt_rows:
        print("No data to plot. Exiting.")
        return

    plot_combined(recent_rows, ckpt_rows, lr_rows, outpath, show=args.show)

if __name__ == "__main__":
    main()
