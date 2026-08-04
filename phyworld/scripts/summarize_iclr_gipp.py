"""汇总 ICLR GIPP 长程与 OOD 评测日志。"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "runs/iclr_gipp/eval"
OUT_DIR = ROOT / "reports/iclr_gipp"
PART_RE = re.compile(r"^\s*(ID|r/m-OOD|v-OOD|both-OOD)\s+n=\s*(\d+)\s+cos=([+-]?[0-9.]+)\s+nMSE=([0-9.]+)")
HOR_RE = re.compile(r"^\s*h=\s*(\d+)\s+n=\s*(\d+)\s+cos=([+-]?[0-9.]+)\s+nMSE=([0-9.]+)")


def parse(path):
    rows, section = [], None
    for line in path.read_text(errors="replace").splitlines():
        if "latent fidelity (pred vs real emb)" in line:
            section = "partition"
        elif "latent fidelity vs horizon" in line:
            section = "horizon"
        elif line.startswith("---") and section == "horizon":
            section = None
        m = PART_RE.match(line) if section == "partition" else None
        if m:
            rows.append(dict(run=path.stem, type="分区", slice=m[1],
                             n=int(m[2]), cos=float(m[3]), nmse=float(m[4])))
        m = HOR_RE.match(line) if section == "horizon" else None
        if m:
            rows.append(dict(run=path.stem, type="步长", slice=f"h={int(m[1])}",
                             n=int(m[2]), cos=float(m[3]), nmse=float(m[4])))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(LOG_DIR.glob("*.log")):
        rows.extend(parse(path))
    csv_path = OUT_DIR / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "type", "slice", "n", "cos", "nmse"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    keys = ["both-OOD", "v-OOD", "r/m-OOD", "h=1", "h=8", "h=16", "h=28"]
    lookup = {(r["run"], r["slice"]): r for r in rows}
    runs = sorted({r["run"] for r in rows})
    lines = [
        "# ICLR GIPP 自动结果汇总", "",
        "本表由完整 rollout 日志自动生成；空项表示对应评测尚未完成。", "",
        "| 实验 | " + " | ".join(f"{k} nMSE↓" for k in keys) + " |",
        "|---|" + "|".join("---:" for _ in keys) + "|",
    ]
    for run in runs:
        vals = []
        for key in keys:
            row = lookup.get((run, key))
            vals.append(f'{row["nmse"]:.4f}' if row else "—")
        lines.append("| " + run + " | " + " | ".join(vals) + " |")
    lines += ["", f"已解析 {len(runs)} 个实验、{len(rows)} 个指标单元。", ""]
    tmp = OUT_DIR / ".AUTO_RESULTS.md.tmp"
    tmp.write_text("\n".join(lines))
    tmp.replace(OUT_DIR / "AUTO_RESULTS.md")
    print(f"汇总完成：{len(runs)} 个实验，{len(rows)} 个指标单元")


if __name__ == "__main__":
    main()
