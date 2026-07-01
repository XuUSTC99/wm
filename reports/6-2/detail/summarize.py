"""Parse 6-2/logs/rollout_*.log -> reports/6-2/piwm_uniform_collision_results.md.
Builds the 4-arm comparison (baseline / pos-only / pos+vel / mf4) per domain:
vx, vy decode rho (K=1 & K=4), latent cos by partition + by horizon."""
import re, time
from pathlib import Path

LOG = Path("/home/qlib/am/wm/reports/6-2/logs")
OUT = Path("/home/qlib/am/wm/reports/6-2/piwm_uniform_collision_results.md")
ARMS = [("baseline", "baseline"), ("pos-only(训练单帧)", "posonly"),
        ("pos+vel(训练单帧)", "posvel"), ("mf4(训练多帧)", "mf4")]
DOMAINS = [("uniform_motion", "uniform_motion"), ("collision", "collision")]
PARTS = ["ID", "r/m-OOD", "v-OOD", "both-OOD"]


def parse(logpath):
    """Return dict with K1[part]=(vx,vy), K4[part]=(vx,vy), cos_part[part], cos_h[h]."""
    if not logpath.exists():
        return None
    txt = logpath.read_text()
    d = {"K1": {}, "K4": {}, "cosP": {}, "cosH": {}}
    for line in txt.splitlines():
        m = re.match(r"\s*PRED-K4\s+(\S+)\s+.*vel0ρ=([+\-0-9.]+)\s+vel1ρ=([+\-0-9.nan]+)", line)
        if m:
            d["K4"][m.group(1)] = (m.group(2), m.group(3)); continue
        m = re.match(r"\s*PRED\s+(ID|r/m-OOD|v-OOD|both-OOD)\s+.*vel0ρ=([+\-0-9.]+)\s+vel1ρ=([+\-0-9.nan]+)", line)
        if m:
            d["K1"][m.group(1)] = (m.group(2), m.group(3)); continue
        m = re.match(r"\s*(ID|r/m-OOD|v-OOD|both-OOD)\s+n=\s*\d+\s+cos=([+\-0-9.]+)", line)
        if m:
            d["cosP"][m.group(1)] = m.group(2); continue
        m = re.match(r"\s*h=\s*(\d+)\s+n=\s*\d+\s+cos=([+\-0-9.]+)", line)
        if m:
            d["cosH"][int(m.group(1))] = m.group(2)
    return d


def col(d, key, part, idx=None):
    if d is None or part not in d.get(key, {}):
        return "—"
    v = d[key][part]
    return v if idx is None else v[idx]


lines = [f"# PIWM Deep-Supervision — uniform_motion + collision（follow parabola）",
         f"\n**生成时间**：{time.strftime('%Y-%m-%d %H:%M')}（由 run_piwm.sh 自动汇总）",
         "\n协议同 [5-26/piwm_deepsup_results.md](../5-26/piwm_deepsup_results.md)：ID-only 1k 训练，"
         "probe on FT loss，rollout 评估在全 OOD eval 集。4 臂 = baseline(无probe) / pos-only(训练单帧) "
         "/ pos+vel(训练单帧) / mf4(训练多帧 frames=4)。velocity 监督列：uniform=action(速度), collision=state(速度)。",
         "\n> ⚠️ 所有解码列用推理 K=4；'单帧/多帧'指训练时 probe 吃几帧。\n"]

data = {}
for dom, dtag in DOMAINS:
    data[dom] = {arm: parse(LOG / f"rollout_{dom}_{atag}.log") for arm, atag in ARMS}

for dom, _ in DOMAINS:
    lines.append(f"\n## {dom}\n")
    dd = data[dom]
    # vx (vel0) K=4
    lines.append("### vx 解码 ρ（K=4）\n")
    lines.append("| partition | " + " | ".join(a for a, _ in ARMS) + " |")
    lines.append("|" + "---|" * (len(ARMS) + 1))
    for p in PARTS:
        row = [col(dd[a], "K4", p, 0) for a, _ in ARMS]
        lines.append(f"| {p} | " + " | ".join(row) + " |")
    # vy (vel1) K=4
    lines.append("\n### vy 解码 ρ（K=4）\n")
    lines.append("| partition | " + " | ".join(a for a, _ in ARMS) + " |")
    lines.append("|" + "---|" * (len(ARMS) + 1))
    for p in PARTS:
        row = [col(dd[a], "K4", p, 1) for a, _ in ARMS]
        lines.append(f"| {p} | " + " | ".join(row) + " |")
    # latent cos by partition
    lines.append("\n### latent cos by partition\n")
    lines.append("| partition | " + " | ".join(a for a, _ in ARMS) + " |")
    lines.append("|" + "---|" * (len(ARMS) + 1))
    for p in PARTS:
        row = [col(dd[a], "cosP", p) for a, _ in ARMS]
        lines.append(f"| {p} | " + " | ".join(row) + " |")
    # latent cos by horizon
    lines.append("\n### latent cos by horizon\n")
    lines.append("| h | " + " | ".join(a for a, _ in ARMS) + " |")
    lines.append("|" + "---|" * (len(ARMS) + 1))
    for h in [1, 2, 4, 8, 16, 28]:
        row = [dd[a]["cosH"].get(h, "—") if dd[a] else "—" for a, _ in ARMS]
        lines.append(f"| {h} | " + " | ".join(str(x) for x in row) + " |")

lines.append("\n---\n\n## 关键问题（待人工解读）\n")
lines.append("1. parabola 上的结论是否复现：单帧 probe 砸 vx 高速 OOD？mf4 是否修回 + 长程 cos 最佳？")
lines.append("2. collision（action=加速度、2 球）与 uniform（vx 恒等）是否表现一致？")
lines.append("3. baseline 的 K=4 vx 在高速 OOD 是否仍是'位置差分白嫖'的强基线？")
lines.append("\n日志：`reports/6-2/logs/`（train_*.log / rollout_*.log / orchestrator.log）")

OUT.write_text("\n".join(lines))
print(f"wrote {OUT}")
