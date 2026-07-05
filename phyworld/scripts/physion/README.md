# Physion / Physion++ OCP 评估 pipeline

评估**任意 lewm checkpoint** 在 Physion / Physion++ 上的物理理解能力（OCP,
object-contact prediction — 预测两物体最终是否接触）。

**用途**：lewm 会话在 phyworld 上探索到更好的方法后，把新 ckpt 一行命令迁移过来，
看它学到的表征能不能泛化到真实感的 Physion / Physion++。

## 一键用法

```bash
cd /home/likun-share/junjxu/wm
le-wm/.venv/bin/python phyworld/scripts/physion/eval_physion_suite.py \
    --ckpt <path/to/your_method_epoch_N_object.ckpt> \
    --tag <method_name> --device cuda:2
```

输出：Physion 8 场景 + Physion++ 的 OCP acc/AUC + Physion mean AUC，
存 `reports/physion/eval_<tag>.json`（供跨方法对比）。

常用选项：`--random-init`（未训练 encoder baseline）、`--skip-pp`（只跑 Physion）、
`--ctx 45`（前 N 帧，默认 45≈1.5s）、`--pool meanstd`。

## 评估方法

- 冻结 ckpt 的 encoder → 提前 ~1.5s 帧（45 帧 @30fps，Physion OCP 标准上下文）
  → meanstd 池化 → **5-fold stratified logistic 回归**预测 OCP。
- 本质是**表征线性可分性探针**（test-set 内 CV），快速比较不同方法的表征质量。
- 只用 encoder（不碰 predictor / action），所以**任何 lewm 变体都能评**
  （不同 loss / structured / dynamics / consistency 都行）。

## Baseline —— 新方法要超过的线

| ckpt | Physion mean AUC | Physion++ readout AUC |
|---|---|---|
| **random-init**（未训练 encoder） | **0.607** | 0.534 |
| **zero-shot collision**（phyworld 训） | 0.572 | 0.517 |

**关键结论**：phyworld 合成数据训的表征，zero-shot 迁移到真实感 Physion
**平均反而不如随机 encoder**（0.572 < 0.607）—— phyworld 训练过拟合了合成域、
把随机 ViT 本来的通用视觉特征训坏了。所以判断一个新方法"迁移有效"的标准是：

> **Physion mean AUC 显著 > 0.607**（超过 random-init baseline），才算真的把物理理解迁移过去了。

逐场景看：**Support / Collide** 是 phyworld 训练最容易出信号的（支撑/碰撞物理最接近），
可重点关注这两个场景的 AUC 是否随方法提升。

## 数据位置

- Physion：`/data1/likun-share/junjxu/physion_raw/_core/`（Test-Core，8 场景 × 150 test trial + labels.csv）
- Physion++：`/data1/likun-share/junjxu/physion_raw/physion_plus/readout_ext/`（800 trials，11 个物理属性场景 mass/friction/bouncy/deform）

## 脚本

| 文件 | 作用 |
|---|---|
| `eval_physion_suite.py` | **一键入口**（Physion 8 场景 + Physion++ 全套） |
| `ocp_probe.py` | 单独的 Physion 探针（可跑单场景 / 加 --random-init） |
| `physion_plus_probe.py` | 单独的 Physion++ 探针 |

## 注意

- ckpt 必须是 **pickled JEPA 对象**（`ModelObjectCallBack` 用 `torch.save(model)` 存的
  `*_object.ckpt`），不是 state_dict。
- 提特征约 5–8 分钟（1200 + 800 视频）；挑一张空闲 GPU（`--device cuda:N`）。
- 这是快速探针，非 Physion 官方 readout-split 协议；要更严格的标准数字可后续扩展
  （Physion++ 的 test split 已下载在 `physion_plus/test_data.zip`，解压后可做 readout 训 / test 测）。
