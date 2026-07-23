# NOTE → Physion/Physion++ session:论文五章重构后的实验缺口清点(2026-07-16)

> **✅ P0 已完成(2026-07-23)**:probe / probe+slot 两臂 ×3 种子已在 Physion++ 跑完,三 mass 场景 3.9–11.9× 差于 FR、区间分离;论文 tab:pp 已加两列、§4.1 与 limitation 已升级为三族全覆盖。以下为历史记录。


**背景**:06_storyline 已重构为五章交付版(动机单线 = "物理注入对 latent WM 是否有效");其 §4.1 现在**明确宣称"Physion++ 上注入同样失败"**。本 NOTE 是对你 session 已跑实验的完整清点 + 缺口,**按优先级排列**。清点基于 `raw_data/runs/physionpp/` 的 67 个 log(2026-07-15 迁移归档)。

## 已有覆盖(够用,勿重跑)

| 实验 | 文件 | 状态 |
|---|---|---|
| TF vs FR 三种子 | `eval_pp_{tf,fr}_s{3072,1234,42}.log` | ✅ 8.3× headline,三种子零重叠 |
| num_preds 阶梯 | `eval_pp_fr_{np20,np20sc,np28sc}_e20*.log` | ✅ h64 1/19 无拐点 |
| init 消融三种子 | `eval_pp_init_{scratch,cube,pusht}{,_s1234,_s42}.log` | ✅ |
| 物理臂(slot/cons 两族) | `eval_pp_{struct,cons,consacc}{,2,_e20}.log` | ✅ 逐场景 3–10× 差于 FR(单种子,差距远超噪声带,可辩护) |
| held-out OOD | `eval_np28sc_{ho_ood,full_seen}.log` | ✅ |
| 增广反转 | `eval_pp_fr_app{03,05}_e20.log` | ✅(已从主线撤下,留 detail/附录) |
| presence probe | `probe_real_emb_pp_fr.log` | ✅(数据弱,已决定不进论文) |

## 缺口(按优先级)

### P0 — probe(deep-sup)臂在 Physion++ 上没跑 ⚠️ 最可能被审稿人点名
- **问题**:5 家族里,Physion++ 只测了 ①slot(struct)和 ④consistency 两族。**②probe(深监督)= deep-sup 2504.03861 = 文献最强竞品**,目前只在 phyworld 被否。审稿人一句"maybe soft supervision works on photorealistic data"就能打回来。
- **要补**:`train_pp_probe`(probeF2 recipe,λ 与 phyworld probeF2 一致)+ `eval_pp_probe_e20`;判决口径 = 逐场景 rollout nMSE vs `eval_pp_fr_e20.log`(FR: mass_dominoes 0.058 / friction_platform 0.041 / mass_collision 0.083 / mass_waterpush 0.122)。单种子(3072)即可——若差距像 struct/cons 一样 3–10×,种子噪声不构成威胁。1 卡 1 run。
- 若结果**probe 在 Physion++ 反而不差**:也**必须**如实报——那是"软注入在照片级仿真的边界条件",发现价值不低于全负。

### P1 — struct/cons 的三个版本口径不一致,须钉死判决版
- **问题**:`eval_pp_struct.log` / `eval_pp_struct2.log` / `eval_pp_struct_e20.log` 并存,逐场景 n= 不同(如 mass_dominoes 1435 vs 525)——分区/子集/epoch 不一致。storyline §4.1 现引 `_e20` 版(与 `eval_pp_fr_e20` 同口径)。
- **要做**:在 `reports/physion/physionpp_ood_longhorizon.md` 里**写死**哪个是判决版本、其余两个版本是什么口径(不同 epoch?不同 eval 子集?),防止论文引错。不用重跑。

### P1 — (可选,有空卡再跑)物理臂补第二种子
- struct/cons/consacc 单种子;差距 3–10× 远超 FR 三种子噪声(std 0.01–0.07),**单种子可辩护**,storyline 已如实标注。若有空卡,优先补 `train_pp_struct_s1234` 一个即可(最接近 FR 的臂,最值得加固)。

### P2 — (可选)TF + 长 horizon 对照
- np 阶梯全是 FR;若审稿人问"TF 配 np28 会不会也好",目前无数据。可选补 `train_pp_tf_np28sc`。不阻塞投稿——论文 claim 是"FR 内部单调性 + TF/FR 同 np 对照",已成立。

### 不用跑的(明确排除,别浪费卡)
- posvel 臂上 Physion++:posvel 的唯一正例依赖"抛体速度可外推",Physion++ 无对应可外推量,跑了也是预期全负;写进 limitation 即可。
- dyn/label-free/grounded 上 Physion++:30 格已在 phyworld 把这三族否掉,Physion++ 上 slot+cons+probe(P0 补完后)三族同向即足以支撑"照片级仿真同样成立"。

## 给 DINO-WM session 的对齐信息(顺带)
storyline §4.4 给 DINO-WM 留了三格数据位:① TF vs FR(至少 1 域,nMSE+倍数);② structpos 一臂 vs FR baseline(一个判决格);③ 可选 presence probe ρ。填法与措辞升级规则见 [06_storyline.md §4.4](06_storyline.md)。**若结果不同向,如实写边界条件,勿硬凹。**
