# M3 验证记录:确定性与扩散生成

日期:2026-09-13。对应方案 §7/§9.1/§11-M3 与 `docs/plans/interaction_retarget_impl_plan.md`。
代码:`src/ireset/{data,model,diffusion,train}/`、`scripts/{precompute_m3_cache,train_generation,eval_generation}.py`,
配置:`configs/model/*.yaml`、`configs/train/pilot.yaml`,缓存与产物:`derived/data/m3_cache/`、`derived/models/`、`derived/data/validation/generation_eval.json`。

## 1. 数据层

- **schema(§4.3)**:`ireset/data/schema.py` 实现完整 `ReferenceCondition`(source/target 物体轨迹、ground plane、接触区间、`world_from_clip`)、`InteractionClip`、`RobotTrajectory`;`to_repair_condition()` 适配到修复层精简条件,两套接口不漂移。
- **clip 坐标系(§5.1)**:由参考人第 0 帧骨盆水平位置 + 髋→肩朝向 yaw 定义(绝不是目标 G1 真值);一致的水平平移/绕 z 旋转不变性有测试。条件通道:52 关键点(源物体系+clip 系)、目标箱体轨迹(clip 系 t+6D 旋转)、4 末端接触调度、静态箱体 AABB 半长与地面 —— 均不含目标根/关节。
- **窗口与划分卫生**:F=60、stride=30,窗口只取自 `derived/data/split/<split>.json` 锁定清单(1,826 NPZ 全部分配;train/val/test 两两不相交有测试)。短于 F 的 clip 与超限教师被显式排除并记录,不静默填充。
- **预计算缓存**(`derived/data/m3_cache/`):1,826 个 NPZ 的参考侧接触调度(双阈值+最短时长,验收级 hull 度量)逐文件缓存;train 侧 clip 系根统计 258,462 帧:mean=[−0.031, 0.118, 0.716] m,std=[0.428, 0.449, 0.080] m。
- **教师标签限位核查(§7.2)**:全部 1,826 个文件、逐帧 29 关节对锁定限位**零越界**(max margin −0.0087 rad),无裁剪、无剔除 —— "剔除或修正超限教师标签并记录"条款当前不触发,记录在案。

## 2. Token 与参数化(§7.1/7.2)

- 38 维联合状态 = clip 系根位置(训练分组标准化)3 + 根旋转 6D + 关节限位映射 29;每帧 30 token(1 根 + 29 关节),根/关节独立输入输出投影,根头输出线性(不裁剪),关节头输出 `2σ(logits)−1` 仅作用于**干净预测**。
- 旋转 6D Gram-Schmidt 正交化,退化(零向量)/非有限显式标记 → 采样端有界重采样,不发坏姿态。编解码往返误差 <1e-5 m / <1e-9 rad;torch 与 numpy 编码一致 <1e-6。
- 扩散:diffusers `DDPMScheduler`/`DDIMScheduler`,`prediction_type="sample"`(x0-prediction)、`clip_sample=False`、DDIM thresholding 关闭;训练 1000 步、采样 25 步起步。**适配测试**:完美 x0 模型下 DDIM 采样末步精确还原目标(末步 prev-SNR=1)、`add_noise` 与闭式一致、固定种子复现、调度器配置断言。

## 3. 模型与损失(§7.1/7.3)

- 主干:30 token/帧 transformer,d_model=240、4 层、6 头、FFN×4,时间注意力窗口 31 帧(布尔 keep-mask,窗口语义有测试);条件交叉注意力到 [2+F·54] 条件 token。同一主干供扩散(带时间嵌入)与确定性(无噪声/无采样环)使用。
- 注意力实现为显式数学路径(softmax(QK^T/√d))而非 `F.scaled_dot_product_attention` 快速后端:本机系统 Python 无 `Python.h`,Triton 驱动编译失败;按用户授权引入 uv 托管 CPython 3.12.13(自带头文件)后 Triton 可用,但保留显式实现以保证数值语义透明(此决定记录于 §6 备注)。
- 损失(权重 `configs/train/pilot.yaml`,单位显式):干净状态重建(pos/rot6d/joint 分组)+ 根旋转测地(rad²)+ 教师 FK 15 关键点误差(m²,clip 系,可微 torch FK)+ 手接触偏差(6 手部 site 对箱 AABB SDF,接触帧,m²,训练代理)+ 时序速度(帧差)。参考拉普拉斯 E_L 为**计划内消融,本轮未实现**。几何损失不替代修复验收。

## 4. 训练 pilot(§9.1)

> 注:本节数字为首轮(接触损失带 §8 所述坐标变换 bug)。修复后重训数字见 §8;`derived/models/*/summary.json` 已被修复后运行覆盖,首轮产物归档于 `derived/models/*_v1_contactbug/`。

- 单卡 RTX 3090,micro-batch=4、梯度累积=4(有效 16)、FP32、AdamW(lr 1e-4,wd 1e-5)、梯度裁剪 1.0,seed 20260913;train 6,431 窗 / val 233 窗(60 个 val 文件上限),4000 优化步(≈10 epoch)。
- 实测(2026-09-13,`derived/models/*/summary.json`):

| 模型 | 步/墙钟 | ms/步 | 峰值显存 | best val loss | 收敛(val) |
|---|---|---|---|---|---|
| 扩散(x0-pred) | 4000 / 64.2 min | 858.8 | 5.47 GiB | 0.0958 | 0.695 → 0.304(1k)→ 0.103(4k) |
| 确定性 | 4000 / 61.3 min | 803.0 | 5.47 GiB | 0.2459 | 1.183(500)→ 0.312(1.5k)→ 0.25(4k) |

- 完整训练成本估计:pilot 未饱和(4k 步损失仍在缓降)。同配置下 ~15 epoch ≈ 1.6 h/模型;后续按验证曲线决定是否延长,不预承诺 GPU 天数。训练曲线见 `derived/models/*/loss_history.json`。

## 5. 验收实验(§10.2/§10.4)

> 注:同 §4,本节为首轮(带 §8 变换 bug)数字;修复后刷新见 §8。

设置:val 分组 8 个 original(120 帧),扩散 DDIM-25 与确定性各生成一条(拼接 60/30),教师 σ∈{0.02,0.05} 扰动作"真实预测误差"对照,同修复器同计时。产物 `derived/data/validation/generation_eval.json`。

**保真度(对教师,120 帧宏平均)**:

| 模型 | 根 RMSE | 根测地角 | 关节 MAE | MPJPE(世界=物体系) | 接触 F1(macro,均值) |
|---|---|---|---|---|---|
| 扩散 DDIM-25 | 5.1 cm | 10.1° | 0.154 rad | 8.9 cm | 0.485 |
| 确定性 | 5.9 cm | 10.8° | 0.398 rad | 11.3 cm | 0.485 |

两类模型均输出**完整状态**(根 7 + 29 关节,四元数单位范数、关节限位满足,sigmoid 头保证),退化窗口 0(8×2 段全部一次采样成功)。MPJPE 8.9cm 与 σ≈0.03–0.05 关节噪声的误差量级相当,即已处于"真实预测误差"区间。

**裸输出独立终检**:16/16 拒绝,失败项一致为 `penetration, foot_support, hand_grasp` —— 裸生成不含可行性,由修复器负责,与管线设计一致。

**修复后接受率(整段)**:扩散 0/8、确定性 0/8、教师噪声 σ=0.02 0/8、σ=0.05 0/8,**全部 `rejected_frame`**(逐帧顺序修复失败,无最终终检误受)。教师基线同样全拒证明:这不是生成质量问题,而是 M2 已记录的"真实搬运片段在支撑/抓取模型下不可修复"(教师脚滑 ~0.3 m/s、深穿透)阻塞了**一切**输入的整段接受——修复接受率的前置是 M2 收尾与教师数据重生成/清洗(方案状态摘要已记)。M3 验收要求的"比较裸输出与真实预测误差下的修复质量"已执行:同一分母下生成输出与教师噪声输出进入同一失败类别,且修复器在扩散输出上的每帧耗时(93–1034 ms/f)高于确定性(88–497 ms/f),与扩散输出逐帧噪声更大一致。

**拼接边界**:8 段 × 2 模型,接缝关节速度 max 0.95–4.3 rad/s,全部**低于**同段内部 p95(2.2–5.5 rad/s)——边界不是速度尖峰来源,SLERP 融合未制造伪运动。

**生成计时(3090,120 帧)**:扩散 DDIM-25 ≈ 8–10 ms/帧(3–4 窗 ×25 次前向,非单次网络调用);确定性 ≈ 0.5 ms/帧。DDIM 步数消融(3 段):5/10/25 步 MPJPE 9.8/9.7/9.8 cm、2.6/3.5/8.3 ms/帧、F1 0.544/0.551/0.564 —— 质量对步数不敏感,10 步为性价比点;正式协议仍按锁定 25 步报告。

**结论**:M3 三项工作项完成;验收条款"两类模型输出完整状态并比较修复质量"达成且如实报告 0% 接受(含教师基线);整段接受率 ≥90% 的输出目标(§10.1)当前被修复器对真实片段的 0% 接受阻塞,转 M2 收尾 + M3 数据重生成路径。

## 6. 工程备注

- 环境:diffusers 0.40.0、pyyaml(显式声明)经 `uv add` 引入;`.venv` 重建于 uv 托管 CPython 3.12.13(原系统 3.12.3 无 dev 头文件,torch/triton 的 SDPA 快速后端无法编译驱动;重建后 torch 2.14.0+cu126 GPU 前向/反向验证通过)。`uv.lock` 已更新并版本管理。
- pytorch_kinematics FK 不支持 3 维批量:损失内拉平 B·F 后前向。
- scipy 四元数为 xyzw,项目存 wxyz —— 拼接 slerp 与评测测地角处显式转换(有测试)。
- 教师 NPZ 的 rot/trans 增强与人/箱一致变换(M0 已证),故 source/target 物体轨迹当前同取自 NPZ `qpos[36:43]`;schema 双字段为未来"物体单独移动需重生成教师"的场景保留。

## 7. 测试

`uv run --locked pytest tests/ -q`:76 项全过(新增 `tests/test_generation.py` 17 项:token 往返/限位映射/退化检测、划分不相交、clip 系不变性、条件器与主干形状、窗口 mask 语义、DDIM 完美模型复原、种子复现、add_noise 闭式、调度器配置断言、拼接恒等/slerp 中点/边界诊断、损失有限且可微、生成器烟测)。第三轮修复后全套 **78 项**(+2,见 §8)。

## 8. 审阅修复轮(2026-09-13 第三轮)

独立代码审阅提出接触损失坐标变换疑点,判定属实,并连带发现一个几何层方向反转:

**判定证据**(脚本对拍,教师 val 24 窗口,教师状态 + MuJoCo FK + M1 已验证的 `world_to_box_geom` 为地面真值):

- losses.py 手接触项的 clip→box 变换 `p_body = R_box_clip @ p_rel` **缺转置**(`R_box_clip` 为 box→clip,应取其转置):修复后与地面真值最大偏差 **5.9e-8 m**,修复前最大偏差 **0.104 m**(逐点、随箱体旋转系统性变化);
- 修复后,AABB 表面点(geom 系构造)经完整链的 |SDF| 残差 ~1e-16 m;修复前 2–6 cm;
- bug 难以从训练曲线察觉的原因:教师接触帧上 |SDF| 均值在两链下几乎相同(0.087 vs 0.088 m;AABB 代理误差 + 教师未真接触本身 ~0.09 m,hull 参考距离 0.147 m),故 `contact_m2` 平台量级不变(记录值 ~0.012–0.016 ≈ 教师 SDF² 水平 0.0077 同量级)——但逐帧梯度方向错误,模型学到了"在错误旋转下的箱面上放手",与 FK/重建损失冲突产生系统性噪声梯度。

**修复**(共三处):

1. `train/losses.py`:变换提为 `LossAssembler.hand_points_box_geom`(einsum `"btji"` 转置形式,注释锚定 `world_to_box_geom`/MuJoCo `geom_xmat` 约定);接触项调用该方法。
2. `geometry/distance_adapter.py` `box_geom_to_world`:第一处 einsum `"ji"`→`"ij"`(geom→body 应为 `R_g @ p + x_g`;原方向翻转导致与 `world_to_box_geom` 不互逆,箱面点往返差 0.26 m)。**该函数此前无任何调用点、无测试覆盖,未污染 M1–M3 已有实验结论**;本次修复并补往返回归测试,防后用(M4 锚点/评测)。
3. `model/backbone.py` `Attn` 类注释过时("Triton 驱动编译失败"):迁移 uv 托管 CPython 3.12.13 后 Triton/SDPA 已可用,保留显式实现的理由更新为 pilot 数值可复现,SDPA 切换列为后续优化。

**回归测试**(新增 2 项,全套 **78 项全过**):`test_contact_chain_matches_mujoco_box_frame`(真实 G1 模型 + 随机箱位姿,损失链 vs MuJoCo FK + `world_to_box_geom`,阈值 1e-4 m)、`test_box_geom_world_roundtrip`(随机箱位姿下 geom↔world 往返 <1e-10,并 guard `R_g` 非对称防测试退化)。

**重训与验收刷新**(同配置 seed 20260913、4000 步;首轮产物归档为 `derived/models/*_v1_contactbug` 留对比):

| 模型 | 步/墙钟 | ms/步 | 峰值显存 | best val(v1 bug 对照) |
|---|---|---|---|---|
| 扩散 | 4000 / 62.7 min | 812.5 | 5.47 GiB | **0.0949**(0.0958) |
| 确定性 | 4000 / 60.6 min | 805.9 | 5.47 GiB | **0.2491**(0.2459) |

验收刷新(同 §5 协议:val 8 个 original、120 帧、DDIM-25、seed 7):

| 模型 | 根 RMSE | 根测地角 | 关节 MAE | MPJPE | 接触 F1(macro) | 修复接受 | 修复 ms/帧 |
|---|---|---|---|---|---|---|---|
| 扩散 | 5.6 cm | 11.3° | 0.151 rad | 9.0 cm | 0.594 | 0/8 | ~491(max 1017) |
| 确定性 | 5.2 cm | 9.9° | 0.403 rad | 10.9 cm | 0.387 | 0/8 | ~259(max 467) |

**与首轮的差异解读**:保真度指标在 8 段小样本噪声范围内持平(val 0.0949 vs 0.0958、0.2491 vs 0.2459;MPJPE 9.0 vs 8.9、10.9 vs 11.3 cm)——符合预期:接触项为五项损失之一(权重 0.5),pilot 阶段主导信号是重建+FK。修复的意义在**梯度方向正确**(对拍 5.9e-8 m)而非损失量级变化:val contact_m2 新旧同为 ~0.010–0.012,即正误两链下教师水平接近,这正是 bug 隐蔽、训练曲线无法暴露它的原因。其余结论不变:教师噪声基线与两模型修复接受率仍全 0/8、同因 `rejected_frame`(M2 教师可行性阻塞);接缝关节速度全部低于内部 p95;生成计时扩散 8.8 ms/帧、确定性 0.5 ms/帧。修复耗时均值较首轮下降(扩散 ~491 vs 93–1034、确定性 ~259 vs 88–497 ms/帧),与接触项不再把模型推向不可行姿态的方向一致,但 8 段样本不作强归因。接触 F1 macro 波动(扩散 0.485→0.594、确定性 0.485→0.387)视为小样本波动,不下方向性结论。

