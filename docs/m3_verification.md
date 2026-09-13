# M3 验证记录:确定性与扩散生成

日期:2026-09-13。对应方案 §7/§9.1/§11-M3 与 `docs/interaction_retarget_impl_plan.md`。
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

- 单卡 RTX 3090,micro-batch=4、梯度累积=4(有效 16)、FP32、AdamW(lr 1e-4,wd 1e-5)、梯度裁剪 1.0,seed 20260913;train 6,431 窗 / val 233 窗(60 个 val 文件上限),4000 优化步(≈10 epoch)。
- 实测(2026-09-13,`derived/models/*/summary.json`):

| 模型 | 步/墙钟 | ms/步 | 峰值显存 | best val loss | 收敛(val) |
|---|---|---|---|---|---|
| 扩散(x0-pred) | 4000 / 64.2 min | 858.8 | 5.47 GiB | 0.0958 | 0.695 → 0.304(1k)→ 0.103(4k) |
| 确定性 | 4000 / 61.3 min | 803.0 | 5.47 GiB | 0.2459 | 1.183(500)→ 0.312(1.5k)→ 0.25(4k) |

- 完整训练成本估计:pilot 未饱和(4k 步损失仍在缓降)。同配置下 ~15 epoch ≈ 1.6 h/模型;后续按验证曲线决定是否延长,不预承诺 GPU 天数。训练曲线见 `derived/models/*/loss_history.json`。

## 5. 验收实验(§10.2/§10.4)

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

`uv run --locked pytest tests/ -q`:76 项全过(新增 `tests/test_generation.py` 17 项:token 往返/限位映射/退化检测、划分不相交、clip 系不变性、条件器与主干形状、窗口 mask 语义、DDIM 完美模型复原、种子复现、add_noise 闭式、调度器配置断言、拼接恒等/slerp 中点/边界诊断、损失有限且可微、生成器烟测)。
