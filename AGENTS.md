# 项目开发约定与上下文

本文件适用于整个项目。信息整理日期:2026-09-11。硬件占用、资源数量和完成状态是检查时的快照,后续以实际文件及验证记录为准。

## 项目目标与范围

- 项目 `ireset` 面向人类-箱体交互动作到 Unitree G1 的重定向,当前仅支持一套固定的 **G1 29-DoF** 资产配置。
- MVP 为平地、单一刚性箱体、低速双手搬运;物体几何和轨迹已知。生成结果包含浮动基座位姿与 29 维关节角。
- 主路线为条件扩散生成、顺序 QP 修复、整段独立验收;确定性预测器使用相同修复器作为对照。SOCP 为可选消融。
- 当前不实现跨机器人泛化、灵巧手、regrasp、复杂地形、动力学保证或真机部署。几何接触不等于可承重抓握,运动学通过不等于 RL 可跟踪。
- 完整设计、阶段依赖和验收标准见 [实现方案](docs/plans/interaction_retarget_impl_plan.md)。资源细节见 [资源登记](resources/README.md),版本与哈希见 [资源清单](resources/manifest.json)。

## Python 与环境管理

- 固定 **Python 3.12.x**,使用 **uv** 管理环境和依赖。`.python-version` 为 `3.12`,`pyproject.toml` 要求 `>=3.12,<3.13`。
- 使用项目 `.venv`;当前已验证解释器为 uv 托管 CPython 3.12.13(2026-09-13 起;原系统 3.12.3 无 dev 头文件,torch/triton 快速后端无法编译,经用户授权迁移)。不要修改系统 Python 或混入其他项目环境。
- 依赖通过 `uv add` / `uv remove` 维护,保留并版本管理 `uv.lock`。同步使用 `uv sync --locked`,项目命令使用 `uv run --locked`。
- 按当前实现步骤添加依赖,不要预先安装整个技术栈。记录 PyTorch CUDA wheel 来源、依赖版本及运行验证结果。
- `.venv`、缓存和大型下载不纳入代码版本管理。上游工具需要独立依赖时单独用 uv 管理,不得破坏主项目的 Python 3.12 约束。

## 成熟依赖优先

- 标准算法优先复用维护活跃的成熟库。实现前检查现有依赖、上游 API 和项目真实需求,尽量不重复造轮子。
- 运动学优先 `pytorch_kinematics` 和 MuJoCo;旋转/插值、Delaunay、稀疏矩阵优先 SciPy;网格预处理优先 trimesh。
- DDPM/DDIM 调度优先 Diffusers;注意力、混合精度和数据加载优先 PyTorch 原生能力;求解优先 OSQP,可选 CVXPY + Clarabel。
- 保持适配层简单,明确单位、坐标系、四元数排列、张量形状、设备与精度。没有实际需求时不创建通用框架或复制库内部实现。
- 自定义代码集中在人-G1 数据配对、交互语义、接触状态和锚点、约束组装、SCP 流程、质量筛选及独立验证。
- 引入依赖前核查 Python 3.12 兼容性、维护状态、许可证、安装/存储成本和所需 GPU/autograd 支持。
- 成熟库确有能力缺口时,记录缺口并实现最小补充,用可信实现或解析用例对拍。成功 import 不代表实际 API、梯度或数值行为已验证。

## 本机计算资源

| 项目 | 已检查配置 |
|---|---|
| CPU | Xeon Gold 5218R,20 物理核/40 逻辑线程,单 NUMA 节点 |
| 内存 | 系统可见约 125GiB |
| GPU | 单张 RTX 3090,24GiB 显存 |
| 驱动 | NVIDIA 580.178.04;驱动显示 CUDA 13.0 支持上限 |
| 系统 | Ubuntu 24.04.5 LTS,x86_64 |
| 磁盘 | 项目位于 NVMe 文件系统,资源下载前最近快照约 210GiB 可用 |

- `nvidia-smi` 中 CUDA 版本不等于已安装 Toolkit/PyTorch runtime。安装后需验证 GPU 前向/反向和实际 runtime。
- 单卡 pilot 从 micro-batch=4、梯度累积=4 起步,按峰值显存调整;修复与数值对拍使用 float64。训练时间以实测为准。
- DataLoader workers=4、独立 clip 处理进程数=4 可作为起点,限制每进程 BLAS/OpenMP 线程以免过度订阅。
- 项目与 `/tmp` 共用文件系统。大规模下载或训练前重新检查空间,控制检查点和缓存数量,不预保存重复滑窗。

## 已有资源与访问规则

以下路径相对项目根目录,外部软链接目标除外。

| 资源 | 位置与状态 |
|---|---|
| 处理版 OMOMO | `resources/data/OMOMO_new`,软链接指向 `/home/c112/codes/bill/smrt-data/unzipped/OMOMO_new` |
| OMOMO 数量 | 4,421 个 PT,其中 largebox 文件 329 个;约 1.79GiB,不是独立演示数的确认结果 |
| 用户提供的 G1 | `resources/robots/g1_holosoma`,含普通手、spherehand、带箱体场景及 mesh |
| 配套箱体 | `resources/robots/largebox`,已补齐 OBJ/URDF/XML 和原场景引用 |
| 公开教师轨迹 | `resources/data/omniretarget/robot-object`,1,826 个 largebox NPZ |
| 教师数据配套模型 | `resources/data/omniretarget/models`,与当前 Holosoma 模型分开保存 |
| 原始归档和数据卡 | `resources/data/omniretarget/robot-object.zip` 及该目录 README/可视化脚本 |
| 上游代码 | `third_party/holosoma/src/holosoma_retargeting`,稀疏 checkout,含配置、加载代码和人体高度元数据 |

- 源数据、原始模型和下载归档保持不变。软链接不提供写保护,写入链接会修改项目外源数据;只通过链接读取,派生结果写入独立目录。
- 模型修正、重采样、标签和数据转换需记录来源、参数、版本和校验值。不要静默修改用户提供的模型或把上游资产直接当作已修正版本。
- 用户提供的五个顶层 G1 文件已确认与 Holosoma 提交 `bccd4d7451640a2800ddc77e469d911a84f91994` 逐字节一致。
- OmniRetarget 固定 revision 为 `135259130158f1cb1799e4ab9b6566c938285fc9`;归档 SHA-256、ZIP CRC 和选定发布资产哈希已校验。详细记录以 manifest 为准,它不是所有轨迹的逐文件哈希清单。
- 主项目忽略两个数据入口和 Holosoma checkout;迁移项目需按清单恢复下载和软链接,不能假定代码仓库自动携带这些资源。
- 当前公开资源已足够启动审计与实现,处理版 OMOMO 可直接提取人体关键点,暂不需要用户额外下载 SMPL/SMPL-H 参数模型。

## 数据约定与防止泄漏

- 教师 NPZ 全部存在 `qpos`、`human_joints`、`fps`。头部检查确认 `qpos=[T,43]`,fps=30,共 323,812 帧(含增强约 3 小时)。
- 按文件名关联到本机 312 个 OMOMO 源文件,仅证明文件名存在,不能视为坐标、人体尺度、增强关系和逐帧时间已对齐。
- 上游 `load_intermimic_data` 从 PT 列 `[162:318]` 提取 `[T,52,3]` 人体关键点,从 `[318:325]` 读取并重排物体状态;本地数值验证尚未完成,适配前核查真实张量。
- 内部位姿统一为 `[x,y,z,qw,qx,qy,qz]`,单位为米、弧度、秒。数据卡中的根/物体位姿为 `[qw,qx,qy,qz,x,y,z]`,必须显式转换并验证。
- 分别保存源人体场景与目标机器人场景的物体轨迹;增强后两者可能不同。禁止仅扰动物体而直接复用不再有效的教师标签。
- 先按原始演示分组划分,再增强和切窗。同一演示的增强、教师重生成和回放数据不能跨训练/验证/测试集。
- 推理条件不得包含目标 G1 真值根轨迹、真值脚落点或依赖完整教师优化得到的逐样本质量门槛。

## 模型已知问题

- 普通 `g1_29dof.urdf` 的 `left_pinky_link` 包含 `<collsion>` 拼写错误,标准解析器不会将其识别为 collision;MJCF 存在该球。需在派生模型中修正并验证。
- spherehand URDF 含 4 个圆柱碰撞体,对应 MJCF 的相关 geom 被注释;两种格式不能假定碰撞模型完全一致。
- 每只脚有 5 个半径 5mm 的碰撞球,踝部球半径 10mm;部分手部 visual 球半径 5mm,实际 collision 球半径仅 1mm。
- spherehand 包含半球手部 mesh 和 tip 球,与普通手的接触几何不同。正式版本锁定前不混用两种模型生成教师与训练标签。
- 小球是局部代理,不保证覆盖全部机器人表面;接触锚点需考虑半径与 site 外参,不能将显示球或球心直接当作接触表面。
- 带箱体场景 mesh 已补齐,但箱体缩放、原点及与发布版资产的关系仍需验证。XML 可解析和文件存在不等于 MuJoCo 已成功编译。

## 实现与验收要点

- 根位姿与 29 关节联合生成;优化变量为根平移 3D、旋转切空间 3D 和关节增量 29D,总计 35 维。不可把四元数存储维数直接当作速度维数。
- 手锚点在目标物体系固定,支撑脚锚点在世界/地面系固定;触地状态验证通过后才能锁定脚锚点。接触切换保持碰撞、限位和速度约束。
- 扩散使用 x0-prediction;Diffusers 对应 `prediction_type="sample"`。限位映射仅作用于干净关节角,关闭会错误裁剪根/旋转的全状态默认裁剪。
- 60 帧动作窗口、31 帧时间注意力窗口为起点。DDIM 25 步包含多次前向,计时不能视为一次网络调用。
- 修复每次迭代更新 FK、距离和雅可比,采用顺序检查并引用已接受前帧。线性化约束可行不等于原非线性约束可行。
- 最终独立检查碰撞、接触、限位、速度、窗口边界及累计漂移;失败轨迹显式拒绝。2mm 是选定碰撞模型与检查采样下的穿透容差,不宣称形式化零穿透。
- 计时包含预处理、生成、拼接、检查、修复、重采样及终检;同时报告整段接受率,不能通过拒绝困难样本制造加速。
- 教师轨迹和回放数据同样需要质量检查;不能把公开教师输出直接认定为可行真值。

## 当前进度与后续顺序

**最新补充(2026-09-26,优先于下文历史完成状态):**[标签分级与 train24 受限验证方案](docs/plans/m2_m3_label_quality_train24_plan.md)已实施完毕(提交 `ee86467`/`21da483`/`ea94719`,`uv run --locked pytest tests/ -q` **186 passed**)。**阶段 A**——订正前序记录:统一版本相对 v3 的质量退化为 **3 项**(sub10 左手认证∩参考交集 **0 帧**[认证总数 3 落在参考窗口外,v3 为 3]、sub10 右覆盖 −5.4pp、sub8 右 −1.25pp;旧记录"认证交集全达标/仅两项偏差"作废,review-fix 验证记录顶部订正指针);"不可能由稳定权重恢复"撤回;sub7 两轮正式重放取证(帧 101、关节 26、12.000000401362618 rad/s、超限 4.0136e-7,两轮同因结构化一致)。速度检查新增未舍入结构化证据(值/上限/超限量/差分帧/关节号,裁决不变)、质量报告四类认证分列+严格 `<`+同区间越阈+t0 无差分速度;标签协议 v1 冻结(`configs/labels/graded_labels_v1.yaml`,旧质量门继续标未通过)。**阶段 B**——分级标签落地:`export_label_manifest.py`(每 demo 一行清单+`labels/*.npz` 掩码/区间 ID,strict 复检单列)、`TeacherWindowDataset(label_manifest=…)`(哈希/T/fps/协议/分组核验、不回退)、`LossAssembler` 分级接触监督(空掩码零损失零梯度、掩码不进 conditioner)。**阶段 C**——六段回归 rq/rv/ns 与冻结 Nq/Nv/Ns 逐段一致(NPZ 字节级相同)、受限转段门通过后执行 train24(`calibration24_train_v1`,三次运行产物字节一致,采纳洁净溯源运行):**13/24 keep < 18/24 预注册数量门未达标**——失败分类:终检速度微超限 3 段(3e-9~4e-7 rad/s,与 sub7 同类)、求解不可行 6 段(regen 手带行贴穿透下界为主)、召回不足 2 段;六个受试者均有可用窗口(40 窗/60.3s)、局部认证监督左/右 50/90 帧(6 段双侧零认证)。**按预注册规则停止规模扩展:不进入 312 originals、不重训**;失败类别最小修复另立方案。详见[验证记录](docs/verification/m2_m3_label_quality_train24_verification.md)。

**2026-09-24 补充(已被上段部分订正):**对 `736f676^..cb8e4c0` 的实现审阅确认 9 项缺陷(1 P1、8 P2),已按[实现审阅修复方案](docs/plans/m2_m3_stage_gate_review_fix_plan.md)全部修复并分项提交:**R1** 探测行判据改为返回距离(截断的 1.2mm 上界与零 `fromto` 不再入 QP)、**R2** 接触记录法向按 geom1→geom2 取反、**R3** 探测缓存改全依赖键(回流新增配对/箱姿变化即时重建)、**R4** `mpjpe_growth` 改逐帧 FK(`scripts/segment_quality_report.py`)、**R5** 拒绝证据绑定最后执行尝试且 `rejected_final` 带 validator 结构化失败帧、**R6** 运行级成本汇总全部轮次(schema 2 + `cost_legacy_final_pass`)、**R7** 违反行描述符覆盖 96 条锚点/速度行、**R8** 尝试事件在检查/回溯后定稿、**R9** 记录实例有效配置与运行中漂移。提交 `c8d266f`/`b9fc080`/`a22e775`/`c991f6b`/`12c5741`/`9658b02`;165 项测试(R1/R2/R3 反例各自可识别原缺陷)。**D 对照**确认诊断类修复零行为差异。**统一版本六段**:Nq/Nv regen **5/6 keep**(sub12 恢复接受、sub7 转为终检速度 @t=101 拒绝、两轮同因;Nq 与 Nv 除耗时外逐项一致)、Ns 标准模式 **6/6**(与旧标准逐段同值,3 段 QP 迭代数因探测行纠正变化)。历史 `mpjpe_growth`/q1/q5 已逐帧重算(旧"误差随段显著下降"作废),完整六段对照与缺口标注在 `derived/data/teachers_regenerated/quality_tables/`。"每侧几何接近覆盖不低于 v3"2 项覆盖偏差与 sub10 左交集退化按标签协议 v1 以用途限制承接(不追认旧门通过);strict/keep/任务工作门 0.40 不变。详见[修复验证记录](docs/verification/m2_m3_stage_gate_review_fix_verification.md)(顶部有 2026-09-26 订正)。

截至 2026-09-22(第十轮;质量门状态以 2026-09-24/26 订正为准),M0、M1 完成,M2 机制完成、验收部分未达标(扰动接受率 60%<95%、单帧耗时远高于 50ms;accepted 结果 100% 过独立终检、不可达 0 误受),M3 机制完成、阶段门数量/共享回归达标(覆盖 2 项偏差,认证交集后经 2026-09-26 订正为 sub10 左 0 帧未达标)(详见下)(M0 最后一项"分组清单/阈值/场景边界"于 M1 工具就绪后补齐):

- M0:依赖锁定(torch 2.14.0+cu126 GPU 验证、mujoco 3.13.0)、派生锁定资产(`derived/robots/g1_29dof_locked*`,31 sites)、71 样本配对审计。**M0 收尾(本轮)**:250/31/31 按演示分组划分(种子 20260911,受试者哈希 zlib.crc32,清单锁定 `derived/data/split/`;总量 3.0h/312 演示/1,826 NPZ;17 个无教师 PT 为 reserve)、场景边界(root [0.305,0.814]m、箱 [0.179,1.561]m,`configs/task/box_carry.yaml`)、参考侧接触阈值标定(hand 0.04/0.072、foot 0.05/0.08,宏 F1 0.876,`configs/eval/protocol.yaml`)。详见 `docs/verification/m0_verification.md`。
- M1:运动学层(`kinematics/` 35 维 retract/雅可比,1000 状态 FD 验证 max 3.9e-9;torch FK 对拍 max 7.1e-7 m)、交互网格(`imesh/` 逐帧 Delaunay+均匀 Laplacian,刚体不变性测试)、几何层(`geometry/` 碰撞对+允许列表、可微 AABB 距离;**MuJoCo mesh 重定心旋转已实证**,box geom 系≠OBJ 系)、接触推断(`embodied/contact_sites.py` 双阈值+去抖、hull 参考度量)与 F1 指标(双空未定义策略)、独立验收器(`repair/validation.py`,含自碰撞、表面距离间隙、逐 site 检查、显式有限性门与 status 状态码、含箱体运动的真自适应细分)。教师质量量化(修复后 71 段:穿透最差 23.2mm、脚滑 ~0.3m/s、接触 F1 0.657 defined-only——教师非可行真值证据)、未约束 GN 纯优化基线(整段 MPJPE 4.9cm、10.6ms/帧、求解首帧 2.6cm)。容差与阈值锁定于 `configs/eval/protocol.yaml`(推断 enter ≤ 验收门槛由代码强制)。详见 `docs/verification/m1_verification.md` 与 `docs/plans/m1_fix_plan.md`。
- M1 审阅修复(2026-09-11 第二轮):F1 双空虚高、脚部球心当表面、AABB 代理进标定、基线首帧 yaw、有限性检查、死配置六项已修并重标定重锁(旧值与失效原因保留在 protocol.yaml 与 m1_verification.md §8)。
- M2(2026-09-12):顺序 QP 修复器四模块(`repair/{context,constraints,scp,pipeline}.py`,OSQP 3 依赖新增);速度限值标定(关节 12/根线 1.8/根角 7,教师分布 max+裕量);**8 项数值 bug 修复**(穿透梯度符号、mesh 旋转项、信赖域行错位、违反度归一、箱-地伪影等,FD 对拍验证);59 项测试全过;M2 验收实验如实记录 60%/95%。详见 `docs/verification/m2_verification.md` 与 `configs/repair/qp_default.yaml`。
- M3(2026-09-13):数据层(`data/{schema,teacher,windows}.py`,完整 §4.3 schema + 修复层适配;clip 坐标系取自参考人初始位置/yaw;划分卫生有测试)、token 层(`model/tokens.py` 38 维 x0:clip 系根位置训练分组标准化 + rot6d + 关节限位映射,退化显式标记)、模型层(`model/{conditioner,backbone,deterministic,generator}.py` + `diffusion/generator.py`:30 token/帧、d240/4层/6头、时间注意力窗口 31、条件交叉注意力;Diffusers x0-prediction `prediction_type="sample"`、`clip_sample=False`)、五项损失(重建/测地/FK/接触/速度)、60/30 窗口拼接(SLERP 融合,接缝速度低于内部 p95)、训练器与脚本(`scripts/{precompute_m3_cache,train_generation,eval_generation}.py`)。**教师 1,826 文件关节零越界**(无剔除/裁剪);根统计 258,462 帧。pilot(修复后):两模型各 4000 步 62.7/60.6 分钟、峰值 5.47GiB、val 0.0949/0.2491。验收(8 段 val,修复后):MPJPE 9.0cm[扩散]/10.9cm[确定性]、根 RMSE 5.6/5.2cm、DDIM-25 ≈8.8ms/帧(5/10/25 步消融质量不敏感);**所有输入含教师噪声基线整段修复接受率 0%**(同因 `rejected_frame`)——§10.1 的 ≥90% 整段接受率被 M2 收尾+教师重生成阻塞,如实记录。详见 `docs/verification/m3_verification.md`。
- M3 审阅修复(2026-09-13 第三轮):手接触损失 clip→box 变换缺转置(einsum `"btij"`→`"btji"`,提为 `LossAssembler.hand_points_box_geom`;教师 val 对拍修复前最大偏差 0.104m → 修复后 5.9e-8m,AABB 表面点 SDF 残差 ~1e-16);连带发现并修复 `box_geom_to_world` geom→body 方向反转(原与 `world_to_box_geom` 不互逆、往返差 0.26m,**无调用点未污染已有实验**)、backbone 过时 Triton 注释。两模型同配置重训、验收数字刷新(见 m3_verification.md §8;首轮产物归档 `derived/models/*_v1_contactbug`)。教训:教师接触帧 |SDF| 在正误两链下均值几乎相同(~0.087/0.088m),损失曲线无法暴露旋转方向错误——几何变换方向必须靠对拍测试锁定。
- P0 分层干预(2026-09-13 第四轮,方案 `docs/plans/p0_relaxed_validation_plan.md` 修订版,验证 `docs/verification/p0_verification.md`):解除教师整段修复 0% 阻塞。**语义修复**(全部不动 §10.1 容差数值):接触带逐对几何分类("触地但未锚定",`scp.py::_collision_rows`)、准静态锚点建立门 + 预测滑动/漂移即释放(`ContactStateMachine.release`)、转移插值检查对齐终检自适应细分、验证器推断速度门 ≤ 检查门(`foot_entry_speed_mps` 0.3→0.02、新增 `hand_entry_speed_mps`;支撑 := 准静态,快滑动为过渡不认证)。**成对放宽模式**(`repair/config_io.py`:同时驱动两配置,"修复带 ≤ 验收门"六对不变式加载时强制;普查定值:脚带 17mm/手带 35mm/手锚 19cm——后者 yaml 注明教师手不锚定、无约束力)。**结果**:strict 0%→2/6(语义修复单独),relaxed 4/6=66.7% 与 17/24=70.8% ≥50% 验收线;`task_mpjpe_tol_m` 0.40→0.101 标定(17 段/1,999 帧,带门槛零下降);3 存量 bug 修复:跨段网格地面点缓存泄漏、根速度行逐分量 vs 终检范数(对角步 1.66× 漏过,sub7 实证)、终检手足检查无定位致后缀重试退化为末 5 帧。教师速度跳变筛查:20/312 original 预注册排除。P0.3 闭环(8 val 段,0.40 工作值):裸教师 25%、噪声基线 12.5%、扩散 12.5%、确定性 25%(机制打通;标定门槛 0.101 对噪声/生成输入过紧的选择偏置已实测);val 段接受率对演示难度强依赖(训练侧 24 段 70.8%)。**relaxed 产物不计入 §10.1 达标**。
- P0 审阅修复(2026-09-14 第六轮,方案 `docs/plans/p0_fix_plan.md`,验证 `docs/verification/p0fix_verification.md`):外部审阅发现四项缺陷,修复全部落地、未改任何容差数值。**F1** 手部推断速度世界系→箱体系(`validation.py::_infer_robot_schedule`;原实现使搬运相抓握区间构造性清零,P0 全部运行手部检查形同虚设;修后 sub7 strict 形成区间并通过,教师悬浮 gap p95 33mm>enter 10mm 使其余区间仍为零——悬浮为数据本性)。**F2** 转移检查用完整前后场景(FrameContext.box_pose_prev、插值箱姿、中间态仅碰撞族行、逐探测点重扫碰撞对)+ 运动量改**箱体角点扫掠**(关闭终检纯旋转细分盲区,两侧一致)。**F3** CheckResult.first_failure_frame 结构化定位(速度族归差分后帧、插值碰撞归区间右端、手锚逐帧对 iv.anchor)+ 全失败检查取最小帧 + 无定位→保守全段重试(原字典序取项+无定位退化末 5 帧重试是 P0 期 strict 低接受率主因)。**F4** 闭环口径(`scripts/closed_loop_metrics.py` 纯函数 + `eval_generation.py` 重写):配对任务 MPJPE(修复后轨迹、共同接受集、机器人关键点 vs 参考人体关键点,箱体系主口径)、成功加权手 F1(双空未定义不填 1、零证据为 None)、slack 帧数独立记录(路径名不再丢失)、元数据精确匹配复用、产物名带模式/门槛;原 8.2/11.0-11.1cm 归类裸输出教师拟合误差。**重测**:21/21 旧接受轨迹新终检仍过(零撤回);6 段 strict 0%→**6/6**、relaxed **6/6**;24 段 relaxed **22/24=91.7%**(0.101 固定门 16/24);任务门重标 **0.113**(22 段/2,599 帧,0.101 留 superseded 对照);8 val 闭环(0.40):裸教师 37.5%、σ0.02 噪声**三条件全满足**(协议可达)、扩散 12.5%(配对差 +4.2mm、接受率不足)、确定性 25%(+7.8cm);0.101 下非教师方法全 0/8。约束定位:生成质量(4k 步未饱和+悬浮教师),教师重生成论据增强。relaxed 产物仍不计入 §10.1。
- 教师重生成·路径 B 启动(2026-09-15 第七轮,验证 `docs/verification/teacher_regen_verification.md`):`regen_mode`(参考驱动主动手部接触·最近几何体行 + 闭合走廊 0.15 m/s + 手部关键点专项目标·表面外投 8mm、weight 25)+ `scripts/regenerate_teachers.py`(teacher 布局产出、strict 筛选、接触覆盖 keep 判据、溯源报告)。先导 2/6 保留(接触覆盖 90-100%、task p95 0.072-0.076、认证区间首次成形;四段拒于拉入转移/教师穿透起始)。**存量 bug 修复**:slack 缺 s≥0(M2 期,s 可至 −51、近不可行系统出破坏性步);新增步长平方 dip 余量(min(6mm, max(0.5mm, 0.3·motion²)));**手部门参考重标**(锚漂 0.01→0.02、相对速度 0.02→0.8、进入 0.02→0.05——旧值排除参考人体本身)。**标准模式 6 段 strict 6/6→4/6,全部基线须重测**;负结果:手部速度硬行与换 grip 冲突已禁用。
- 教师重生成鲁棒性修复(2026-09-15 第八轮,方案 `docs/plans/teacher_regen_robustness_plan.md`,验证 `docs/verification/teacher_regen_robustness_verification.md`):**F1 走廊上界同步收紧已实现**(regen 上界减 `solver_band_margin`,与 solver band 行同一公式;对齐单测锁定)、**`--verbose` 诊断已实现**(`RepairPipeline.frame_log`:端点/探测逐次记录手距、走廊 dist/upper、margin、motion、violation rows 含几何体对名;`regenerate_teachers.py --verbose` 逐帧摘要+逐段 frame_log JSON)。**P1 检查点证伪 F3 前提**:sub3 t=0 合法(手距箱 31.9/68.3mm)且修复成功,t=1 实败于**脚部**端点+插值——按方案自身决策树取消 F3;F2/F4 触发条件/有效性不成立,未实现。**四段失败机理被插桩证据修订**(均非方案 §2 的走廊-余量不同步):sub12=教师滑步行走+零运动累积偏离后大步弦线穿箱、sub16=抓取区间激活前自碰撞弦线(膝~手 −6..−11mm)、sub7=教师手部 1-2 帧贯穿箱体俯冲不可复现。**先导重测 2/6(同帧同因),≥4/6 目标未达成;sub10/sub8 零退化**;strict 容差零改动;后续需新设计(探测引导 pair 激活、发散感知跟踪、regen 输入站点速度筛查、t=0 走廊下界),见验证记录 §5。
- 测试 99→102→103→109→134→165→**186 项全过**(`uv run --locked pytest tests/ -q`;2026-09-26 标签分级轮新增 21 项:`tests/test_label_evidence.py` 12 项[认证总数 vs 交集反例、严格 `<`、区间边界 ID、同区间越阈、报告值掩码复算、速度微超限证据、裁决语义不变、CLI 门卫] + `tests/test_label_quality.py` 9 项[数据契约拒绝、切窗掩码/ID、零监督零梯度、有效帧归一化对拍、conditioner 无泄漏、清单→损失冒烟]。此前审阅修复轮新增 31 项:R1 距离命中/零距离伴随/装配行 FD 3 项、R2 法向符号与倾斜接触 2 项、R3 缓存依赖 3 项、R7 违反行描述符 5 项、R5/R8 尝试与拒绝证据 7 项、R6 成本视图 4 项(`tests/test_cost_contracts.py`)、R4/R9 报告与溯源 7 项(`tests/test_report_contracts.py`,含 CLI 冒烟与 R5 拒绝块)。此前第十轮新增 25 项契约测试:`tests/test_repair_contracts.py` 覆盖 F0 尝试归因/日志纯度/后缀重试成本/失败快照重放、F1 pair 身份与四类成功返回复核/行预算截断、F2 `q`/`P` 梯度对拍与 50x/10x 反例、§7.1 探测插值/探测雅可比/穿箱转移/低于子步长阈值不激活、§7.2 稳定目标参考驱动与梯度对拍、箱体系雅可比 FD;第九轮 6 项:pair 回流进下一线性化、行预算不丢已发现 pair 且上报截断、成功返回前全量重扫、穿箱端点拒绝+多起点收敛、探测行 regen 上界放开、30mm→24mm 合法接近端点+中点;第八轮新增走廊-band margin 对齐 1 项,走廊测试期望值按 F1 更新;路径 B 新增 3 项:slack s≥0 装配、主动行最近几何选择+走廊、预测悬浮不释放锚;P0-fix 新增 15 项;P0 原有 6 项;2 项按"支撑 := 准静态"新语义更新)。关键工程事实:教师 NPZ qpos 为四元数在前布局需显式转换;扫描箱凸包与 AABB 差 mean~7cm(AABB 仅训练代理);MuJoCo freejoint 角速度 qvel 为体系;Python 内建 `hash()` 跨进程随机,划分种子必须用稳定哈希;参考脚部高度 82% 帧低于 0.02m,脚部阈值不可由 F1 标定;pytorch_kinematics FK 不支持 3 维批量(损失内拉平 B·F);系统 Python 3.12 无 dev 头文件导致 Triton 驱动编译失败,`.venv` 已重建于 uv 托管 CPython 3.12.13(注意力层保留显式数学实现以维持 pilot 数值可复现,SDPA 切换列为后续优化);先导四段失败的逐帧机理诊断依赖插值探测/端点记录,已固化为 `--verbose`,不再依赖临时脚本;全量 pair 扫描 ~1.4ms(1107 对)、快族 ~0.1ms——动态激活策略按族分成本设计。本轮新增事实:`mj_geomDistance` 未命中时精确返回 `distmax` 且 `fromto` 保持零值,行判据必须取返回距离(零距离真实接触同样塌缩线段,须保留);接触法向为 geom1→geom2;`largebox` 为 mesh geom 且 geom 系与体坐标间有固定重定心旋转,倾斜面方向期望不能由体四元数推得;网格-平面距离在“最深顶点切换”处有折点,FD 步长须 ≈3e-7(1e-6 已被折点污染);探测缓存键必须覆盖前帧/箱姿/监视与候选身份/协议/预算;标准模式同样启用探测行(与 `regen_mode` 无关)。标签分级轮新增事实:R9 漂移检查的 status 哈希覆盖**全部**未跟踪文件——运行期间新建的无关 docs/脚本同样触发 `code_changed_during_run`,长实验期间不得写仓库任何文件;旧质量报告的几何接近计数用 `<=` 与 keep 判据的严格 `<` 不一致(阈值上 2mm 诊断带不能放宽准入);报告速度序列 t0 的占位零会污染含 t0 的参考区间统计(须记 NaN);`certified_frames` 总数与认证∩参考交集在区间错位时可差至全部(报告四类必须分列)。
- M2-M3 收尾阶段 0-3(2026-09-16 第九轮,方案 `docs/plans/m2_m3_completion_plan.md`,验证 `docs/verification/m2_m3_completion_verification.md`):**阶段 0** 三项统计缺陷修复(`FrameResult` 真实迭代数+新增 `qp_calls`、restoration 后早退 `slack_used` 如实、`ms_per_processed_frame` 按实际处理帧/p95+旧口径保留注明)+ 尝试级 frame_log(skip_check/attempt:来源 prediction/prev_accepted/slack_fallback、回溯、状态、统计)+ 历史归因更正(sub7"教师贯穿箱体"实为求解候选穿透、原始 t=23/24 为 +55.7/+52.6mm;sub3 脚部浅穿透 −0.57/−0.62mm 非"手部 −1.4mm 深穿透";v2→v3 计时增量不可全归 verbose)。**阶段 1 动态碰撞对激活**(`scp.py::_scan_pairs`/`note_discovered_pairs`):候选集不再冻结于 z_init——端点/探测发现违规 pair 回流下一线性化、每接受步 partial refresh(快族 robot_box/ground ~0.1ms;全量 1107 对 ~1.4ms 仅帧首+成功返回前)、成功返回前全量重扫复核不返回未检查状态、截断规则已发现 pair 优先+溢出上报 `last_truncated`、帧边界清除携带。**阶段 2 插值走廊语义**(`probe_mode`):走廊上界为帧末目标,探测行只保留防穿透下界——30mm→24mm 合法接近中点不再误拒(复现用例),端点/求解路径不变。**阶段 3 重测(v4=阶段1+2 组合,`pilot_diagnosis6_v4_stage12/`)**:2/6 保留,**阶段门 <4/6 未达标,按 §7 门禁停止扩展不进入 24/312 段**;四段失败全部收敛为端点 `state_constraints`(插值走廊误拒类别消除);sub3 t=1→11、sub7 t=24→34 前移(脚部问题被阶段 1/2 消解),sub12 t=2 诚实快拒(教师 120mm/帧滑步不可行),sub16 t=22(自碰撞弦线回流后端点 −5.3mm 仍超 −1.8mm 门);标准模式 4/6 与基线一致零回归;**sub10 认证区间归零(3+2 帧→0)+接触 R 0.905→0.838 的可用性退化如实记录**(无中间态走廊后拉入更快、箱体系抖动增大,准静态进入门难触发;keep 判据用物理接触覆盖故 keep 维持),24 段标定前须恢复或给结论。阶段 4 条件项(跟踪策略对照、探测点约束+雅可比对拍、t=0 走廊下界消融)留待下一轮。

- 阶段门修复实施(2026-09-22 第十轮,方案 `docs/plans/m2_m3_stage_gate_fix_plan.md`,验证 `docs/verification/m2_m3_stage_gate_fix_verification.md`):**F0-F2 与 F4 分项提交**(`736f676` F0+F1、`4ce1baa` F2、`c702096` §7.1 探测点约束、`50afc75` 箱体系雅可比 einsum 修正、`2253998` §7.2 稳定软目标、`00e2ab4` `rejected_final` 归因、`3831eb2` §8 train24 冻结、`23bc211` 质量对照工具)。**F2 数学契约**:`q=2wJᵀr`、`P=2wJᵀJ`(旧实现把归一化残差再乘 `D⁻²`,平移 50×/关节 10×),旋转块 `J_l⁻¹(e_rot)/D`,FD 1e-5/1e-6/1e-7 锁定;修正后步长变小改变了状态链(B3 regen 1/6,按方案不回退)。**§7.1 探测点约束**:插值中间态碰撞进线性化与 merit(缓存 pair/alpha、前帧固定、箱姿插值、复合雅可比 `T=alpha·J_l(alpha·D)·J_l⁻¹(D)` FD 对拍 ~1e-10、独立探测行预算 12 且截断上报、成功退出前按完整场景重探);B3 1/6→5/6。**§7.2 稳定软目标**:参考驱动箱系手部稳定残差(`ref_delta` 只由输入计算),权重 0/0.1/1.0 预注册对照**结果为负**——0.1 使 sub10 左认证 0 帧(违反 ≥3)、1.0 以认证深度换数量(sub8 右 40→0),最终保留权重 0。**六段门(最终统一版本)**:**5/6 keep**(sub12 拒绝:速度终检 @t=3,两轮同因、可复现),认证交集帧 sub10 左/右 3/13、sub8 左/右 27/40(均 ≥ v3 的 3/3/4/0),标准模式 **6/6 accepted**(旧 4/6,无撤回),134 测试全过;安静/verbose 与提交前后共三次运行的状态、成本计数与保留 NPZ sha256 逐段一致。**未达标项**:每侧几何接近覆盖 2/12 偏差(sub10 右 −5.4pp、sub8 右 −1.25pp),已定位到 F1 状态链的瞬时脱离深度(B1 同值;v3 本身即有 72mm 最大脱离,本轮 95mm),据此按方案 §9 **不自动转段**,需先完成标签用途限制与质量协议修订的审阅;**未运行** train24/312 重生成与重训。**§8**:`calibration24_train_v1` 已冻结(旧 `p02_calibration` 19 train/2 val/3 test 原样保留;新清单 = 19 保留 + 按稳定 `sha256("20260917:<demo>")` 补 5:sub10_075/sub16_012/sub7_029/sub12_077/sub8_048;清单哈希 `52a98507…`;三条 test demo 标记已暴露,31 test 分母不变)。关键工程事实:箱体系手部 site 与网格关键点雅可比 einsum 曾把坐标轴求和(秩 1、方向错,FD 误差 ~1.4),修正后 3e-10,该缺陷由 §7.2 梯度对拍暴露。

**2026-09-17 审阅订正(优先于上述历史归因;已于 2026-09-22 实施,见上一条与验证记录):**阶段 0/1 尚有缺口:pair 回流混淆 `regen` 行角色与 `robot_box` 类别、QP 跟踪线性项重复尺度归一化、`best_feasible` 返回漏完整复核、verbose 检查带副作用、汇总漏失败尝试。sub12/sub16 存在端点通过但插值碰撞,不能按预测 `skip_reason` 归为全部端点失败;sub12 原始 t=1->2 场景运动 48.55mm,约 120mm 来自候选转移,不支持原始输入不可恢复归因。sub10 v3 认证左右各 3 帧,退化原因尚未分项证实。临时修正 pair 类别为 3/6,正式基线仍 2/6。新方案 [m2_m3_stage_gate_fix_plan.md](docs/plans/m2_m3_stage_gate_fix_plan.md) **仅编写完成,修复与重测待执行**。清单核查:diagnosis6 全 train;旧 calibration24 为 19 train/2 val/3 test,须保留历史并另冻结 train24,记录测试样本已暴露的限制。

后续按方案推进:

1. M2 收尾(优先,阻塞 §10.1 整段接受率):扰动接受率 60%→95%(多起点已入 P0,余下锚定带两段收紧、持久 QP 工作区)、单帧耗时(F1-F4 语义下最难段 p95 仍超 1s,sub3/sub7 的 slack 尝试 76/80,slack 占比与难段暂停是最耗项)、约束版纯优化基线;教师侧新基线为六段 regen 5/6 与标准模式 6/6(第十轮),扰动集须在 F1-F4 最终语义下重测。
2. M3 剩余:实现审阅修复(R1-R9)已完成;标签分级方案(2026-09-26)已完成——证据订正、标签协议 v1、分级标签接口、六段回归、train24 受限验证。**train24 裁决:13/24 < 18/24 未达标,不进入 312 originals、不重训**;后续按失败类别另立最小修复方案:(a) 速度终检边界与帧级容差对齐(3 段,超限 3e-9~4e-7 rad/s)、(b) regen 手带主动行贴穿透下界的不可行(6 段中 5 段)、(c) 接触召回不足(sub10 系 2 段)。修复后重新冻结重跑 train24(≥18/24 + 动作覆盖 + 产物契约)才可转 312→干净标签重训。分级标签接口(清单+掩码+数据集+损失)已就绪可直接消费。不能以变体计数或几何接近代替正式验收/持续抓握认证。
3. M4:闭环比较与最终评测(P0-fix 后机制与口径齐备:配对任务 MPJPE/成功加权 F1/三条件判定;σ0.02 噪声已全条件满足,strict 闭环留待教师重生成后)。

资源变化同步更新 `resources/README.md` 和 manifest;设计变化同步更新实现方案。本文件保留关键约定与阶段摘要,不要把未执行的计划写成已验证结果。
