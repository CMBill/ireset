# 项目开发约定与上下文

本文件适用于整个项目。信息整理日期:2026-09-11。硬件占用、资源数量和完成状态是检查时的快照,后续以实际文件及验证记录为准。

## 项目目标与范围

- 项目 `ireset` 面向人类-箱体交互动作到 Unitree G1 的重定向,当前仅支持一套固定的 **G1 29-DoF** 资产配置。
- MVP 为平地、单一刚性箱体、低速双手搬运;物体几何和轨迹已知。生成结果包含浮动基座位姿与 29 维关节角。
- 主路线为条件扩散生成、顺序 QP 修复、整段独立验收;确定性预测器使用相同修复器作为对照。SOCP 为可选消融。
- 当前不实现跨机器人泛化、灵巧手、regrasp、复杂地形、动力学保证或真机部署。几何接触不等于可承重抓握,运动学通过不等于 RL 可跟踪。
- 完整设计、阶段依赖和验收标准见 [实现方案](docs/interaction_retarget_impl_plan.md)。资源细节见 [资源登记](resources/README.md),版本与哈希见 [资源清单](resources/manifest.json)。

## Python 与环境管理

- 固定 **Python 3.12.x**,使用 **uv** 管理环境和依赖。`.python-version` 为 `3.12`,`pyproject.toml` 要求 `>=3.12,<3.13`。
- 使用项目 `.venv`;当前已验证解释器为 Python 3.12.3,uv 为 0.12.3。不要修改系统 Python 或混入其他项目环境。
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

截至 2026-09-12(第四轮),M0、M1 完成,M2 机制完成、验收部分未达标(扰动接受率 60%<95%、310ms/帧>50ms;accepted 结果 100% 过独立终检、不可达 0 误受;教师整段修复 0% → M3 须重备数据)(M0 最后一项"分组清单/阈值/场景边界"于 M1 工具就绪后补齐):

- M0:依赖锁定(torch 2.14.0+cu126 GPU 验证、mujoco 3.13.0)、派生锁定资产(`derived/robots/g1_29dof_locked*`,31 sites)、71 样本配对审计。**M0 收尾(本轮)**:250/31/31 按演示分组划分(种子 20260911,受试者哈希 zlib.crc32,清单锁定 `derived/data/split/`;总量 3.0h/312 演示/1,826 NPZ;17 个无教师 PT 为 reserve)、场景边界(root [0.305,0.814]m、箱 [0.179,1.561]m,`configs/task/box_carry.yaml`)、参考侧接触阈值标定(hand 0.04/0.072、foot 0.05/0.08,宏 F1 0.876,`configs/eval/protocol.yaml`)。详见 `docs/m0_verification.md`。
- M1:运动学层(`kinematics/` 35 维 retract/雅可比,1000 状态 FD 验证 max 3.9e-9;torch FK 对拍 max 7.1e-7 m)、交互网格(`imesh/` 逐帧 Delaunay+均匀 Laplacian,刚体不变性测试)、几何层(`geometry/` 碰撞对+允许列表、可微 AABB 距离;**MuJoCo mesh 重定心旋转已实证**,box geom 系≠OBJ 系)、接触推断(`embodied/contact_sites.py` 双阈值+去抖、hull 参考度量)与 F1 指标(双空未定义策略)、独立验收器(`repair/validation.py`,含自碰撞、表面距离间隙、逐 site 检查、显式有限性门与 status 状态码、含箱体运动的真自适应细分)。教师质量量化(修复后 71 段:穿透最差 23.2mm、脚滑 ~0.3m/s、接触 F1 0.657 defined-only——教师非可行真值证据)、未约束 GN 纯优化基线(整段 MPJPE 4.9cm、10.6ms/帧、求解首帧 2.6cm)。容差与阈值锁定于 `configs/eval/protocol.yaml`(推断 enter ≤ 验收门槛由代码强制)。详见 `docs/m1_verification.md` 与 `docs/m1_fix_plan.md`。
- M1 审阅修复(2026-09-11 第二轮):F1 双空虚高、脚部球心当表面、AABB 代理进标定、基线首帧 yaw、有限性检查、死配置六项已修并重标定重锁(旧值与失效原因保留在 protocol.yaml 与 m1_verification.md §8)。
- M2(2026-09-12):顺序 QP 修复器四模块(`repair/{context,constraints,scp,pipeline}.py`,OSQP 3 依赖新增);速度限值标定(关节 12/根线 1.8/根角 7,教师分布 max+裕量);**8 项数值 bug 修复**(穿透梯度符号、mesh 旋转项、信赖域行错位、违反度归一、箱-地伪影等,FD 对拍验证);59 项测试全过;M2 验收实验如实记录 60%/95%。详见 `docs/m2_verification.md` 与 `configs/repair/qp_default.yaml`。
- 测试 59 项全过(`uv run --locked pytest tests/ -q`)。关键工程事实:教师 NPZ qpos 为四元数在前布局需显式转换;扫描箱凸包与 AABB 差 mean~7cm(AABB 仅训练代理);MuJoCo freejoint 角速度 qvel 为体系;Python 内建 `hash()` 跨进程随机,划分种子必须用稳定哈希;参考脚部高度 82% 帧低于 0.02m,脚部阈值不可由 F1 标定。

后续按方案推进:

1. M2 收尾(优先):扰动接受率 60%→95%(多起点重启、锚定带两段收紧、持久 QP 工作区)、单帧 310ms→<50ms;约束版纯优化基线。
2. M3:确定性/扩散生成(60 帧窗口、拼接);教师数据重生成/清洗(M2 筛选结论);task gate 验证集标定。
3. M4:闭环比较与最终评测。

资源变化同步更新 `resources/README.md` 和 manifest;设计变化同步更新实现方案。本文件保留关键约定与阶段摘要,不要把未执行的计划写成已验证结果。
