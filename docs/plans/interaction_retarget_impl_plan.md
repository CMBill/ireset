# G1 交互保持重定向:扩散生成 + 顺序 QP 修复

## 技术实现计划(代码级,修订版)

> 项目代号:`ireset`(Interaction-preserving Retargeting)。
> 范围:仅使用一套固定的 Unitree G1 29-DoF 机器人资产;不包含跨机器人训练、零样本构型迁移或其他机器人实验。
> 修订日期:2026-09-11。论文事实与本项目设计分别列出;性能数字除明确引用外均为待验证目标。

> 当前推进入口（2026-10-11）：见[文档索引](../README.md)、[OmniRetarget 优先教师总方案](omniretarget_first_training_plan.md)与[W1 A3/B/C 草案](omniretarget_w1_a3bc_plan.md)。A2 已收口，UR 为完整 no_quality_gain；W1 尚未实施。本文继续描述项目最终目标与基础契约，旧阶段实施、审阅和停线记录按索引访问。

## 1. 项目目标与范围

### 1.1 目标

给定**人类-箱体交互参考、已知箱体几何与轨迹、平地信息**,生成 G1 的**浮动基座位姿和 29 维关节角轨迹**,经约束修复及独立验证后输出供下游跟踪使用的运动学参考。

核心研究问题:学习到的初值能否在保持交互质量和整段成功率的同时,降低纯约束优化的端到端成本。扩散模型是否优于确定性预测器,作为必须测量的实验问题。

### 1.2 MVP 边界

**纳入**:

- G1 29-DoF、固定碰撞模型、固定手部接触代理;锁定 URDF/MJCF、关节顺序、末端 site 和资产版本。
- 单一刚性箱体几何、平地、低速双手搬运,包含初次接触建立、搬运、放下与解除接触。
- 箱体轨迹作为输入,不参与生成或修复;无解输入允许被拒绝。
- 监督扩散生成完整状态,顺序 QP 修复,独立几何和轨迹验收;确定性预测器使用相同接口和修复器。
- 先使用可恢复配对的公开 G1 数据;缺少配对时用固定版本 Holosoma 自行生成 G1 教师数据。

**不纳入当前验收**:

- 多机器人、多构型条件、机器人 ID 嵌入、跨构型泛化。
- regrasp、推/按、灵巧手指控制、物体形状泛化、复杂地形。
- 力矩、摩擦锥、力封闭等动力学可行性保证;几何接触代理不等于能真实承重的抓握。
- 下游 RL 训练和真机部署。RL 接口可后置,不能作为自碰撞或其他违规的兜底理由。

### 1.3 完成标准与性能目标

1. **验收器可靠**:超容差、非有限数、求解失败或时间不连续的结果必须拒绝;不能只返回轨迹而省略失败状态。
2. **已接受轨迹全部通过 §10 的整段检查**:碰撞模型下最大穿透深度不超过 2mm,关节限位数值容差不超过 1e-5 rad,支撑脚最大速度不超过 2cm/s,并满足接触、关节速度及基座速度界。
3. **测试集质量**:整段接受率目标不低于 90%,且相对使用相同验收器的纯优化基线下降不超过 5 个百分点;手部接触 F1 不低于基线的 95%,关键点误差不超过基线 + 2cm。阈值只允许在 M0-M2 pilot 后、最终测试前修订并留记录。
4. **完整比较**:报告纯优化、确定性预测加修复、扩散加修复的质量与耗时。只有在上述质量门槛满足且端到端加速 >1 时,才认为加速假设成立;2× 为初始性能目标,5× 为挑战目标。
5. **诊断目标**:顺序检查跳过率 ≥30%、修复帧平均 SCP 迭代 ≤3、单帧修复均值 <50ms,均需实测,不能代替整段质量和端到端指标。

测试集以独立原始演示为单位,目标 100 段。若单箱体有效独立演示不足,使用全部预留测试演示并报告数量,不得用滑窗或增强变体凑成 100 个独立样本。

## 2. 论文依据与设计定位

| 来源                           | 核查事实                                                       | 本项目采用方式                                             |
| ------------------------------ | -------------------------------------------------------------- | ---------------------------------------------------------- |
| 多构型扩散论文 §III-A         | 动作包含浮动基座,普通关节使用角度、位置和速度冗余特征          | 生成基座和关节角,位置与速度由 FK/差分得到;这是本项目简化   |
| 多构型扩散论文 §III-B、§IV-B | 预测干净动作 X0;sigmoid 用于关节角限位                         | 使用 x0-prediction,只限制干净关节角输出                    |
| 多构型扩散论文 §IV            | 60 帧训练片段、31 帧时间注意力窗口;新增骨架后继续训练约 5 小时 | F=60、时间窗口=31 起步;不把 5 小时解释为目标机器人数据时长 |
| OmniRetarget §III-A           | 优化浮动基座和关节;均匀 Laplacian 权重;支撑脚位置约束          | 作为 G1 教师优化与交互损失依据                             |
| OmniRetarget 附录 VI-C、VI-D   | 物体系交互网格;SOCP 的锥约束来自 L2 信赖域,速度界为线性约束    | 默认改用归一化 L∞ 信赖域形成 QP;L2/SOCP 作为可选消融      |
| OmniRetarget §V-B             | 线性化仍可能产生少量穿透                                       | 增加候选状态复核、独立几何验收与拒绝机制                   |

本方案中的配对监督、接触锚点、检查后修复、回放训练和阈值均属于本项目设计,不视为原论文已验证结论。当前公开代码可能与论文描述不同,复现时分别记录论文版本、代码提交和配置。

## 3. 系统架构与技术栈

### 3.1 流水线

```text
离线:锁定 G1/箱体资产 -> 数据配对与分组划分 -> 教师轨迹检查
     -> 参考逐帧交互网格、接触调度与锚点 -> 训练生成器

在线:人类参考 + 箱体轨迹 + 地面
     -> 条件构建与初步可达性筛查
     -> 生成 (root, q):扩散 DDIM 多步 / 确定性预测一次
     -> 重叠窗口拼接
     -> 顺序 check-then-repair,引用已接受的前一帧
     -> 整段独立验收
     -> accepted 轨迹 / rejected 及失败原因

可选:训练分组内生成 -> 修复 -> 质量筛选 -> 有界回放训练
```

静态几何资产可离线预处理;新参考的条件提取、逐帧网格、锚点构建、拼接、检查及回退都计入在线耗时。扩散 DDIM 25 步意味着约 25 次去噪网络调用,不是一次前向。

### 3.2 模块与目录

```text
ireset/
  configs/
    robot/g1.yaml
    task/box_carry.yaml
    model/diffusion.yaml, deterministic.yaml
    train/supervised.yaml, replay.yaml
    repair/qp_default.yaml, socp_ablation.yaml
    eval/protocol.yaml
  src/ireset/
    embodied/    g1_spec.py, correspondence.py, contact_sites.py
    data/        omomo.py, omniretarget_converter.py, pairing.py, dataset.py, augment.py
    imesh/       delaunay.py, laplacian.py, transform.py
    kinematics/  torch_adapter.py, mujoco_adapter.py, rotation_adapter.py
    geometry/    distance_adapter.py, collision_pairs.py
    model/       tokens.py, attention.py, denoiser.py, deterministic.py, conditions.py
    diffusion/   scheduler_adapter.py, sampling.py, losses.py
    repair/      context.py, constraints.py, scp.py, pipeline.py, validation.py
    train/       supervised.py, replay.py
    eval/        metrics.py, baselines.py, timing.py, report.py
    viz/         render.py
  scripts/       audit_data.py, convert_data.py, generate_teacher.py, train.py, sample.py, eval.py
  tests/         test_conventions.py, test_fk.py, test_contacts.py, test_repair.py, ...
```

**技术栈**:固定 Python 3.12.x,统一使用 uv 管理项目环境和依赖;学习/数值库为 PyTorch 2.x、Hydra、SciPy、OSQP、MuJoCo、trimesh、pytest。训练记录可使用 wandb。可微 FK 优先适配成熟 PyTorch 运动学库(如 `pytorch_kinematics`),MuJoCo 提供修复用 FK/雅可比及独立几何查询;不同时自写两套 PoE。CVXPY + Clarabel 仅用于 SOCP 消融。

本节目录和 §14 接口是计划,不表示这些模块已经实现。

### 3.3 本机硬件基线

2026-09-11 使用 `lscpu`、`free -h`、`nvidia-smi`、`lsblk` 和 `df -h` 实测。可用资源为当时快照,训练与计时前需再次记录。

| 项目         | 本机配置/检查结果                                                                           |
| ------------ | ------------------------------------------------------------------------------------------- |
| CPU          | Intel Xeon Gold 5218R @ 2.10GHz,单路 20 物理核/40 逻辑线程,单 NUMA 节点                     |
| 内存         | 系统可见 125GiB(约 128GB 配置),当时可用约 119GiB;swap 8GiB                                  |
| GPU          | 单张 NVIDIA GeForce RTX 3090,24576MiB 显存(24GiB);当时占用约 100MiB,利用率 0%               |
| NVIDIA 驱动  | 580.178.04;`nvidia-smi` 显示 CUDA 13.0 支持上限                                           |
| 系统         | Ubuntu 24.04.5 LTS,x86_64,Linux 7.0.0-31-generic                                            |
| 项目存储     | SOLIDIGM SSDPFKNU020TZ NVMe,整盘约 1.9TiB;项目所在文件系统约 480GiB,剩余约 53GiB,使用率 89% |
| 其他存储     | ST4000NM0265-2DC107 硬盘约 3.6TiB,当前未显示挂载点,不计入项目可用空间                       |
| 解释器与工具 | `/usr/bin/python3.12`:Python 3.12.3;`/home/c112/.local/bin/uv`:uv 0.12.3                |

`nvidia-smi` 中 CUDA 版本不表示本机已安装相同版本的 CUDA Toolkit 或 PyTorch runtime。后续选择兼容 Python 3.12、RTX 3090 和当前驱动的 PyTorch CUDA wheel,安装后实测 `torch.cuda.is_available()`、设备名称、runtime 版本及 GPU 前向/反向;当前硬件检查不等于这些依赖已验证。

本机足以支持本方案的单 GPU 原型与 pilot,完整训练时间需实测。起步 micro-batch=4、梯度累积=4(有效 batch=16),生成网络评估 AMP/bfloat16 的稳定性与收益;修复数值计算和 FK 对拍使用 float64。DataLoader workers=4 起步,独立 clip 教师生成可从 4 个进程开始,每进程限制 BLAS/OpenMP 线程数,避免 40 逻辑线程过度订阅。24GiB 显存是否足够目标 batch 仍需测量,不能由参数量单独推断。

数据、uv 缓存、虚拟环境和检查点占用需共同核算。项目与 `/tmp` 位于同一文件系统,改用 `/tmp` 缓存不会增加容量;先下载小样本,检查点仅保留 best/last 和必要里程碑,不预计算保存重复滑窗。其他分区/磁盘未确认挂载和写入路径前不用于容量预算。

### 3.4 Python 3.12 与 uv 环境约定

- `.python-version` 固定为 `3.12`,`pyproject.toml` 设置 `requires-python = ">=3.12,<3.13"`,禁止项目意外切换至其他 Python 次版本。
- 虚拟环境放在项目 `.venv`,由 uv 创建与同步;当前复用已安装的 Python 3.12.3,不修改系统 Python。
- 依赖通过 `uv add`/`uv remove` 维护,提交 `pyproject.toml`、`.python-version` 和 `uv.lock`;日常运行统一 `uv run`,复现时使用 `uv sync --locked` 与 `uv run --locked`。
- 当前只建立无训练依赖的最小环境。M0 逐步加入并锁定 PyTorch CUDA wheel 来源和其余依赖,不一次安装未核查的全部技术栈。需要上游不同环境的教师工具单独用 uv 管理,不能将其依赖混入主环境或改变主项目 Python 3.12 约束。
- `.venv` 与缓存不纳入版本管理。实验记录额外保存 Python 精确补丁版本、uv 版本、锁文件、GPU 驱动与 PyTorch runtime,因为 `3.12` pin 允许后续补丁更新。

```bash
uv sync --locked
uv run --locked python --version
```

新增依赖时先 `uv add <package>` 并检查锁文件变化。待脚本实现后,训练与评估入口采用 `uv run --locked python scripts/train.py` 和 `uv run --locked python scripts/eval.py`。

### 3.5 成熟依赖优先的工程约定

标准算法和基础设施优先使用维护活跃、文档完整的成熟库。实现前先检查现有依赖与上游 API,项目保留薄适配层,明确坐标、单位、张量形状和精度约定,不复制库内部算法。此约定同时记录于仓库根目录 `AGENTS.md`。

| 能力                       | 优先复用                                         | 项目代码边界                                                        |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| 可微 FK、运动链解析        | `pytorch_kinematics`                           | G1 关节映射、浮动基座组合、site 外参及梯度验证                      |
| 修复 FK/雅可比、碰撞与渲染 | `mujoco`                                       | 35 维增量转换、碰撞对选择及独立验收流程                             |
| 离线旋转和插值             | `scipy.spatial.transform.Rotation` / `Slerp` | 统一 wxyz/xyzw 与坐标系约定                                         |
| 可微旋转变换               | 优先检查现有运动学依赖的 transforms API          | 验证 SO(3) 指数映射、6D 表示和梯度支持;缺失项再评估专用库或最小补充 |
| 网格加载和几何预处理       | `trimesh`                                      | 资产校验与 G1/箱体几何适配                                          |
| Delaunay 与稀疏算子        | `scipy.spatial.Delaunay`、`scipy.sparse`     | 逐帧交互点组织、语义索引和 Laplacian 权重组装                       |
| DDPM/DDIM 噪声与采样更新   | Hugging Face`diffusers` schedulers             | 状态编码、条件去噪网络、采样循环与损失                              |
| 注意力、混合精度、数据加载 | PyTorch attention/SDPA、AMP、DataLoader          | 骨架偏置、参考条件和有效帧 mask                                     |
| QP/SOCP 求解               | `osqp`;可选 `cvxpy` + `clarabel`           | 约束/目标组装、SCP 信赖域和接受规则,不自写求解器                    |
| 配置、测试和实验记录       | Hydra/OmegaConf、pytest、可选 wandb              | 项目配置 schema、领域测试及指标定义                                 |

可选库先验证 Python 3.12、许可证、维护状态、安装/磁盘成本以及 GPU/autograd 支持再引入,通过 uv 锁版本。优先利用现有依赖解决需求,不为少量转换功能引入未经评估的大型框架。

确需自写的部分限于人-G1 配对、交互网格语义、手足接触状态机、约束线性化与组装、SCP 流程、质量筛选和评测。若成熟 API 缺少可微箱体距离、特定旋转表示等功能,先记录具体缺口,再实现最小、可验证的辅助函数;不扩展为通用 SDF、碰撞或运动学引擎。

依赖 smoke test 应覆盖项目实际用法,包括根姿态变换、梯度、GPU dtype、调度器参数化和几何距离符号,不能仅以成功 import 作为适配完成的证据。

## 4. 数据层与配对

### 4.1 当前可用数据与阻塞条件

**自动获取补充(2026-09-11)**:已取得固定提交的 Holosoma 重定向代码、缺失 largebox 资产,以及固定版本的 OmniRetarget robot-object 轨迹和发布版 G1/箱体模型。实际 1,826 个 NPZ 均含 `qpos`、`human_joints`、`fps`,共 323,812 帧、30Hz;按文件名匹配本机 312 个 OMOMO 源文件。发布数据实际字段多于数据卡,可以优先核查内嵌人体轨迹来恢复监督配对,但坐标/缩放与时间对应仍待验证。下载位置与哈希见 `resources/README.md`、`resources/manifest.json`;保留发布模型与当前 Holosoma 模型的版本边界。

**本地接入状态(2026-09-11)**:用户已提供 `resources/robots/g1_holosoma` 模型目录;处理版 OMOMO 已通过软链接接入 `resources/data/OMOMO_new`,指向 `/home/c112/codes/bill/smrt-data/unzipped/OMOMO_new`。静态检查发现 4,421 个 `.pt` 文件,其中 largebox 文件 329 个;尚未确认独立演示数、张量语义或教师配对。模型碰撞球、格式差异和缺失箱体 mesh 详见 [资源登记](../../resources/README.md)。原始资源保留,数据链接仅作只读输入,派生数据写入独立目录。

截至修订日,OmniRetarget 公开数据卡描述 G1 共约 4 小时数据,其中 `robot-object` 约 3 小时,另外两类各约 0.5 小时;并非全部适用于单箱体 MVP。数据卡列出的 `.npz` 字段为 `fps`、`qpos`,没有承诺参考人关键点、接触标签和完整增强元数据。LAFAN1 重定向结果未随该数据集发布。

优先下载小样本检查实际键、维度、资产、时间同步与来源。公开数据仅在配对可信时进入条件监督训练;无法恢复配对的轨迹不伪造人类条件,可用于独立数据审计或后续动作先验实验。

Holosoma 使用 InterMimic 处理版本的 OMOMO,不是直接读取原始 OMOMO。自生成路径须锁定这条预处理链;当前代码可保存 `human_joints`,但仍需自行保存源演示 ID、原始/目标物体轨迹、增强参数、几何资产和转换记录。

### 4.2 配对、坐标及资产约定

每个训练样本必须可以追溯到同一段人类演示、相同时间轴及相应目标场景:

- 确认原始演示 ID、人物、箱体 ID、裁剪区间、源帧率、重采样方式、增强父 ID 与生成版本。
- 分别保存参考人交互的物体轨迹和目标机器人交互的物体轨迹;场景增强后两者可能不同。
- 锁定 G1 29 关节名称与排列、限位、速度界、碰撞资产、手部接触代理及地面高度。不能只凭 `29` 维推断关节顺序。
- 内部位姿统一为 `[x, y, z, qw, qx, qy, qz]`,长度用米、角度用弧度、时间用秒。公开数据卡中的根/物体排列为 `[qw, qx, qy, qz, x, y, z]`,必须显式转换。
- 四元数归一化并处理相邻帧符号连续性;旋转重采样用 SLERP,不用逐分量线性插值。
- 人到 G1 的解剖关键点映射、手掌表面接触点映射和脚底 site 映射分别定义。

### 4.3 统一 schema

```python
@dataclass
class ContactInterval:
    effector: str              # left_hand/right_hand/left_foot/right_foot
    start: int                 # inclusive
    end: int                   # exclusive
    support: str               # object or ground
    site_ids: tuple[str, ...]   # calibrated G1 local sites
    anchor_local: np.ndarray | None
    # [S, 3]; hand anchors in object frame; foot anchors resolved at touchdown
    normal_local: np.ndarray | None
    confidence: float

@dataclass
class ReferenceCondition:
    source_clip_id: str
    human_kps_world: np.ndarray  # [T, K_h, 3], required
    source_obj_pose: np.ndarray  # [T, 7], human reference scene
    target_obj_pose: np.ndarray  # [T, 7], desired G1 scene
    obj_asset_id: str
    ground_plane: np.ndarray    # [4], normalized plane equation
    contacts: list[ContactInterval]
    frame_dt: float
    world_from_clip: np.ndarray # [4, 4], from human reference, not target truth

@dataclass
class InteractionClip:
    condition: ReferenceCondition
    q: np.ndarray               # [T, 29], teacher G1 joint angles
    root: np.ndarray            # [T, 7], teacher G1 floating base pose
    provenance: dict            # asset hashes, source group, augmentation, versions
    validation: dict            # teacher quality and pairing audit
```

监督数据必须有完整 `ReferenceCondition`。推理不接收目标 G1 真值根轨迹;训练时也不能把目标 `root` 混入参考条件。

### 4.4 划分与增强

- **先按原始演示分组划分,再增强和切窗**。同一演示所有增强、教师重生成和回放样本留在同一分组。记录人物重叠,需要人物泛化时另设按人物划分实验。
- 主设置目标为 train/validation/test = 80/10/10,小样本时按组数调整并锁定清单;报告独立演示数、实际时长、窗口数,不把窗口数等同于独立数据量。
- 有效在线几何增强仅采用一致的水平平移和绕竖直轴旋转,同步变换人、G1 根、物体和世界系锚点;地面保持一致。规范化后等价的增强不能算成新的任务多样性。
- 物体相对机器人单独移动、改变初始高度或轨迹时,重新求解教师轨迹,保存源/目标场景关系。不能保留原标签直接当作有效增强。
- 训练窗口 F=60;短片段使用有效帧 mask。随机噪声用于扩散输入或修复扰动实验,不污染干净监督标签。
- 帧率先统一为 30Hz;8 小时在 30Hz 下为 864,000 帧,本项目实际规模以单箱体配对审计结果为准。

## 5. 条件表示与交互网格

### 5.1 参考条件

- 人体关键点转换到**源物体系**:`x_source_obj = R_source_obj^T (x_world - t_source_obj)`。
- 目标箱体位姿、参考人根/足的世界系运动、地面与重力方向保留为独立条件。用参考人初始位置和 yaw 定义固定 clip 坐标系,只去掉水平原点和 yaw,保留高度与重力关系。
- 单箱体使用确定的尺寸和固定表面点作为几何条件;MVP 不训练 PointNet 形状编码器。
- 接触调度、手部锚点及其可信度进入条件;锚点构建见 §8.1。未来脚落点尚未锁定时用参考足迹与有效性 mask,不能使用目标真值脚落点。
- 整体一致刚体变换在相应局部坐标中保持相对几何;物体单独移动不具有这种标签不变性。

### 5.2 固定 G1 骨架与对应

`G1Spec` 包含固定 29 关节及浮动基座节点的骨架拓扑、关节轴、连杆变换、限位和接触 sites。保留静态邻接嵌入作为网络先验,不构建多构型图编码器或机器人 ID 分支。

`eta_anatomical` 用于人-G1 关键点匹配;`contact_sites` 用于手部代理、脚底点和表面法向。两者不能互相代替。参考交叉注意力可用 eta 作为稀疏偏置;无对应节点仍能访问全局参考 token,避免全 mask 行和信息断路。

### 5.3 逐帧交互网格

- 从 eta 中选定共同语义点,形成 K 个一一对应的人/G1 点,再加固定物体表面采样点 P 和地面点 E。不能假设全部人关键点都有 G1 对应。
- **每帧参考侧构建 Delaunay 拓扑并预计算均匀权重 `L_t`**。同一帧的目标侧与全部 SCP 迭代复用该拓扑;不把整段拓扑固定一次。
- `delta_ref[t] = L_t @ X_ref[t]`;目标点由 G1 FK、目标物体和地面构成,统一变换到目标物体系后计算 `delta_robot[t]`。
- 源/目标表面点和地面点的语义索引保持一致,地面点不是跟随箱体移动的附着点。新参考逐帧建网格的成本计入推理时间。
- 默认参考能量为 `E_L = mean(||delta_robot - delta_ref||^2)`。它是软几何损失,不保证接触、非穿透或动力学可行性。
- 数值退化先检查重复点、尺度和共面性,再用固定种子的尺度相关微扰或 `QJ`;记录失败率及排除分布,不能静默丢弃困难测试样本。

验收包括逐帧维度/索引对齐、刚体一致变换测试和不同接触阶段的网格可视化。Laplacian 在世界系对平移不变、对旋转等变;局部坐标不变性测试必须同步变换全部相关几何及坐标系。

## 6. 运动学与几何层

### 6.1 浮动基座与雅可比

物理状态记为 `z_t = (T_WB,t, q_t)`,其中 `T_WB` 为根位姿,`q_t` 为 29 维关节角。存储维数为 7+29=36;优化增量使用:

```text
xi = [delta_p_world(3), delta_theta_world(3), delta_q(29)] in R^35
p_next = p + delta_p_world
R_next = Exp(delta_theta_world) @ R
q_next = q + delta_q
```

雅可比统一相对上述 35 维增量。MuJoCo 的 qvel 约定与该增量可能不同,适配器负责转换,不得把 7 维四元数存储导数直接视作 6 维自由基座速度。

- `pytorch_kinematics` 的 FK 用于训练几何损失;MuJoCo 原生 FK/雅可比用于修复与验证。项目只实现约定转换与接触 site 适配,不自写 PoE 或运动链解析器。
- 固定/被动关节、URDF 零位旋转和手部 site 外参显式处理。
- 验收:G1 1000 个随机有效状态,双精度 FK 位置与旋转误差目标各 <1e-6 m/rad;对 35 维雅可比做流形有限差分验证并记录绝对/相对误差。

### 6.2 碰撞模型与距离查询

- 锁定物体、地面、G1 连杆和接触代理的 collision 几何。手部代理用于几何搬运参考,不表示灵巧手闭合或力封闭。
- 修复可用胶囊/凸几何加速,但先检查其覆盖关系。非保守简化不能作为验收的唯一依据。
- 优先使用 MuJoCo 几何距离查询和原生雅可比构建修复约束;训练损失若需要可微箱体/平面距离,先检查已引入库的支持,确有缺口时仅补充解析点-箱体/平面函数并与独立查询对拍。距离经过 FK 后对关节角通常仍然非线性,需要线性化。
- 非接触碰撞对采用正安全间隙;接触对采用允许接触的距离带,不能同时要求该对保持正间隙和精确接触。
- MVP 包含主要非相邻连杆自碰撞检查与约束。相邻/设计内重叠几何使用明确允许列表,不得整体关闭自碰撞。

### 6.3 近似误差与独立验收

胶囊中心线采样若间距为 h,精确 SDF 的 1-Lipschitz 性可给出最近采样距离对应的 h/2 空间误差界;若再使用插值 SDF,还需加入已验证的插值误差上界。没有这些条件时不能从离散采样推断完整胶囊安全。

MVP 不依赖 128³ SDF 网格。后续若扩展网格 SDF,需按物体尺度选择分辨率,测试表面附近、棱角、内部与边界外查询,保存距离符号、误差和梯度约定。

最终验证使用锁定 collision 模型的 MuJoCo 碰撞/距离查询,独立于优化的采样近似,检查全部未被允许列表排除的碰撞对,而非只检查优化器当前激活的约束。验证点包括关键帧及按运动幅度自适应细分的帧间插值;报告最大检查间隔,不宣称对任意连续时间或视觉网格具有形式化零碰撞证明。

## 7. G1 生成模型

### 7.1 输出与 token

- 每帧 30 个动作 token:1 个根 token + 29 个关节 token;不需要多构型 padding。
- 根 token 预测位置 3D 与旋转 6D 表示;转换到物理状态时正交化为 SO(3),退化或非有限输出触发拒绝/重采样。
- 关节 token 预测一个关节角。不同 token 类型使用独立输入/输出投影,避免将根特征解释为关节角。
- 人体条件、目标物体轨迹、静态箱体尺寸、地面和接触条件使用交叉注意力。源参考根通道与待预测 G1 根 token 分开。
- 起步 F=60、时间注意力窗口=31。网络可先用 d_model=240、4 层、6 头,这是借鉴论文规模的 pilot 配置,最终依显存和验证结果确定。

### 7.2 扩散参数化

使用 DDPM 训练、DDIM 采样、**x0-prediction**。联合扩散归一化后的根位置、6D 旋转表示和 29 维关节角;物理旋转由输出投影得到。记扁平联合维度为 3+6+29=38。

```text
u_joint_clean = 2 * (q - q_lo) / (q_hi - q_lo) - 1
u_joint_pred = 2 * sigmoid(joint_logits) - 1
q_pred = q_lo + (u_joint_pred + 1) * (q_hi - q_lo) / 2
```

只约束干净角度预测,不裁剪前向加噪状态或把 sigmoid 用于高斯噪声。根位置用训练分组统计标准化;旋转 6D 使用固定表示,损失按尺度加权。剔除或修正超限教师标签并记录,不静默裁剪标签。

采样器基于同一个归一化 `x0_pred` 重建噪声并做 DDIM 更新,末步解码为物理 `(root, q)`。训练 1000 个噪声时刻、采样 25 步起步,消融 5/10/25 步;不默认开启 CFG 或推理时能量引导。

噪声日程、`add_noise` 和 DDIM `step` 优先复用 `diffusers.DDPMScheduler` / `DDIMScheduler`,设置 `prediction_type="sample"` 对应 x0-prediction。训练与推理复用同一组 beta 日程配置;关闭调度器默认的全状态裁剪/动态阈值(如 `clip_sample=False`,支持时设置 `thresholding=False`),避免错误裁剪根位置和旋转。关节限位由本项目干净角度输出映射负责。适配测试核对归一化、时间步选择、末步输出和固定种子复现,不手写 DDPM/DDIM 更新公式。

### 7.3 损失与确定性基线

训练损失包含带 mask 的干净状态重建、根旋转测地误差、教师 FK 关键点误差、手部接触偏差和时序速度误差。低权重 `E_L` 可作消融;各损失单位、归一化与权重在验证集上确定。几何损失不能代替修复验收。

确定性预测器使用相同条件、输出参数化和容量相近的时空网络,去掉噪声时刻及采样循环,直接预测 `(root, q)`。它与扩散使用同一训练划分、同一修复器和同一计时口径。

### 7.4 长序列与拼接

默认 60 帧窗口、stride=30。各窗口使用同一个 clip 坐标系,不单独重置世界原点。重叠区角度/位置采用确定的加权融合,旋转使用 SO(3) 插值;融合只是候选轨迹,随后整段顺序修复。

报告边界处关节/基座速度、加速度和接触连续性。跨窗口接触区间共享同一锚点状态。若拼接失败率高,再评估前缀条件生成或窗口联合优化,不通过放宽边界验收隐藏问题。

## 8. 顺序 QP 修复器

### 8.1 接触调度、锚点和可达性

- 调度从参考人的表面距离和相对速度推断,结合可用标注;使用进入/退出双阈值和最短持续时间去抖,人工核对 pilot。不能假定公开轨迹自带可靠抓取标签。
- 手锚点由接触建立时的人手表面点映射到指定箱体表面,使用校准的 G1 手部 site 外参;必要时用小规模 G1 可达性优化选择同一语义表面内的锚点,费用计入在线时间。
- 每个抓握区间内,手锚点固定在**目标物体系**。MVP 不允许搬运中重新选锚点绕过滑移约束。
- 支撑脚锚点固定在**世界/地面坐标系**。触地时从满足地面距离、足底姿态及进入速度条件的候选落脚状态建立;被接受后固定至离地。不能将悬空脚或未验证扩散状态直接注册为支撑锚点。
- 手部定义接触点与必要的法向对齐;平脚支撑使用多个足底 site 或位置加姿态约束。明显脚尖/脚跟滚动且不能由当前支撑模型表示的片段不属于 MVP 支撑假设,需明确标记。
- 初步筛查箱体轨迹、手臂可达范围、关节限位与接触时序;筛查只提供必要条件,通过不代表存在可行轨迹。基座允许调整,物体保持固定。

### 8.2 线性化子问题

在状态 `z_k` 附近优化 §6.1 的 `xi in R^35`,`z_prev` 是已接受的前一帧。用 Gauss-Newton 残差近似得到 PSD 二次目标:

```text
min  ||r_prediction(z_k) + J_prediction xi||^2_Wp
   + w_L ||r_mesh(z_k) + J_mesh xi||^2
   + ||r_temporal(z_k, z_prev) + J_temporal xi||^2_Ws
   + w_reg ||D^-1 xi||^2

s.t. q_lo <= q_k + delta_q <= q_hi
     |q_k + delta_q - q_prev| <= dt * joint_velocity_limit
     基座平移/相对旋转的线性化速度界
     d_j(z_k) + J_dj xi >= clearance_j                # non-contact pairs
     -penetration_tol <= d_c(z_k) + J_dc xi <= gap_tol # contact pairs
     |r_anchor(z_k) + J_anchor xi| <= anchor_axis_tol
     接触相对速度界、支撑足底姿态界
     ||D^-1 xi||_inf <= trust_radius
```

手锚点残差为 `R_obj^T (x_hand(z) - t_obj) - anchor_obj`;脚锚点残差为 `x_foot(z) - anchor_world`。手相对速度在物体系计算,脚速度在支撑坐标系计算;关节速度限制不能替代这两类约束。

`D` 为带物理单位的步长尺度矩阵,分别缩放基座平移、基座旋转和关节角。逐轴位置/速度界要按范数关系设置,例如每轴界取目标 L2 界的 `1/sqrt(3)`,并在更新后检查真实 L2 残差。

默认保留小权重 `E_L` 和时序目标;`w_L=0` 只作为消融,所有版本仍须通过任务一致性验收。Laplacian 是 soft cost,手足接触由显式约束承担。

非接触间隙从 2mm 起步,叠加有依据的几何近似误差。不得使用 `trust_radius * ||gradient||` 冒充线性化误差保证。信赖域配合真实候选检查控制非线性误差。

默认 OSQP 求解 QP。可选 SOCP 消融仅将归一化信赖域换成 L2 球并使用 Clarabel;保持其他目标/约束一致以便比较。名义相同半径的 L∞ 盒与 L2 球并不等价,需说明尺度匹配方法。

### 8.3 SCP 状态更新与失败处理

每次迭代重新计算当前状态的 FK、距离、雅可比及残差;仅静态资产、参考拓扑和固定锚点可跨迭代缓存。OSQP warm start 与稀疏结构复用不等于复用过期雅可比。

```python
def solve_frame(z_init, z_target, ctx, cfg):
    z = z_init
    trust = cfg.scp.trust_radius_init
    best_feasible = None
    for iteration in range(cfg.scp.max_iters):
        current = evaluate_nonlinear(z, ctx)
        if current.feasible:
            best_feasible = select_lower_objective(best_feasible, z, ctx)
            if current.task_ok and stationary(z, z_target, ctx):
                return accepted(z, iteration)
        linearization = linearize_at(z, ctx)
        step = solve_qp(linearization, z_target, trust, cfg)
        if not step.solved:
            trust, z = feasibility_restore_or_shrink(z, ctx, trust, cfg)
            continue
        candidate = retract(z, step.xi)
        trial = evaluate_nonlinear(candidate, ctx)
        if accept_by_merit_and_violation(current, trial, linearization, cfg):
            z = candidate
            trust = update_trust_radius(trust, current, trial, cfg)
        else:
            trust *= cfg.scp.shrink_factor
    return accept_if_fully_valid_else_reject(best_feasible, z, ctx, cfg)
```

这是流程伪代码,辅助函数需在 M2 定义。merit 使用目标与按单位归一化的约束违反度,根据实际/预测下降比接受步长;初始不可行时允许降低违反度的中间状态,但中间状态不输出。可行性恢复可临时引入惩罚 slack,退出仍必须满足原约束。不得在求解后独立裁剪关节角而不复核其他约束。

### 8.4 顺序 check-then-repair

```python
def repair_trajectory(prediction, condition, cfg):
    accepted_prefix = []
    contacts = initialize_contact_state(condition)
    for t, z_pred in enumerate(prediction):
        ctx = build_context(t, accepted_prefix, contacts, condition)
        if state_and_transition_valid(z_pred, ctx) and task_quality_valid(z_pred, ctx):
            result = accepted(z_pred, skipped=True)
        else:
            result = solve_frame(z_pred, z_pred, ctx, cfg)
        if not result.accepted and cfg.fallback.local_pure_optimization:
            result = recover_with_reference_objective(accepted_prefix, ctx, cfg)
        if not result.accepted:
            return rejected_with_diagnostics(t, accepted_prefix, result)
        accepted_prefix.append(result.state)
        contacts.commit_validated_transition(result.state, ctx)
    return validate_full_trajectory(accepted_prefix, condition, cfg)
```

首帧显式初始化支撑接触,不引用不存在的前帧。接触状态只在候选被接受后提交。候选碰撞检测可以批处理,但最终跳过判定必须引用已接受邻居,不采用原方案的固定两遍扫描。

`task_quality_valid` 只使用参考关键点、接触与交互能量,其绝对阈值在验证集上标定并冻结。在线检查不得读取目标 G1 真值或先完整运行教师优化来取得逐样本门槛;§10 的相对基线指标用于离线方法比较。

终检失败时定位窗口并回退到其起点,保留此前已接受状态及锚点,重新修复整个后缀;不能只替换一帧或拼接新噪声片段后直接输出。

### 8.5 切换与回退

- 接触分为 approach / hold / release。hold 强制距离、锚点和相对速度;切换区按参考阶段约束接近/离开,不把固定 ±2 帧全部设为无约束自由帧。
- 全程保持非穿透、自碰撞、限位与速度约束。接触建立必须验证距离/姿态和进入速度,解除接触后不再强行钉扎。
- 回退顺序:本帧可行性恢复 -> 从已接受前缀状态做带交互目标的局部纯优化 -> 可选重采样失败窗口并重跑后缀 -> 显式拒绝。候选种子数固定,所有候选耗时计入总时间。
- 固定预算耗尽、轨迹不可达或最终超容差时返回具体状态码;`needs_review` 不能伪装为 `accepted`。

修复器不持有生成模型。外层生成调度器根据 `TrajectoryResult` 的失败帧和前缀检查点执行扩散重采样,再调用修复器重跑后缀;前缀锚点必须随检查点恢复。纯优化回退使用参考交互目标,其初值取已接受前帧,首帧则使用参考估计的根位姿和 G1 名义姿态。确定性基线跳过无意义的换种子重试。

## 9. 训练与可选回放

### 9.1 教师数据与监督训练

1. 先跑通少量 G1 配对,检查根/手/脚/箱体对齐及教师整段可行性。
2. 教师数据也运行独立验收;不把 OmniRetarget 输出称为无条件可行真值。修正或拒绝超阈值样本,记录拒绝比例和分布。
3. 在同一分组上训练确定性预测器和 x0 扩散,训练资源按实际窗口数校准。
4. 使用本机 RTX 3090 24GiB 单 GPU pilot,按 §3.3 从 micro-batch=4、梯度累积=4 起步;记录精度、峰值显存、每步耗时、收敛曲线后再估计完整训练成本,不预承诺 GPU 天数。
5. 验证根轨迹误差、物体系关键点误差、接触偏差、裸输出/修复后质量和生成开销;只在验证集选权重、采样步数和修复阈值。

### 9.2 受筛选的回放训练(M5 可选)

这是优化器辅助的数据聚合,不声称具备经典 DAgger 的理论保证。仅使用训练分组的条件,保留原始监督数据占比,禁止测试参考和最终测试结果进入回放或调参。

```python
for cond in sample_training_conditions():
    prediction = model.sample(cond)
    result = repair_pipeline(prediction, cond)
    if result.accepted and reference_consistency_gate(result, cond):
        replay.add(cond, result.trajectory, provenance=result.diagnostics)
train_epoch(original_data, replay, replay_fraction=cfg.replay_fraction)
```

回放需同时检查整段可行性、接触 F1、锚点漂移、参考几何偏差及修复幅度,不能只凭修复残差下降入库。保留失败案例和按原始演示去重后的采样上限,避免简单样本占据回放池。

验收以固定验证集上的质量不下降和在线成本减少为准,同时报告回放生成与重训练成本。跳过率升高但任务偏差扩大视为失败,回退原模型。

## 10. 评测协议

### 10.1 指标及容差

| 类别     | 指标与定义                                             | MVP 门槛/报告项                                   |
| -------- | ------------------------------------------------------ | ------------------------------------------------- |
| 输出     | 整段接受率,分母含所有预先登记测试输入                  | 目标 ≥90%,相对纯优化下降 ≤5 个百分点            |
| 碰撞     | 机器人-物体、机器人-地面、物体-地面、自碰撞            | 接受轨迹最大穿透 ≤2mm;超过该阈值的检查时刻数为 0 |
| 限位     | 每关节越界量                                           | ≤1e-5 rad 数值容差,报告原始最大值                |
| 速度     | 关节速度、基座线速度和角速度                           | 不超过固定配置上界;值与来源在 M0-M2 锁定          |
| 脚接触   | 支撑期所有受约束 site 的世界系速度及累计锚点漂移       | 最大速度 ≤2cm/s,累计漂移 ≤1cm,足底距地 ≤5mm    |
| 手接触   | 接触期 site-物体表面距离、固定锚点偏差、物体系相对速度 | 表面间隙 ≤1cm、锚点偏差 ≤1cm、相对速度 ≤2cm/s  |
| 姿态     | 手部法向/支撑足底姿态                                  | 满足接触代理中锁定的容差,记录最大角度偏差         |
| 接触保持 | 从生成几何独立推断左右手/脚接触,与参考调度比较         | 分别报告 precision/recall/F1;手 F1 ≥0.95 × 基线 |
| 任务     | 物体系对应关键点 MPJPE、归一化 E_L                     | MPJPE ≤基线 +2cm;E_L 及分阶段偏差完整报告        |
| 时序     | 拼接/接触切换处速度、加速度及 jerk                     | 报告 p50/p95/max,尖峰定位;不能用均值掩盖边界失败  |
| 修复     | 根平移/旋转修正、关节 RMS/max 修正                     | 分开以 m/rad 报告,不混用统一 0.15rad 阈值         |
| 系统     | 跳过率、迭代数、回退率、失败原因、分阶段耗时           | 均值/p50/p95/max,拒绝案例也计时                   |

原论文的 contact preservation 与这里新增的 F1 不是同一指标,需要用相同独立评估器重新测基线。接触推断不直接复制输入调度;超过穿透容差的“手在箱内”不能算有效接触。终检还要复核非接触阶段违规接近、接触顺序及完整搬运区间,避免只生成局部静止姿态获得高分。

F1 同时报告已接受样本上的原始值和全测试集按演示宏平均的成功加权值,拒绝样本在后一指标中记 0;§1.3 的相对门槛采用后一指标。MPJPE 在共同成功样本上配对比较并列明样本数,同时强制整段接受率门槛,不为拒绝轨迹编造位置误差。

2mm 是所选 collision 模型和声明采样方案下的验收容差,不是“数学上零穿透”。碰撞拒绝、未知状态和人工审核均不算成功,人工筛掉的测试输入也不能从原始分母移除。

### 10.2 基线和消融

1. 固定版本的纯 OmniRetarget/Holosoma G1 优化,保留复现配置和原始指标。
2. 使用本项目相同几何、接触和终检标准的纯优化版本,用于匹配质量的速度对照;与原版结果分开报告。
3. 确定性生成裸输出 / 相同修复器后的输出。
4. 扩散生成裸输出 / 相同修复器后的输出。
5. 必要消融:DDIM 步数、w_L=0/非零、是否回放;L2/SOCP 和窗口策略按资源选做。

不再安排跨机器人实验。GMR 可作为附加参考,不是 MVP 必需依赖。

### 10.3 计时与有效加速

```text
T_total = T_reference_processing + T_generation + T_stitch
        + T_check + T_repair + T_fallback + T_final_validation
```

固定 CPU/GPU、并行度、批量大小、精度、求解器容差、冷/热缓存和随机种子;GPU 计时需同步。逐样本延迟与批量吞吐分别报告,各方法享有相同并行资源和可复用输入。

30% 跳过率仅在忽略其他开销且单帧修复成本不变时带来约 1/0.7=1.43× 的修复阶段加速。5× 目标需要实测证明更少迭代、更低每步成本等额外收益。

同时报告全测试集总耗时、成功轨迹/小时以及双方共同成功样本上的配对延迟比,并列出共同样本数。不能通过更早拒绝难例制造加速,也不能只计成功样本而隐藏失败和重试成本。教师生成与模型训练成本单列,可估算相对纯优化的使用次数盈亏平衡点。

### 10.4 回归和压力测试

固定验证分组中最多 32 段完整参考作回归集,与最终测试集隔离。覆盖拾取、移动、放下、长接触、短片段和窗口边界。额外构造根位姿扰动、关节噪声、近障碍、自碰撞、超速物体轨迹、不可达箱体和错误接触标签,验证修复与拒绝路径。

所有测试同时检查状态/帧间约束、输出状态码与诊断。仅扰动教师轨迹容易低估生成器误差,M3 之后必须加入真实预测分布下的修复评估。

## 11. 里程碑与依赖

> 核心顺序:数据配对和资产 -> 独立验收器与教师基线 -> 修复器 -> 生成模型 -> 闭环比较。工作量估计以一名熟悉运动学与学习框架的开发者为参考,需在 pilot 后更新。
>
> **状态摘要(2026-09-13)**:M0、M1 已完成并通过验收;M2 机制完成、验收部分达标(扰动接受率 60% < 95%、单帧 310ms > 50ms 未达;accepted 结果 100% 过独立终检、不可达 0 误受),收尾工作继续。M3 机制完成并通过验收(两类生成器完整输出 + 与真实预测误差对照;生成保真 MPJPE 8.9cm[扩散]/11.3cm[确定性],拼接边界无速度尖峰,DDIM 25 步 ≈8-10ms/帧;但**所有输入含教师噪声基线的整段修复接受率均为 0%**,与 M2 教师筛选结论同因——修复接受率被修复器收尾与教师数据重生成阻塞,见 `docs/verification/m3_verification.md` §5)。
>
> **P0 临时放宽协议(2026-09-13,`docs/plans/p0_relaxed_validation_plan.md` + `docs/archive/verification/p0_verification.md`)**:为解除上述 0% 阻塞而插入的基准建立阶段。核心产出:(1) 语义修复——逐对几何接触带分类("触地但未锚定")、准静态锚点建立/中途释放、转移插值检查对齐终检自适应细分、推断速度门 ≤ 检查门(支撑 := 准静态)、根速度行改范数(修复逐分量 vs 终检范数的存量不一致);附带修复跨段网格状态泄漏与终检定位缺失。(2) 成对放宽模式 `config_io.load_mode_configs`(修复带 ≤ 验收门不变式代码强制;碰撞/限位/速度界/脚支撑门不放宽)。(3) 结果:**strict 模式 0% → 2/6(语义修复单独);relaxed 模式 4/6=66.7%(6 段)/ 17/24=70.8%(标定集)≥ 50% 验收线**;task gate 标定 `mpjpe_tol_m` 0.40→0.101(17 段/1,999 帧,教师接受率零下降;对噪声/生成输入结构性过紧的选择偏置已实测并记录)。P0.3 闭环(8 val 段,0.40 工作值):裸教师 25%、噪声基线 12.5%、扩散 12.5%、确定性 25%——机制打通,扩散低于裸教师、与噪声基线持平,如实记录。**relaxed 产物一律不得计入 §10.1 达标**;教师悬浮 p95 19mm/手漂 p90 18cm 属数据本性,路径 B(教师重生成)论据增强。

> **P0 审阅修复(2026-09-14,`docs/archive/plans/p0_fix_plan.md` + `docs/verification/p0fix_verification.md`)**:对上述 P0 结果的四项代码审阅修复,未改任何容差数值。(F1) 手部准静态推断速度改箱体系——原世界系门使搬运相抓握区间构造性清零、P0 全部运行的手部检查形同虚设;修后 sub7 在 strict 下形成手部区间并通过认证,但教师悬浮(gap p95 33mm > enter 10mm)使其余全部运行区间仍为零,悬浮确证为数据本性。(F2) 转移检查改用完整前后场景(插值箱姿、碰撞族行限定、逐探测点重扫),运动量改箱体**角点扫掠**——关闭终检自身的纯旋转细分盲区。(F3) 终检失败结构化首帧定位 + 全检查取最小 + 无定位保守全段重试——原字符串解析按字典序取项且无定位失败退化为末 5 帧重试(必然失败),此为 P0 期 strict 低接受率的主因。(F4) 闭环口径重写:配对任务 MPJPE(修复后接受轨迹、共同成功集、机器人关键点 vs 参考人体关键点)、成功加权手 F1、slack 独立记录、产物元数据与模式/门槛命名;原 8.2/11.0-11.1cm 归类为裸输出教师拟合误差。**重测**:strict 6/6=100%、relaxed 24 段 22/24=91.7%(0.101 固定门 16/24);21/21 旧接受轨迹新终检仍过、零撤回;任务门重标 **0.113**(0.101 留作对照);闭环(0.40):裸教师 37.5%、σ0.02 噪声**三条件全满足**、扩散 12.5%(配对差 +4.2mm 但接受率不足)、确定性 25%(配对差 +7.8cm)——协议可达,约束在生成质量(4k 步 + 悬浮教师),教师重生成论据进一步增强。测试 99 项全过。

> **路径 B 执行(2026-09-15,`docs/archive/verification/teacher_regen_verification.md`)**:教师重生成机制落地(`RepairConfig.regen_mode` + `scripts/regenerate_teachers.py`:参考驱动主动手部接触 + 闭合走廊 + 手部关键点专项目标,strict 筛选 + 物理接触覆盖 ≥50%)。先导(诊断 6 段全长度):**2/6 保留**(接触覆盖 90-100%、任务 MPJPE p95 0.072-0.076、手部认证区间在教师侧首次成形);四段拒于拉入时刻转移检查与教师自带穿透起始帧(后续工作)。过程中修复 **slack 松弛缺 s≥0 下界的 M2 期存量 bug**、新增步长平方标度的插值下探余量、**手部门按参考分布重标**(锚漂 0.01→0.02、相对速度 0.02→0.8 反甩脱、进入 0.02→0.05;脚部不动)——标准模式 6 段 strict 基线 6/6→4/6,全部基线待 M2 收尾统一重测。测试 102 项全过。

> **收尾阶段门审阅与修复计划(2026-09-17)**:正式 v4 仍为 2/6,阶段 0/1 尚有 pair 类型污染、QP 目标尺度、成功返回复核及诊断统计缺口;临时类别修正 3/6 不计验收。先按[阶段门修复方案](../archive/plans/m2_m3_stage_gate_fix_plan.md)修正确性并重建六段基线,再按证据实施路径约束/接触稳定策略,达到至少 4/6 与质量要求后进入后续。旧“教师穿透/120mm 滑步输入不可恢复”等归因已被原始输入与候选区分的证据订正,见[验证记录订正](../archive/verification/m2_m3_completion_verification.md)。旧 24 段标定清单含 19 train/2 val/3 test,历史结果不属于 train-only 标定;后续须冻结新 train24 并记录历史测试暴露。**本次仅制定方案,未实施新修复或重生成。**

> **阶段门修复实施(2026-09-22,`docs/archive/plans/m2_m3_stage_gate_fix_plan.md` + `docs/archive/verification/m2_m3_stage_gate_fix_verification.md`)**:按修复方案实施 F0-F2 与 F4 并分项提交(`736f676` F0+F1、`4ce1baa` F2、`c702096` §7.1 探测点约束、`50afc75` 箱体系雅可比 einsum 修正、`2253998` §7.2 稳定软目标默认关闭、`00e2ab4` `rejected_final` 归因、`3831eb2` §8、`23bc211` 对照工具)。**F2**:跟踪项契约 `q=2wJᵀr`、`P=2wJᵀJ`(旧实现平移偏大 50×、关节 10×),旋转块改用 `J_l⁻¹(e_rot)/D`;修正后步长变小改变了状态链,按方案不回退。**F4 §7.1**:插值中间态碰撞进入线性化与 merit(复合雅可比 `T=alpha·J_l(alpha·D)·J_l⁻¹(D)` 与中心差分对拍 ~1e-10、独立探测行预算、成功退出前全量重探)。**F4 §7.2**:参考驱动箱系手部稳定软目标(权重 0/0.1/1.0 预注册)结果为负——0.1 使 sub10 左认证归零、1.0 以认证深度换数量,保留权重 0。**六段门**:regen 5/6 keep(sub12 速度终检两轮同因)、认证交集帧 sub10 3/13、sub8 27/40(均 ≥ v3)、标准模式 6/6 无撤回;每侧几何接近覆盖 2/12 项偏差(sub10 右 −5.4pp、sub8 右 −1.25pp,定位到 F1 状态链的瞬时脱离深度)未达标,按方案 §9 不自动转段(未运行 train24/312/重训)。§8 已冻结 train-only `calibration24_train_v1`(旧 `p02_calibration` 保留作历史)。测试 134 项全过。

> **阶段门实现审阅与修复计划(2026-09-23,2026-09-24 实施完成)**:对 `736f676^..cb8e4c0` 的审阅确认 9 项缺陷:探测距离截断值误入 QP、接触法向反号、回流缓存未失效、MPJPE 增长未逐帧 FK、最后失败归因错配、重试成本漏导出、探测违反行偏移错误、尝试事件过早定稿及有效配置记录错误。按[实现审阅修复方案](../archive/plans/m2_m3_stage_gate_review_fix_plan.md)先修诊断/报告并验证行为一致(D 对照:5/5 保留 NPZ 与冻结最终版逐字节相同),再修数值/缓存并重测六段。**统一版本结果(MuJoCo/OSQP 与容差零改动)**:R1-R9 分项提交(`c8d266f`/`b9fc080`/`a22e775`/`c991f6b`/`12c5741`/`9658b02`),165 项测试;Nq/Nv regen 5/6 keep(sub12 速度终检问题消除并接受、sub7 出现可复现的 @t=101 速度终检拒绝,两轮同因)、Ns 标准模式 6/6 与旧标准逐段一致(3 段 QP 迭代数因探测行纠正变化)、“每侧几何接近覆盖不低于 v3”仍有 2 项偏差(sub10 右 −5.4 pp、sub8 右 −1.25 pp,S01/S10 证明与稳定权重无关);历史 `mpjpe_growth`/q1/q5 已逐帧重算(旧“误差随段显著下降”解读作废),旧 5/6、6/6 仅作历史对照。strict/keep/任务工作门 0.40 不变,**不自动转 train24**。详见[修复验证记录](../archive/verification/m2_m3_stage_gate_review_fix_verification.md)。

> **标签分级与 train24 受限验证(2026-09-26,`docs/plans/m2_m3_label_quality_train24_plan.md` + `docs/archive/verification/m2_m3_label_quality_train24_verification.md`)**:教师数据质量准入与小规模验证。证据订正:统一版本相对 v3 为 **3 项**退化(上段"2 项偏差"中的 sub10 左认证交集实为 **0 帧**——认证总数 3 落在参考窗口外;"不可能由稳定权重恢复"撤回;sub7 拒绝的结构化速度证据:帧 101、关节 26、超限 4.0136e-7 rad/s,两轮一致)。落地分级标签协议 v1(`configs/labels/graded_labels_v1.yaml`:动作/FK、几何接近标记、认证交集监督、拒绝四类用途,旧质量门继续未通过)与消费接口(`export_label_manifest.py` 清单+掩码、`TeacherWindowDataset(label_manifest=…)`、损失分级监督;186 项测试)。六段回归与冻结基线逐段一致后,train24(`calibration24_train_v1`,三次运行字节一致)结果:**13/24 keep < 18/24 预注册数量门未达标**——失败分类为终检速度微超限 3 段(3e-9~4e-7 rad/s)、regen 手带行不可行 6 段、接触召回不足 2 段;六受试者均有可用窗口(40 窗/60.3 s)、局部认证监督左/右 50/90 帧。**按预注册规则停止规模扩展:312 originals 重生成与干净标签重训不启动**,待失败类别最小修复后重新冻结重跑。

### M0:数据与 G1 资产审计(约 1-2 周)

- [x] 检查本机 CPU/内存/GPU/存储,固定 Python 3.12 与 uv 最小项目环境。
- [x] 按 §3.5 逐模块核查成熟库 API,在 Python 3.12 uv 环境中锁定所需依赖,验证 PyTorch GPU 前向/反向及 MuJoCo 加载,记录资源和磁盘预算。(2026-09-11,torch 2.14.0+cu126 / mujoco 3.13.0,记录见 `docs/verification/m0_verification.md` §1;后续训练栈依赖按里程碑继续引入)
- [x] 固定 G1 29-DoF、箱体、关节顺序、碰撞对及手足 sites。(派生 `derived/robots/g1_29dof_locked*`,31 命名 sites,`G1Spec` 固化,spherehand 备选保留)
- [x] 获取公开小样本与对应处理版 OMOMO,核对实际字段、位姿顺序、帧率和配对。(71 样本审计:NPZ 四元数在前布局确认、PT↔NPZ 精确相似变换、时间轴一致,`docs/verification/m0_verification.md` §3)
- [x] 不能恢复配对时跑通 G1 教师生成,保存来源与增强元数据。(配对已恢复,该路径当前不需要;教师生成工具仅在自产数据时启用)
- [x] 确定独立分组清单、可用时长/数量、接触推断阈值和场景边界。(2026-09-11 于 M1 工具就绪后完成:250/31/31 按演示分组、受试者分层、种子 20260911,清单锁定 `derived/data/split/`;总量 2.998h/312 演示/1,826 NPZ;场景边界与阈值锁定于 `configs/task/box_carry.yaml`、`configs/eval/protocol.yaml`,见 `docs/verification/m0_verification.md` §3.5)

- **验收**:至少 10 段可信配对或明确的数据不足报告,包含渲染和数值对齐证据;未打通配对不启动大规模监督训练。(**已达成**:24 original + 47 增强变体验证,数值 + EGL 渲染证据)

### M1:运动学、交互表示与独立验收(约 1-2 周)

- [x] 适配 `pytorch_kinematics`/MuJoCo 原生 FK,验证根增量约定及 35 维雅可比。(1000 状态流形 FD 验证 max 3.9e-9;torch/MuJoCo FK 对拍 max 7.1e-7 m;MuJoCo freejoint 角速度为体系的实证已记录,见 `docs/verification/m1_verification.md`)
- [x] 接触区间、手锚点、脚触地锚点和逐帧交互网格。(双阈值+最短时长推断、hull 参考度量、锚点逐 site;逐帧 Delaunay+均匀 Laplacian,刚体不变性测试通过)
- [x] 复用 MuJoCo 距离/碰撞查询,仅补充必要的可微距离辅助函数,完成主要自碰撞和独立终检。(碰撞对+允许列表、解析 AABB 为训练代理[hull 差 mean~7cm 已量化]、`IndependentValidator` 含真自适应细分、表面距离间隙、显式有限性门与 status 状态码)
- [x] 纯优化基线评估;锁定容差和参考任务质量门槛。(未约束 GN 跟踪器:整段 MPJPE 4.9cm、10.6ms/帧、求解首帧 2.6cm;容差与阈值锁定于 `configs/eval/protocol.yaml`,推断 enter ≤ 验收门槛由代码强制)

- **验收**:已知违规样本能被检测,坐标变换测试通过,完整教师轨迹质量可量化。(**已达成**:6 类注入违规全部检出;不变性/等变测试通过;71 段教师质量报告——穿透 ≤23.2mm、脚滑 ~0.3m/s、接触 F1 0.657 defined-only,教师确非可行真值)
- 审阅修复轮(2026-09-11 第二轮):F1 双空虚高、脚部球心当表面、AABB 代理进标定、基线首帧 yaw、有限性检查、死配置六项已修并重标定重锁(旧值与失效原因保留在 protocol.yaml 与 m1_verification.md §8)。

### M2:顺序 QP 修复器(约 2 周,关键路径)

- [x] 带根增量的目标/约束线性化、OSQP 复用与 SCP 接受规则。(2026-09-12:`repair/{context,constraints,scp}.py`;刚体最近点距离雅可比 FD 验证;QP 每迭代重建,warm-start 复用列为后续优化;修复穿透梯度符号翻转等 8 项数值 bug,见 `docs/verification/m2_verification.md` §2)
- [x] 不可行恢复、接触状态机、顺序检查、终检和后缀回退。(slack 恢复帧内粘滞、锚点仅从接受状态锁定、1/3-1/2-2/3 转移插值检查、修复态复检、后缀重试保留前缀锚点;59 项测试全过)
- [x] 教师扰动恢复、不可达输入拒绝与长序列漂移测试。(实验已运行:教师整段接受率 0%——筛选结论喂给 M3 数据准备;不可达 0 误受;扰动集接受率 60% < 95% 目标未达,成因与达标路径见 `docs/verification/m2_verification.md` §4)

- **验收**:预先定义的可恢复扰动集整段通过率目标 ≥95%,所有返回 accepted 的结果独立终检通过;同时报告耗时分布和不可行案例,不以帧成功率替代整段成功率。(**部分达标,未勾选**:扰动接受率 60%/目标 95%、单帧 310ms/目标 <50ms;accepted 结果 100% 过独立终检 ✓、不可达 0 误受 ✓、不可行案例如实报告 ✓。机制完成,收敛与计时收尾继续)

### M3:确定性与扩散生成(约 2 周)

- [x] 固定 G1 token、条件接口、根/关节联合输出与 x0 参数化。(2026-09-13:38 维联合状态 = clip 系根位置[训练分组标准化]+rot6d+关节限位映射,30 token/帧独立投影;条件 = 52 关键点双坐标系 + 箱体轨迹 + 接触调度 + 静态几何,不含目标真值根/关节;完整 §4.3 schema 与修复层适配)
- [x] 确定性监督训练、Diffusers DDPM/DDIM scheduler 适配、60 帧窗口及拼接。(x0-prediction `prediction_type="sample"`、`clip_sample=False`,完美模型末步精确还原/种子复现/闭式 add_noise 有测试;同主干两封装;60/30 窗口线性交叉淡化 + 四元数 SLERP,接缝速度低于内部 p95)
- [x] 单 GPU pilot 后制定训练预算,记录实际模型规模/训练成本。(d_model=240、4 层、6 头、FFN×4、时间注意力窗口 31;两模型各 4000 步 ≈64/61 分钟、859/803 ms/步、峰值 5.47GiB,best val 0.096/0.246;~10 epoch 未饱和,完整预算约 1.6h/模型起步,`docs/verification/m3_verification.md` §4)
- [ ] (新增,源自 M2 筛选结论)教师数据重生成或清洗:教师整段修复接受率 0%(支撑滑移 ~0.3m/s、抓取漂移超任务门槛),直接作为修复输入不可行;`task_mpjpe_tol_m` 亦须在验证集标定(方案 §8.4)。M3 验收实验再次证实:教师噪声基线与生成输出同因 `rejected_frame` 全拒,该问题现阻塞 §10.1 整段接受率 ≥90% 的输出目标。 **2026-09-24 状态:实现审阅修复(R1-R9)已完成,统一版本 regen 5/6 keep、标准模式 6/6;“每侧几何接近覆盖不低于 v3”仍有 2 项偏差且与稳定权重无关,须先完成标签用途限制与质量协议修订的审阅,满足全部门后再按原条件推进 24/312 段。生成/闭环侧须在教师重生成后重测。**

- **验收**:两类模型均能输出完整状态,比较裸输出与真实预测误差下的修复质量,不以”优于随机动作”作为唯一证据。(**已达成**,`docs/verification/m3_verification.md` §5:两模型完整状态输出、限位/四元数合法、退化 0;8 段 val 对照中生成误差 MPJPE 8.9/11.3cm 与 σ≈0.02-0.05 教师噪声同量级,修复接受率与教师基线同为 0% 并如实报告失败类别;裸输出 16/16 被独立终检以 `penetration, foot_support, hand_grasp` 拒绝——可行性由修复器负责)

### M4:闭环与最终评测(约 1-2 周)

- [ ] 全流程状态码、诊断、质量报告和包含回退的计时。
- [ ] 验证集锁定配置后,运行最终独立测试集及必要消融。

- **验收**:§1.3 与 §10 的质量/报告标准完整执行;若速度未提高,明确报告加速假设未成立并定位瓶颈,不直接宣称目标完成。

### M5:受筛选回放(可选,约 1 周)

- [ ] 训练分组内生成/修复/筛选、来源追踪和回放比例控制。
- [ ] 固定验证集比较质量、在线成本及额外训练成本。

- **验收**:质量不退化且修复成本下降;否则保留 M4 模型。无跨构型任务。

### M6:RL 数据接口(可选,单独估时)

- [ ] 从已接受轨迹导出 G1 root/q、按指定约定计算的线/角/关节速度、物体位姿及接触信息。
- [ ] 格式转换/重采样后再次检查接触与碰撞,记录坐标和帧率。

- **验收**:目标加载器正确读取并渲染完整交互。20s sim2sim 跟踪属于后续扩展,未做前不宣称动力学或 RL 可跟踪性已验证。

M0-M4 粗估约 7-10 周,数据补齐和教师生成可能延长关键路径;M5/M6 不计入核心工期。

## 12. 关键设计决策

| #   | 决策                                      | 理由/验证                                         |
| --- | ----------------------------------------- | ------------------------------------------------- |
| D1  | 仅固定 G1 29-DoF 与单箱体                 | 减少资产和数据不确定性,取消多构型接口             |
| D2  | 配对条件中分离源/目标物体轨迹             | 支持正确的场景增强,防止标签错配                   |
| D3  | 生成 root+q,修复 35 维切空间增量          | 全身接触需要基座自由度                            |
| D4  | 物体系交互条件 + 保留地面/重力/参考根信息 | 避免丢失行走和支撑关系                            |
| D5  | 逐帧拓扑、均匀 Laplacian、显式接触锚点    | 软交互能量与硬接触职责分开                        |
| D6  | 成熟 FK 库 + MuJoCo 对拍;解析箱体距离     | 优先验证约定与几何正确性                          |
| D7  | x0-prediction,有界映射仅用于干净关节角    | 与扩散噪声模型保持一致                            |
| D8  | L∞ 信赖域 QP 默认,L2/SOCP 可选           | 避免混淆速度约束与锥约束来源                      |
| D9  | 顺序检查及整段独立终检                    | 使用最终邻居,发现传播/拼接违约                    |
| D10 | 小权重 E_L 默认,w_L=0 消融                | 保留参考交互,不把纯投影视为任务保证               |
| D11 | 接触状态机 + 经验证后锁定锚点             | 避免自由切换帧和悬空钉扎                          |
| D12 | 确定性/扩散共用修复器                     | 检验扩散收益是否覆盖多步成本                      |
| D13 | 回放需整段质量筛选                        | 防止将可行但偏离任务的输出反复强化                |
| D14 | 成熟依赖优先,标准算法通过薄适配复用       | 自定义代码集中在交互语义和验证,具体例外先记录缺口 |

## 13. 风险与应对

| 风险                                         | 影响                 | 应对                                                  |
| -------------------------------------------- | -------------------- | ----------------------------------------------------- |
| 单箱体配对数据少,公开文件缺少来源/接触元数据 | 监督训练阻塞或过拟合 | M0 先审计,自跑 G1 教师,按独立演示数报告               |
| 固定物体轨迹对 G1 不可达                     | 硬约束无解           | 预筛查、有限回退、显式拒绝;不改变物体轨迹隐瞒失败     |
| 手部代理与足底模型不适合参考动作             | 接触约束错误         | 校准 sites/法向,人工检查 pilot,清楚界定可支持接触阶段 |
| 几何近似漏检或尺度误差                       | 误接受穿透           | 独立碰撞模型检查,采样误差预算及帧间细分               |
| 局部线性化/紧接触导致 QP 不可行              | 修复失败或迭代过多   | 单位缩放、merit 接受、可行性恢复、局部纯优化          |
| 窗口拼接与顺序误差累积                       | 长时程接触/速度失败  | 共享锚点、累计漂移检查、后缀重修和全段复核            |
| 学习模型忽视参考条件或回放漂移               | 可行但任务错误       | 参考质量门槛、独立接触 F1、原数据占比与模型回退       |
| 扩散开销超过优化节省                         | 无端到端加速         | 确定性对照、采样步数消融、完整计时,接受负结果         |
| 运动学通过但不能承重/平衡                    | 下游 RL 失败         | 限定结论,后续单独进行动力学或跟踪验证                 |

## 14. 接口、配置与参考

### 14.1 核心张量

| 名称            | 形状                            | 约定                                                       |
| --------------- | ------------------------------- | ---------------------------------------------------------- |
| q               | `[B, F, 29]`                  | G1 固定关节顺序                                            |
| root            | `[B, F, 7]`                   | xyz + wxyz                                                 |
| 联合扩散状态    | `[B, F, 38]`                  | root position(3) + rot6d(6) + joints(29),实际按 token 分组 |
| 动作 token      | `[B, F, 30, d_model]`         | 1 根 + 29 关节                                             |
| 参考关键点      | `[B, F, K_h, 3]`              | 源物体系及必要的 clip 系通道                               |
| 源/目标物体位姿 | 各`[B, F, 7]`                 | 物理位姿存储,模型旋转编码为 6D                             |
| 接触调度        | `[B, F, 4]` + 有效性/阶段信息 | 四个语义末端,各可含多个 site                               |
| Laplacian 坐标  | `[T, K+P+E, 3]`               | 每帧 L_t,一一对应的 K 个解剖点                             |
| 修复增量        | `[35]`                        | 世界系根平移/旋转增量 + 关节增量                           |

### 14.2 关键接口

```python
class G1Kinematics:
    def keypoints(self, q: Tensor, root: Tensor) -> Tensor: ...
    def contact_sites(self, q: Tensor, root: Tensor) -> Tensor: ...
    def jacobian(self, q: Tensor, root: Tensor, site_ids: list[str]) -> Tensor: ...
    # [..., S, 3, 35], matching the tangent convention in section 6.1

class InteractionMesh:
    def build_frame(self, condition: ReferenceCondition, t: int) -> FrameMesh: ...
    def energy(self, state: RobotState, frame_mesh: FrameMesh) -> Tensor: ...

class G1Generator:
    def sample(self, condition: ReferenceCondition, seed: int) -> RobotTrajectory: ...
    # RobotTrajectory: root[T, 7], q[T, 29], dt, coordinate metadata

class RepairPipeline:
    def repair_trajectory(
        self, prediction: RobotTrajectory, condition: ReferenceCondition,
        cfg: RepairConfig,
    ) -> TrajectoryResult: ...
    # accepted: bool; trajectory only for accepted results
    # rejected results may carry diagnostic partial states, never exportable trajectories
    # diagnostics: frame_status, residuals_by_unit, iterations, skipped, fallbacks, timing

class IndependentValidator:
    def validate(
        self, trajectory: RobotTrajectory, condition: ReferenceCondition,
    ) -> ValidationReport: ...
```

### 14.3 配置起点

以下为 pilot 起点,不是已验证的最优超参数。限位/速度、接触姿态容差从锁定资产和协议读取,不使用无限大默认值。

```yaml
robot_asset: g1_29dof_locked
solver: osqp
scp:
  max_iters: 10
  trust_norm: inf
  trust_radius_init: 1.0
  step_scale: {root_translation_m: 0.02, root_rotation_rad: 0.05, joint_rad: 0.1}
  shrink_factor: 0.5
  nonlinear_acceptance: true
  feasibility_restoration: true
  allow_output_slack: false
objective:
  prediction_weight: 1.0
  laplacian_weight: 0.1
  temporal_weight: 0.1
  regularization_weight: 0.001
  normalize_residual_units: true
geometry:
  object_distance: mujoco_geometry
  noncontact_clearance_m: 0.002
  self_collision: true
validation:
  penetration_tolerance_m: 0.002
  joint_limit_tolerance_rad: 0.00001
  hand_gap_m: 0.01
  hand_anchor_error_m: 0.01
  hand_relative_speed_mps: 0.02
  foot_ground_gap_m: 0.005
  foot_speed_mps: 0.02
  foot_anchor_drift_m: 0.01
  independent_geometry: true
  check_interpolated_states: true
trajectory:
  mode: sequential
  contact_transition: state_machine
  revalidate_full_trajectory: true
fallback:
  local_pure_optimization: true
  additional_resample_attempts: 2
  recompute_suffix: true
  reject_on_failure: true
sampling:
  prediction_type: x0
  clip_frames: 60
  temporal_attention_window: 31
  stride: 30
  ddim_steps: 25
  guidance: none
```

### 14.4 参考链接

- [Multi-Embodiment Robotic Retargeting via Guided Diffusion Model, v2](https://arxiv.org/html/2505.20857v2)
- [OmniRetarget, v3](https://arxiv.org/html/2509.26633v3)
- [OmniRetarget 公开数据卡](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/blob/main/README.md)
- [Holosoma 重定向说明](https://github.com/amazon-far/holosoma/blob/main/src/holosoma_retargeting/README.md)
- [Holosoma 当前交互网格实现](https://github.com/amazon-far/holosoma/blob/main/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py)

公开数据和代码链接会更新;M0 将实际使用的下载版本/提交、资产哈希和转换配置写入数据 manifest,而非依赖可变的 main 链接作为复现记录。
