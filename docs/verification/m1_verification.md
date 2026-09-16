# M1 验证记录

日期:2026-09-11。M1(运动学、交互表示与独立验收)实现与验证记录。配套产物:`derived/data/validation/{fk_validation,teacher_quality,baseline_quality}.json`、`configs/eval/protocol.yaml`。全部 31 项测试通过(`uv run --locked pytest tests/ -q`)。

## 1. 运动学层(方案 §6.1)

新增依赖:`pytorch-kinematics==0.10.0`、`rtree==1.4.1`(trimesh 邻近查询)。

| 模块 | 内容 |
|---|---|
| `kinematics/rotation_adapter.py` | 四元数 wxyz/xyzw 转换(唯一合法转换点)、SO(3) exp/log、6D 表示、torch 可微实现 |
| `kinematics/mujoco_adapter.py` | 批量 FK、35 维 retract(`p+dp`、`Exp(dθ)@R`、`q+dq`)、35 维雅可比;`RobotState` |
| `kinematics/torch_adapter.py` | pytorch_kinematics 可微 FK(URDF 链 + 浮动基座组合、site 偏移) |

**实证确认的约定**:MuJoCo freejoint qvel 线速度为世界系、角速度为**体系**,故 `J35 = J_mj @ blkdiag(I, R^T, I)`。35 维雅可比从不使用 7 维四元数存储导数。

**验证结果**(`scripts/validate_fk.py --n 1000`,双精度):
- 解析 vs 流形中心差分雅可比:相对误差 max **3.9e-9**(目标 <1e-6 ✓)
- torch vs MuJoCo FK(全部 31 sites):max **7.1e-7 m**(目标 <1e-6 ✓)

开发中修复的关键 bug(记录以防回归):retract 的 einsum 曾写成 `Exp @ R^T`;`world_to_object`/`world_to_box_geom` 的 R/R^T 缩并方向曾写反——往返测试(R·R^T=I)无法检出,由旋转等变/刚体不变测试捕获。

## 2. 交互网格(方案 §5.3)

`imesh/delaunay.py`(逐帧 Delaunay + 均匀权重 Laplacian 稀疏算子,退化输入固定种子扰动 + QJ 回退)、`imesh/transform.py`(世界↔物体系,torch 可微)。

- 点布局:`[0:K) K=15 解剖点 | [K:K+P) 物体表面固定采样 | [K+P:+E) 地面点(每帧变换入物体系,不附着物体)`。
- 每帧在**参考侧**建拓扑;目标侧与全部 SCP 迭代复用同一算子。
- 测试通过:算子平移不变/旋转等变;**全场景一致刚体变换下物体系 delta 不变**;能量零点;重复点退化回退。

## 3. 几何层(方案 §6.2/6.3)

`geometry/collision_pairs.py`:自碰撞对 = 非相邻(树距 >2)碰撞体对 + 文档化允许列表;环境对 = 机器人-地/机器人-箱/箱-地。允许列表卫检确保条目引用真实 body。

`geometry/distance_adapter.py`:解析点-箱(AABB)与点-面距离,torch 可微。

**两个重要发现**(已写入测试与配置):
1. **MuJoCo 编译时 mesh 重定心带旋转**:largebox 的 geom 系 = 存储网格系,yaw 偏约 48°。世界点必须经 `p_geom = R_g^T(R_b^T(p−t_b)−x_g)` 映射(R_g/x_g 由 `d.geom_xmat/xpos` 或静态帧给出)。验证:存储系映射下 MuJoCo hull 距离与 trimesh hull 距离 max 差 **2e-4 = 探测球 2r**(即映射精确)。
2. **AABB 仅是训练代理**:扫描箱(非水密、带圆角)凸包与 AABB 距离差 mean ~7cm、max ~14cm。验收一律用 MuJoCo hull 查询。

## 4. 接触推断与锚点(方案 §8.1/5.1)

`embodied/contact_sites.py`:SMPLH 52 关节名/索引表、手脚点集、双阈值(enter/exit)+ 最短时长去抖的区间提取;参考侧手(物体系表面距离)/脚(高度+速度门控)推断;手锚点=建立时手质心在箱 geom 系最近表面点,脚锚点=触地时世界位置。均为候选,经验证后才提交。

`eval/metrics.py`:帧级接触 precision/recall/F1(按末端 + 宏平均)。

## 5. 独立验收器(方案 §6.3/10.1)

`repair/validation.py`:`IndependentValidator` 消费场景 qpos[T,43],检查:
- 全部非排除碰撞对(含自碰撞)穿透,关键帧 + 帧间中点插值(自适应细分,报告最大检查间隔);
- 关节限位(1e-5 rad)与速度(临时值:关节 6 rad/s、基座 1.5 m/s / 3 rad/s,模型无速度限位,M2 标定);
- 脚支撑期速度/漂移/离地间隙;手抓取期间隙/锚点漂移/物体系相对速度(接触由**生成几何自身**推断,不复制输入调度)。

**检出测试**(6 项通过):注入穿透/超限/超速/支撑期滑移均被检出;NaN 不得通过;干净教师基线的几何检查工作正常。

## 6. 教师轨迹质量量化(M1 验收项)

`scripts/validate_teacher.py --n 4`(6 受试者 × 4 original,24 段,120 帧均值):

| 检查 | 通过率 | 最差值 | 门槛 |
|---|---|---|---|
| 关节限位 | 100% | −0.0087 rad(余量) | ≤1e-5 越界 |
| 速度 | 50% | 2.09× 临时限 | 1.0× |
| 穿透 | 0% | **16.5mm** | 2mm |
| 脚支撑 | 0% | 支撑期速度 ~0.30 m/s、漂移 ~3.9cm | 2cm/s、1cm |
| 手抓取 | 8% | 锚点漂移达 ~1.35m(含悬停段) | 1cm |
| 接触 F1(机器人推断 vs 参考推断,宏平均) | — | **0.872** | — |

**结论**:教师轨迹系统性地不满足项目验收门槛(软接触优化的真实属性:贴地滑移、深穿透、软抓取漂移)——证实"教师不是无条件可行真值",监督数据必须过筛(M3 前处理),也直接论证 M2 修复器必要性。参考/机器人两侧独立推断的接触 F1 0.872,说明推断链路自洽。

## 7. 纯优化基线(方案 §10.2)

`eval/baselines.py::InteractionTracker`:逐帧阻尼 Gauss-Newton(35 维切空间、信赖盒、限位裁剪——M2 改为硬约束),首帧由参考骨盆位置+肩线 yaw 估计根位姿、名义关节角,后续帧温启动。箱体轨迹为固定输入。

2 段实测(120 帧):**MPJPE 4.2cm、10.6ms/帧**(单核 Python/MuJoCo);验证不通过(穿透 8-10cm)——无碰撞约束的跟踪器如预期把机器人压入箱体,量化了"几何约束必须进优化/修复循环"。该实现是 M4 计时对照的未约束下界;完整约束版基线在 M2 QP 就绪后补齐。

## 8. 审阅修复与重新锁定(2026-09-11 第二轮)

代码审阅发现 6 个问题,按 [m1_fix_plan.md](../plans/m1_fix_plan.md) 四阶段修复。**机制先修、再重标定、后重锁**;以下数字全部为修复后口径。测试 51 项全过(新增 `test_metrics.py`、`test_contacts.py`、`test_baselines.py`,扩展 `test_validation.py`)。

### 修复内容

1. **F1 未定义策略**:双空末端(两侧皆无区间)不再白拿 1.0——记 `None` 并从宏平均剔除,报告 `n_defined`;单侧空记 0;新增全测试集成功加权口径(拒绝记 0)。
2. **阈值不变式 + 表面距离 + 逐 site**:推断 enter 阈值不得超过其区间必须满足的验收门槛(`__post_init__` 强制);脚部离地间隙改用 `mj_geomDistance` 球面真实距离(修正"球心 z 当间隙"的半径 bug——实测与 MuJoCo 逐值一致);漂移/速度/锚点全部逐 site(质心会漏检原地旋转,回归测试锁定)。
3. **参考侧改用 hull 度量**:新增 `largebox_hull_surface_distance`(凸包缓存);AABB 误差在判定区与阈值同量级(mean 1.8cm@真实 2cm),仅保留为训练代理。
4. **基线首帧根位姿**:yaw 公式修正(左肩在 +y,`atan2(-d_x, d_y)`);骨盆 site→root 偏移(本资产恰为 0,`pelvis_contour_link` 在根原点,已测明锁定)。求解首帧 MPJPE 8.1cm→**2.56cm**(与审阅者实测吻合)。
5. **显式有限性门**:`rejected_nonfinite` / `rejected_degenerate_quaternion` 状态码前置,NaN 箱体四元数不再令 scipy 崩溃;新增 `status` 字段(M2 `TrajectoryResult` 前置件)。
6. **自适应细分落实**:`max_substep_motion_m`/`max_subdivisions` 真正生效(原为死配置);笛卡尔量度含**箱体位移**(纯箱体运动也触发细分——测试中发现的真实缺口);`max_check_interval_m` 语义改为细分后间隔,超预算显式标注。

### 重标定(Phase C)

扩展网格 [0.005..0.06] + skip 策略 + hull 度量(18 段):宏 F1 全程平坦(0.662–0.668,弱辨识,列入 pilot 复核);锁定 ref_hand 0.04/0.072(比值 1.8 为记录在案的未标定假设);**ref_foot 退出网格**——82% 参考帧足高低于 0.02m,F1 饱和无辨识力,由不变式定为 0.005/0.02(文档说明不可由 F1 标定)。

### 修复后基线(Phase D,旧值保留对照)

| 项 | 旧锁定值(已失效) | 修复后 | 失效原因 |
|---|---|---|---|
| 标定宏 F1 | 0.876 | **0.668**(defined-only) | 双空记 1.0 虚高 + AABB 代理 |
| ref_foot 阈值 | 0.05/0.08 | 0.005/0.02 | F1 饱和假精度,改由不变式决定 |
| 机器人侧 enter | hand 0.02 / foot 0.012 | 0.01 / 0.005 | 不得宽于验收门槛 |
| 教师穿透最差 | 16.5mm | **23.2mm** | 自适应细分发现更深时刻;含增强变体(71 段) |
| 教师速度通过率 | 50%(24 段) | 31%(71 段) | 样本含增强;限值仍为临时 |
| 教师接触 F1 | 0.872 | **0.657**(defined-only) | 同标定口径修正 |
| 教师全段通过率 | 0% | 0%(不变) | 穿透独立支撑该结论 |
| 基线首帧(求解后) | 8.1cm | **2.6cm** | yaw 公式修正 |
| 基线整段 MPJPE | 4.96cm | 4.92cm | GN 后续帧本可收回,量级结论不变 |

脚部通过率仍为 0%:阈值伪失败已消除(贴地切触严格通过,回归测试锁定),残余为教师真实贴地滑移(~0.30 m/s)——真阳性,方案 §9.1 预期一致。

### 遗留到 M2(更新)

- 速度限值正式标定;手部 exit 阈值与悬停段区间 pilot 复核(锚点漂移最差 ~2.4m 出现在含悬停的长区间)。
- 约束版纯优化基线(QP);教师数据修正/拒绝流程。
