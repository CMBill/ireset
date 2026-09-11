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

## 8. 遗留到 M2

- 速度限值正式标定(当前临时);手部 exit 阈值 pilot 复核(悬停段拉长区间)。
- 帧间插值检查的完整自适应细分(当前中点一级)。
- 约束版纯优化基线(QP);教师数据的修正/拒绝流程。
