# 全项目 CPU 多核优化方案

日期:2026-10-05。状态:**预注册(实施前冻结);串行基线已登记(§3),实现与验收按 §8 交付顺序推进**。

本方案是纯工程性能轮:**不改变任何算法、profile、阈值与转段条件**。D+ 终局裁决(R1 1/6、R2 1/6、R3 2/6,winner=None)、train24/生产/重训暂停状态继续有效;本方案的任何运行不构成对候选矩阵的重开或对旧门的追认。

## 1. 目标与约束

覆盖教师重生成、QP 修复、FK、碰撞查询、独立验证、数据预处理、标签与审计、评估及训练的数据加载,提高批处理吞吐和可优化部分的单段速度。

- 默认最多使用 **16 个物理核**,允许显式调整到 20 核;默认不计入超线程收益。
- 保持轨迹数组**逐位一致**,以及求解顺序、失败身份(`status`/`t_fail`/attempt 与 check id)、迭代/QP 数、接触事件和质量门一致。计时字段、输出路径、代码版本(rev/dirty 证据)与新增执行元数据使用 §6.2 的**明确比较排除名单**。
- 保持全部算法、profile、阈值和转段条件。性能验收使用独立目录(§6.3),不写入任何既有实验根。
- 允许验收超过 8 小时;完整运行耗时单独登记,局部测量不代替完整验收。
- 默认路径兼容:`--workers` 缺省 1(串行)时走**原串行代码路径**(不引入进程池);已有 `--workers 4` 用法(precompute_m3_cache)继续有效。

## 2. 统一并行执行机制(实现契约)

- 标准库 `ProcessPoolExecutor` + **spawn**;任务按演示或独立审计记录调度,结果**按原输入顺序**汇总与打印(串行输出的逐 demo 块顺序保持)。
- CPU 批处理入口统一支持 `--cpu-budget 16`(缺省)、`--workers auto`、`--native-threads 1`;`auto` = min(可见物理核, 预算, 任务数),显式整数 worker 被采纳并截到 [1, 20]。已有 `--workers 4` 用法继续有效(等价显式 4)。
- **线程环境在数值库导入前设置**:父进程在创建进程池之前写入 `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB` 环境变量(spawn 子进程在 exec 时继承,早于其任何 import);当前进程内已加载的池用新增小型依赖 **`threadpoolctl`** 核验并限制,PyTorch CPU 线程用 `torch.set_num_threads` 控制(仅当 torch 已导入才触碰,避免给纯修复 worker 引入 torch 导入成本)。核验失败的兜底:保留原设置、收缩并发并把该进程亲和性钉到单核。
- 线程设置必须先通过逐位对拍(§6.1);不兼容时保留原设置,收缩并发并限制 CPU 亲和性。
- **外层演示并行时关闭内层并行**(worker 内 native-threads=1;MuJoCo/OSQP 本身单线程);单段串行运行不强制任何池。短任务按"含启动与传输成本"的测量保留串行执行(§4 清单逐项登记理由)。
- 每个任务隔离:MuJoCo model/data(每 worker 独立 `RepairPipeline`/`IndependentValidator` 实例,且 `frame_log`、`_vel_proj`、`_c2_diag`、锚状态本就按 `repair_trajectory` 调用重置)、OSQP 实例(FrameSolver 内部按帧创建)、接触/锚/发现碰撞对(帧边界清除,已有契约)、trace(诊断不回流)。测试显式验证跨任务/跨演示无残留状态。
- 父进程集中更新报告;任务异常、子进程崩溃(`BrokenProcessPool`)或缺失结果导致整批 `batch_status=incomplete` 与**非零退出**;续跑语义沿用既有 §12 状态机(只复用身份与产物哈希均通过的完整任务),本方案不新增部分续跑机制。
- 报告追加执行配置块:并行模式、workers、cpu_budget、native_threads、实际线程数(threadpoolctl info)、CPU 亲和性、spawn 上下文、提交/排队/计算/汇总耗时、逐任务时长与 worker 峰值 RSS(`resource.getrusage`)、失败任务身份。新增运行依赖(threadpoolctl)纳入完整性清单。

## 3. 串行基线(已登记,冻结)

优化前参照使用**已冻结的正式产物**,不重跑:

| 参照 | 内容 | 位置 |
|---|---|---|
| R3 六段串行(权威) | profile R3、strict、task gate 0.40、thread_env=8,rev `89d2365`;6 段逐段墙钟 880.8/1831.7/889.6/893.8/3601.5/588.6 s(合计 ≈8686 s,最长 sub7 3601.5 s);keep=sub16/sub7 | `derived/data/teachers_regenerated/train24fix3_recovery_20261003_258802c/recovery/R3/sixseg/rq/regen_report_diagnosis6_strict.json` |
| keep 段 NPZ | sub16/sub7 产物逐字节比较对象 | 同目录 `sub16_largebox_000.npz`/`sub7_largebox_000.npz` |

单段微基准(本方案立项测量,当前工作树,sub3 前 30 帧、N0 strict):墙钟 ≈2.0–2.3 s;cProfile 分布约 OSQP 求解 28%、`mj_forward` 15%、`mj_geomDistance` 9%、旋转数学(scipy `Rotation`/`quat_to_mat`/`so3_log`,**不可替换否则破坏逐位一致**)≈26%;`FrameEvaluator.forward` 602 次调用中 208 个唯一状态、**58% 为立即重复**;`geom_distance` 60302 次中 45389 唯一(25% 重复)。结论与方案提示一致:**主要加速来自独立任务并行**,单段收益以实测为准(§6.4 目标)。

## 4. CPU 路径清单(覆盖表;措施或保持串行的理由)

| 路径 | 措施 | 理由/事实 |
|---|---|---|
| 教师重生成 `regenerate_teachers.py` | **跨演示并行**(D2):单 demo 执行函数抽取,worker 级 `RepairPipeline`/validator 持久实例,输入序汇总 | 六段串行 8686 s、段间完全独立(种子化缓存+按调用重置状态);主要收益项 |
| 正式驱动 `train24fix3_convergence.py` | 移除 `_thread_env` 固定 8 线程假设(缺省保持 8=历史正式环境字节不变,`--native-threads`/`--regen-workers` 显式覆盖并向 regenerate 子任务传导) | 子任务即 regenerate/label export,预算传导;8 仅为历史缺省 |
| QP(`build_qp`/`solve_qp`) | **不实施**缓冲区复用/静态索引缓存 | 装配中无重复静态结构(仅 `np.eye(35)` 级分配);OSQP 求解占 28% 为不可替换 C 代码;复用预分配缓冲对数值序有零收益高风险,依"未获收益不进默认路径"排除 |
| FK(`FrameEvaluator.forward`) | **免恢复快路径**:请求的 43 维 qpos 与当前 data.qpos 逐位相同则跳过 `mj_forward`(全库唯一写点已核,`_probe_data`/`_motion_data` 独立) | 58% 立即重复;data 派生数组与 qpos 恒一致,跳过即逐位等价 |
| 碰撞查询(`geom_distance`/probe 变体) | **按状态缓存**:键=(当前状态 qpos 字节, geom 对, distmax),状态变更即清空 | 25% 重复;查询是纯函数,命中返回存储副本逐位等价 |
| 独立验证(`validation.py`) | **保持串行** | 30 帧段验证 0.1 s(墙钟 5%);逐 µs 碰撞查询的进程 IPC 开销远超收益;"先全部关键帧再插值探针"的合并语义与并列最坏值/首失败帧/细分封顶依赖单一有序遍历 |
| 数据/标签(precompute_m3_cache) | **逐文件并行接入统一池**(spawn+旗标,保持 `--workers 4` 缺省语义与输入序汇总) | 真实批量 1,826 文件,单文件独立推断,最大收益项 |
| 标签/质量/筛查/审计工具(export_label_manifest、segment_quality_report、screen_teacher_velocity、audit_data) | **保持串行**(实测裁定) | 批量 ≤ 数十项、分钟级、每次运行调用一次;重活仅在 keep 子集(validator 复检),spawn 启动+管线构建成本与整段串行运行同量级;"含启动与传输成本的测量保留串行" |
| 审计/重放(train24fix2/3 工具) | **保持串行**(默认) | 冻结帧/反事实/重放本身是取证链,身份绑定依赖原顺序;批量大时可用 `--workers` 显式加速,不改变判据 |
| 训练与生成评估(trainer/eval_generation) | DataLoader worker 内 native-threads=1(`worker_init_fn`);CPU 推理优化必须过相同数值契约 | CUDA 模型由主进程持有,CPU worker 不初始化 CUDA |
| 资产处理/可视化 CPU 部分 | 保持串行 | 一次性构建(锁定资产)或交互式渲染,无批量吞吐目标 |

## 5. 各计算路径的兼容性要求

- **教师重生成与修复评估:**段内逐帧采纳、SCP 迭代、动态切面、线搜索、回溯及后缀重试顺序不变;`--frames` 截断、screening、task 统计、NPZ 写出逻辑同一函数;verbose `frame_log` 逐 demo 语义不变(每次 `repair_trajectory` 重置,天然隔离)。
- **QP、FK 与几何:**保持 OSQP builtin 后端和每次求解行为;缓存只命中"同一已更新状态内的重复查询",状态更新后失效;矩阵运算及行顺序不变(缓存项 §4)。
- **独立验证:**不并行化;接触滞回、锚建立和累计漂移继续在完整有序序列上计算,独立于修复器。
- **数据与标签:**根统计和指标归约保持原数组顺序与计算方式;划分与哈希种子不变。
- **审计与驱动:**阶段前置门和候选执行顺序保持;处理数 `n==expected`、来源与完整性核验继续强制执行。
- **训练与生成评估:**CPU 推理优化必须通过相同数值契约(逐位)。

## 6. 验证与性能验收

### 6.1 逐位对拍(门 1,全部必须通过)

依次验证**旧实现**(=冻结串行基线 §3,不重跑)、**优化后单进程**(`--workers 1` 走原串行代码路径,与新代码逐位等同由构造保证+测试钉死)、**优化后多进程**:

1. 全量既有测试套件通过(346+1s 基线之上只增不减)。
2. 新增并行边界测试(§7)通过:乱序完成、重复演示、跨段状态隔离、随机输入一致性、插值首失败与并列最坏值(经既有 validation 测试间接覆盖——验证器未改动)、缓存失效、worker 崩溃、续跑与产物篡改(既有 §12 机制)。
3. 真实数据缩比对拍:diagnosis6 子集 × 截断帧,串行(workers=1)与并行(workers=2/4,native-threads=1)**逐行逐字段**相等(排除名单 §6.2)且 keep 段 NPZ 逐字节相等;线程环境差异(native=1 vs 历史 EnvVar=8)不改变数值(经验证:OSQP/MuJoCo/小矩阵 numpy 均单线程执行)。
4. 冻结 R3 六段的**完整并行对照**:并行运行 profile R3 全六段至独立目录,与 §3 冻结串行报告逐行比较(排除名单外全等):status/t_fail/accepted、`cost` 全部计数(迭代/QP/attempts/backtracks/checks)、`c2_hand_speed`、`regen_anchor`、`rejection` 证据、`hand` 召回/认证区间、`task_mpjpe`、`velocity_projection`;sub16/sub7 keep NPZ sha256 相等。任何一项不符即**验收失败**,定位差异属"内部优化"还是"调度改变"并停止。

### 6.2 比较排除名单(显式)

行级:`wall_s`、`total_s`、`solve_frames_s`、`final_validation_s`、`preprocess_s`、`screening_s`、`hand_recall_s`、`quality_stats_s`、`output_s`、`segment_wall_s`、`ms_per_*`、`cost` 内嵌计时键(`*_s`、`timings_s`)。报告级:`freeze`(时间戳/git/dirty/thread_env)、`end`、`provenance`、`execution` 新块、`effective.argv`(新增执行旗标)与 `effective.screening` 之外的执行字段。其余全部字段必须相等。

### 6.3 独立验收目录

`derived/data/teachers_regenerated/cpu_parallel_acceptance_20261005/`;不写入任何既有实验根;运行期间仓库零写入(R9 契约;验收运行前全部实现已提交)。

### 6.4 性能目标(门 2)

- 完整六段并行吞吐达到**理想并行上限的 ≥75%**(理想墙钟 = 冻结串行逐段成本的最大值 3601.5 s,按 §3 记录;加速比 = 串行合计/并行实测)。
- 单段端到端中位耗时不回退超过 5%(FK/距离缓存只允许变快或持平;若实测变慢,缓存退出默认路径)。
- 冷启动(含 spawn 与 worker 管线构建)与持久池分别报告;`1/2/4/8/16` 并发的短任务与均衡批量曲线按实测登记。

## 7. 新增测试(登记后实施)

`tests/test_cpu_parallel.py`:workers 解析(auto/预算/截断/非法值)、线程环境在 spawn 子进程内先于数值库导入生效(子进程自报 env+threadpoolctl info)、结果按输入序汇总(任务时长抖动)、worker 崩溃→整批 incomplete+非零退出、跨任务状态隔离(worker 全局可变状态逐任务重置)、重复演示不损坏产物、真实短段串行 vs 并行逐位对拍(rows 排除名单比较+NPZ 字节)、FK 快路径与距离缓存:同段开关缓存逐位相等+命中计数>0+状态变更后失效+首帧不误命中(全零哨兵反例)。

## 8. 交付顺序

1. **本文件**:方案登记 + CPU 路径清单(§4)+ 串行基线(§3)。
2. 统一资源控制(`src/ireset/resctrl.py` + `threadpoolctl`)与跨演示并行 + 严格兼容对拍(§6.1.1–3)。
3. FK 快路径与几何距离缓存(逐位一致测试钉死)。
4. 其余预处理/审计/评估/数据加载入口接入 + 驱动线程假设移除。
5. 完整验收(§6.1.4 + §6.4)、性能报告与验证记录、AGENTS.md 更新、提交。

## 9. 停止条件

逐位门(§6.1)任一失败即停:不发布并行默认路径、不降低比较标准、不以"仅诊断差异"放行;性能门(§6.4)未达时不移除串行路径、不扩大并发。任何属算法行为的差异(即使更优)按 R9 精神登记为失败并回退该优化项。
