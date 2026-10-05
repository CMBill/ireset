# 全项目 CPU 多核优化验证记录

日期:2026-10-05。实现提交:`462981a`(方案 `docs/plans/cpu_parallel_optimization_plan.md`,预注册于实现之前)。性质:**纯工程性能轮**——算法、profile、阈值、转段条件零改动;D+ 终局(R1 1/6、R2 1/6、R3 2/6,winner=None)与 train24/生产/重训暂停状态不变。本记录对应方案 §6 验收与 §8 交付顺序。

## 0. 结论速览

| 门 | 结果 |
|---|---|
| 逐位门(§6.1):真实缩比对拍(串行 vs 并行×2/×4、N0 与 R3 profile) | **通过**(报告逐行 IDENTICAL,keep NPZ 逐字节) |
| 逐位门:FK/距离缓存开关对比(真实 30 帧段) | **通过**(轨迹字节+诊断除计时逐位相等;skips 347/hits 12978) |
| 逐位门:六段 R3 并行 vs 冻结串行(§3 基线) | **通过**(6 行 IDENTICAL+sub16/sub7 NPZ sha256 相等;keep 集一致) |
| 逐位门:六段 R3 优化后单进程 vs 冻结串行 | **通过**(同上;缓存全六段规模逐位中性实证) |
| 性能门:吞吐 ≥75% 理想上限 | **通过**(2.331× = 理想的 96.6%) |
| 性能门:单段中位回退 ≤5%(缓存裁定) | **通过**(新串行/旧串行中位 1.007,合计 1.009×;逐段最大 1.015,sub7/sub8 反而 0.972/0.960) |
| 已测登记:并行批量内单段中位 1.066(调度成本) | **超 5%,登记为已知权衡**(内存带宽争用;吞吐 2.31× 的代价;延迟敏感单段运行用缺省 --workers 1) |
| 边界测试 12 项+全量套件 | **359 passed + 1 skipped**(基线 346+1s 净增 12,零回归;终跑 955s 全绿) |
| 性能门(§6.4):吞吐 ≥75% 理想上限 / 单段中位回退 ≤5% | 见 §5 |
| 串行默认路径不变 | workers=1 经 parallel_map 串行分支=原共享管线路径;执行记录统一 |

## 1. 交付内容(提交 462981a)

- `src/ireset/resctrl.py`:统一资源控制(§2 契约全部落地)。`resolve_workers`(`auto`=min(物理核 20,预算 16,任务数),显式整数截 [1,20]);`apply_native_threads`(环境变量先于进程池写入→spawn 子进程 exec 时继承=早于其任何数值库导入;当前进程已加载池由 threadpoolctl 运行时限流并经 `verify_native_threads` 复读核验;torch 仅当已导入才触碰——纯修复 worker 不付 torch 导入成本;限流异常兜底=保留原设置+钉单核亲和);`parallel_map`(spawn;结果与 `on_result` 严格按输入序;任务异常/`BrokenProcessLoop` 取消剩余并整批中止;逐任务 compute_s/worker pid/峰值 RSS)。
- `scripts/regenerate_teachers.py`:单 demo 执行抽取 `_process_demo`/`_emit_demo`(串行循环与并行父进程共用同一输出序列→stdout 逐块同序);`--workers/--cpu-budget/--native-threads`;并行=每进程持久 `RepairPipeline`+筛查 validator(worker initializer),MuJoCo data/OSQP/接触锚/发现碰撞对/frame_log 逐任务隔离(frame_log 本就按 `repair_trajectory` 调用重置);keep NPZ 由 worker 直写(numpy savez zip 时间戳 1980 固定=字节确定性);崩溃→`batch_status=incomplete` 报告+退出码 1;报告新增 `execution` 块(排除名单内)。
- `src/ireset/repair/constraints.py`(D3):`FrameEvaluator` 状态复用——(a) `forward`/`probe_forward` 免恢复快路径:请求的 43 维 qpos 与当前状态逐位相同即跳过 `mj_forward`(全库唯一 data.qpos 写点=forward 自身已核,派生数组与 qpos 恒一致,`_probe_data`/`_motion_data` 独立);(b) `geom_distance`/`probe_geom_distance` 按状态 memo(键=(当前状态,(ga,gb,distmax)),forward 状态变更即清空;命中返回独立副本,调用方可变契约不变)。`state_cache=False` 测试钩子;`cache_stats` 仅属性、不进任何报告。
- D4:`precompute_m3_cache` 统一池(spawn+三旗标,`--workers 4` 缺省语义保留、输入序汇总);`train24fix3_convergence` 移除固定 8 线程假设(缺省 8=历史正式环境逐字节不变,`--native-threads`/`--regen-workers` 向 regenerate 子任务传导);`train/trainer.py` DataLoader `worker_init_fn` 限 worker 单线程(env 先行+torch.set_num_threads(1),CPU worker 不触 CUDA)。
- 工具:`cpu_parallel_compare.py`(排除名单结构化对比+keep NPZ sha256)、`cpu_parallel_bench.py`、`cpu_parallel_acceptance_summary.py`。依赖 +`threadpoolctl==3.7.0`。

## 2. 逐位对拍证据(§6.1 门 1)

1. **缓存开关(真实数据)**:sub3 前 30 帧 strict N0,缓存开/关各跑一次——`status/accepted` 同;轨迹 `tobytes()` 逐位相等;诊断(剥离计时键与 `timing_stats`)全等;`cache_stats`:fwd 602 调用/skipped 347(58% 立即重复,与立项测量一致)、dist 60302/hits 12978、probe 命中 36。墙钟 1.84s vs 1.88s(1.02×;sub3 skip 帧占比高,收益以六段实测为准)。
2. **CLI 串行 vs 并行(真实缩比)**:sub3+sub16×25 帧 strict——`cpu_parallel_compare` **IDENTICAL**(行字段全等、排除名单外);sub3+sub16×12 帧 workers=2 同(测试内钉死,含 execution 块断言)。
3. **R3 profile 路径(真实深接触)**:sub3×4 帧(双侧 C2@1,恢复相+固定区间手锚全走通)串行 vs workers=2——**IDENTICAL**。
4. **线程环境轴**:并行 worker native=1 vs 串行父进程(默认多线程 BLAS 环境)结果逐位相等(上列 2/3 即为该轴证据:OSQP/MuJoCo 单线程、小矩阵 numpy 不分层)。
5. **既有冻结用例/profile 兼容**:求解路径唯一改动=可证明逐位中性的缓存(第 1 项);全量套件 359+1s 含全部冻结工件连续性测试(train24fix3 continuation/formal/recovery 等)通过。S1/S2 的 5 帧冻结帧与 113 反事实的完整重放未在本轮重跑——其行为等价由(缓存逐位中性+套件中钉死的冻结工件核验+六段验收)承载;如需独立重放,`train24fix3_convergence --stage recovery` 的 §12 复用机制可直接执行。
6. **边界测试**(tests/test_cpu_parallel.py,12 项):workers 解析(auto/预算/截断/非法)、spawn 子进程 env 先于数值库导入(子进程内自报)、输入序+时长抖动、任务异常与 worker 硬杀(os._exit)整批中止、跨任务状态隔离、重复任务、种子随机输入串行=并行、FK 首帧哨兵(全零请求不误跳过)+跳过+状态失效反例、真实 10 帧缓存开关逐位、真实 CLI 串行 vs 并行报告+NPZ 逐字节。

## 3. 调度器测量(§6.4 曲线)

- spawn 池冷启动 ≈0.15–0.20s(空载任务实测);**短任务保持串行**的交叉点由此实测登记:16×5ms 任务串行 0.076s < 任何并行配置(≥0.15s)。
- 均衡合成批量(16×~45ms BLAS 任务):串行 0.686s→w16 0.250s(2.7×;单任务计算量随并发上升 0.68→1.02s=内存带宽饱和,合成负载特性,不代表 MuJoCo/OSQP 真实负载)。
- 真实吞吐以 §5 六段为准。

## 4. 六段 R3 三方验收(§6.1.4 + 用户方案"旧实现/优化后单进程/优化后多进程"分解)

运行根 `derived/data/teachers_regenerated/cpu_parallel_acceptance_20261005/`(独立目录,既有实验根零写入;两次运行期间仓库零写入,`end.code_changed_during_run=False`):

| 运行 | 配置 | 逐位对比(vs 冻结串行) |
|---|---|---|
| 冻结串行(§3 基线,rev 89d2365) | 驱动 env=8 线程,旧代码 | — |
| 优化后多进程 | `--workers 6 --native-threads 1`,新代码 | `cpu_parallel_compare`:**IDENTICAL**(6 行全部算法字段+effective 除 argv/note;sub16/sub7 keep NPZ sha256 相等;keep 集 {sub16,sub7} 一致) |
| 优化后单进程 | `--workers 1`,env=8(同冻结),新代码 | 见 §5(缓存单段效果的直接测量) |

并行执行块:workers=6、native=1、collect_s=3726.8、每 worker 峰值 RSS ≤1.74GiB、R9 干净。

## 5. 性能门判定(§6.4)

**吞吐门:通过。** 加速比 = 冻结串行合计 8686.0s / 并行墙钟 3726.8s = **2.331×**,为理想上限(合计/最长段 = 8686.0/3601.5 = 2.412×)的 **96.6% ≥ 75%**;并行墙钟 ≈ 最长段(sub7)自身墙钟(3722.6s)。

### 5.1 三方分解(用户方案"依次验证旧实现、优化后单进程、优化后多进程,明确区分内部优化与调度改变造成的差异")

| 段 | 旧串行(s) | 新串行(s) | 并行(s) | 缓存比(新/旧) | 调度比(并行/新串行) |
|---|---:|---:|---:|---:|---:|
| sub10_largebox_000 | 880.8 | 892.6 | 954.0 | 1.013 | 1.069 |
| sub12_largebox_001 | 1831.7 | 1848.8 | 1930.0 | 1.009 | 1.044 |
| sub16_largebox_000 | 889.6 | 894.3 | 957.8 | 1.005 | 1.071 |
| sub3_largebox_000 | 893.8 | 907.6 | 965.5 | 1.015 | 1.064 |
| sub7_largebox_000 | 3601.5 | 3502.3 | 3722.6 | **0.972** | 1.063 |
| sub8_largebox_000 | 588.6 | 564.9 | 638.8 | **0.960** | 1.131 |
| **中位** | | | | **1.007** | **1.066** |
| 合计/墙钟 | 8686.0 | 8610.5 | 3726.8 | 1.009× | 2.310× |

**缓存裁定(门括号语义):通过。** 新串行 vs 冻结串行中位 1.007、合计 8610.5s vs 8686.0s(1.009× 净加速);逐段回退最大 1.5%(噪声量级),sub7/sub8(逐帧求解占比最高的两段)加速 2.8%/4.0%。缓存保持默认开启。

**并行批量内单段中位 1.066:超过 5%,登记为调度成本的已知权衡,不掩盖。** 归因:6 个稠密进程并发的共享内存带宽争用——形态与合成基准一致(16 任务计算量随并发 0.68→1.02s),最短最稠密的 sub8 受影响最大(1.131);**A/B 已排除线程设置**:同段(sub3×4 帧 R3,恢复相全开)串行 unset=169.14s / worker native=1=169.61s(+0.3%)/ native=8=166.32s(−1.7%),±2% 非主因。按方案 §9:串行路径保持缺省(`--workers 1`),并发为显式选择——批量吞吐 2.31× 以单段延迟 ~6.6% 为代价,延迟敏感的单段运行不进入并行路径。

**逐位门全六段规模实证**:并行与优化后单进程两次运行对冻结串行基线 `cpu_parallel_compare` 均 **IDENTICAL**(排除名单外全部行字段+effective 配置/profile/筛查参数;sub16/sub7 keep NPZ sha256 相等;keep 集 {sub16,sub7} 三方一致;两次运行 `end.code_changed_during_run=False`)。

## 6. 裁决与边界

- 全部改动不构成对候选矩阵重开或旧门追认;R3 并行运行产物位于独立验收目录,不写任何既有实验根。
- 未实施项及理由(方案 §4 覆盖表):QP 装配缓冲复用(无静态结构重复分配,数值序风险零收益)、验证器并行化(单段验证仅 ~5% 墙钟,µs 级查询 IPC 远超收益)、标签/质量/筛查/审计工具并行(分钟级一次性运行,spawn 成本同量级)。
- 单段串行路径(`--workers 1`)保持为缺省;六段门公式与比较排除名单见方案 §6.2。

## 7. 验收产物清单(cpu_parallel_acceptance_20261005/)

- `R3_parallel/sixseg/rq/regen_report_diagnosis6_strict.json` + `sub16/sub7.npz`(并行 6 workers 运行;execution 块在报告内)
- `R3_serial_opt/sixseg/rq/regen_report_diagnosis6_strict.json` + `sub16/sub7.npz`(优化后单进程,env=8 同冻结)
- `compare_vs_frozen.json` / `compare_serial_vs_frozen.json`(两次逐位对比,均 IDENTICAL)、`perf_gates.json`(吞吐/回退判定)
- `run_stdout.log`/`run_time.log`(两次运行的完整 stdout 与 /usr/bin/time 峰值内存)
- 复核命令:`uv run --locked python scripts/cpu_parallel_compare.py --baseline <冻结报告> --candidate <验收报告>`;`scripts/cpu_parallel_acceptance_summary.py --serial <冻结> --parallel <并行>`;三方分解脚本数字在 §5.1 表内可复算。

## 7.1 全量套件一次既有瞬态失败的定位与修复(本轮终跑前)

全量套件在优化后两次于 `test_report_contracts.py::test_regenerate_cli_report_has_effective_config` 失败,断言 `row["wall_s"] + 5e-3 >= row["total_s"]` 实测 `0.03 + 0.005 >= 0.035` 失败——**测试容差设计缺陷,非行为缺陷**:`wall_s` 为两位小数舍入字段,其 ±5ms 舍入半步恰好吞掉 5ms 容差,且二进制下 `0.03+0.005 < 0.035`(同款瞬态在提交 `89d2365` 的全量运行中已出现并记录"文件级复跑稳定");两次失败中底层计时差均在 5ms 契约内。修复=测试容差改为 `1e-2 + 1e-9`(吸收舍入半步与浮点表示,契约本身不变),修复后全量 **359+1s 全绿**(955s)。生产行为零改动。

## 8. 裁决

1. **逐位门:全部通过**(缩比、R3 深接触路径、全六段并行、全六段优化后单进程;NPZ 字节;keep 集)。并行执行不改任何算法行为,证据为全六段规模(含两个保留段与四个失败段的失败身份/t_fail/计数/rejection 证据逐字段相等)。
2. **性能门:吞吐通过(96.6% ≥ 75%);缓存单段门通过(中位 1.007 ≤ 1.05)**;并行批量内单段中位 1.066 > 5% 登记为内存带宽争用的调度成本(§5.1,已定因、不掩盖),处置=串行保持缺省、并发显式 opt-in。
3. 工程边界不变:D+ 终局、train24/生产/重训暂停状态不变;本轮任何运行不构成候选重开或旧门追认;验收全部位于独立目录。
4. 后续使用:批量重生成(六段/train24 规模)推荐 `--workers N`(预算 16/显式 20);延迟敏感单段用缺省串行;`precompute_m3_cache` 缺省 4 workers 语义保留;驱动正式阶段缺省环境(8 线程、串行子任务)逐字节不变,`--regen-workers/--native-threads` 为显式选项。
