# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-22 · 十八次单次授权 dispatch 均已消费并 fail-closed，参数更新 0。第十八次完成 runtime、前向、损失、七次 VJP 和 outcome 写入，但独立 review 发现 outcome 数组 roster 混入后续 JSON 收据而拒绝。已修正 outcome 数组绑定；packet `attempt_049` 绑定空运行根 `attempt_019`，独立复核通过，仍未授权。**

## 结果

生成器读取四份已封口元数据：R6 admission、D6 独立裁决、D6 artifacts index、I/C `attempt_006` CUDA acceptance；另读取并 SHA-256 绑定 39 个当前实现/方案文件。它不打开被索引指向的 `initial.pt`、`probe_inputs.npz`、root-stats NPZ、数据集或生产资产，也不重新哈希历史绑定的 41 个资产。R6 的历史绑定仅作为已封口元数据引用；本轮未重新验证其 payload 当前仍与登记相符。

当前材料目录：`derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/attempt_049/`

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `admission.json` | `3911dddc884555548e7639fc7bcf094c703a63d5d4d5ff27c0825242e63fb5f9` | `executable=true`、`authorized=false`、独立复核绑定 |
| `authorization_draft.json` | `557afb8ba061ccb3f0c717671699693a4a826c16e7d5ed9c444139516c7f5d18` | `authorized=false`、`template_only=true`、六项 checklist 为 null |

Admission canonical metadata SHA 为 `a851e7235028943cee0d28623538f4f0a04c1107de0a482b18ecdc6db04efafb`。它绑定 D6/R6/I/C 固定元数据、六窗与 41 项资产记录、39 文件实现闭包、review receipt `independent_review/attempt_028/review.json` SHA `39b2b2e3…`（commit `8788092`、76 项测试）、package-pinned worker argv/environment、runtime evidence/FK source hashes、D6 原生 runtime-report schema 与实际 step0 tensor-check 列表、request/response 配对回执契约、outcome 仅数组文件绑定契约、`cpu_threads=20` 环境及完整 scene setup call matrix。当前 packet 绑定全新 `attempt_019`，生成时该根不存在。

材料重申单次运行 300s 硬上限（240s 工作+60s 收口）、一次 attempt、参数更新 0。冻结 call matrix 分列来源读取、NPZ 成员、生产 scene/setup、只读 Trainer 构造、数值调用及禁止更新；worker I/O matrix 分列 48 个来源文件、60+2 个 NPZ 成员访问、20 个 NPY 与 8 个 JSON 工件写入、独立复核读回 27 次/哈希 20 次；supervisor matrix 登记消费事件、启动、实际 wait、日志、cleanup 检查上限与返回回读。运行时 worker meter 在生产 setup 调用点逐项计数，数值核心的 hook/VJP meter 复核前向与梯度调用，NPZ decoder、artifact store、独立读盘复核记录实际 I/O；supervisor 对 worker result 中的矩阵、运行时身份和来源 SHA 逐项比对 admission 包。真实值仍未运行；当前证明只来自临时 worker。

## Fail-Closed 边界

`scripts/supervise_w1_d6_shared_gradient.py` 对未 executable 的 packet 仍在授权文件读取与 spawn 前拒绝，退出码 3；对 executable packet 才进入签名授权校验与一次性 worker dispatch。其合成进程监督和 signed-path fake-worker 已通过实际 `wait()`、超时回收、日志定额、ledger 与 pending return receipt 测试。失败发生在 dispatch 后时，CLI 记录 `dispatch_started_outcome_unknown`，不谎报 worker 未启动或授权未读取。

验证：最新独立复核执行 compileall、worker/supervisor signed-path 门和相关 I/C 定向回归 **76 passed**；review receipt `independent_review/attempt_028/review.json` SHA `39b2b2e3…` 绑定 commit `8788092`、39 文件闭包，复核本身真实 payload reads/worker spawns/updates 均为 0。新增 outcome 数组 roster 正控/后续 JSON 收据排除反例；当前 packet 仍未授权。

## 一次性授权执行（两次失败，2026-11-22）

第一次用户明确授权生成 `authorization/attempt_001/authorization.json`，SHA `56aa3ad5…`，绑定 packet `attempt_027`。首次 supervisor 预检因 `attempt_root.parent` 不存在而退出，账本为空且未 spawn；补建父目录后同一授权进入一次 dispatch。Worker 在数值模块导入/源读取前因 locale 环境不匹配退出；账本 `56aa…` 保留 consumed/launch_started/failed，运行根 `attempt_001` 保留失败收据。该授权不可重用。

第二次用户明确授权生成 `authorization/attempt_002/authorization.json`，SHA `e36a7c7a…`，绑定 packet `attempt_030`、一次 attempt、300s/240s 和 0 更新。Worker 在来源读取前以 `admission_asset_source_roster` 退出。失败工件记录 source reads/hashes、checkpoint loads、asset reads、construction 和 updates 均为 0；attempt `002/d6` 为空，无 artifacts。Supervisor wait exit 2、未超时、进程组已回收、日志未溢出；账本 `e36a…` 保留 consumed/launch_started/failed。该授权已消费，不能重试。

locale 错误由 `7ad5b2c` 修复，早期失败留痕由 `94d8fab` 修复。第二次失败发现真实 package 的资产 roster 位于 `source.d6_bindings.production_assets.files`，而 worker 错误读取 `source.production_assets.files`；commit `4a280a2` 修正读取层级。随后 commit `87d5b86` 将真实运行根推进到从未使用的 `attempt_003`。当前 packet `attempt_032` 经独立复核；两份既有授权均已消费，不可重用。后续 dispatch 必须绑定 `attempt_032` 并获得新的明确授权。
locale 错误由 `7ad5b2c` 修复，早期失败留痕由 `94d8fab` 修复；资产 roster 层级由 `4a280a2` 修复，运行根推进由 `87d5b86` 修复。第三次授权绑定 packet `attempt_032`、SHA `9e3fa69d…e1347`，运行根 `attempt_003`；worker 完成 48 个来源文件（含 41 资产、root-stats 和 6 个 D6 文件）的登记读取后，在 `root_stats_value_contract:mean` 处退出。原因是训练侧 root-stats 为 float64，而 worker 原先只接受 float32；checkpoint load、模型构造、FK/VJP 和参数更新均为 0。该授权已消费，账本和失败工件均保留，不重试。修复提交 `daf6bf6` 将下一候选根推进到 `attempt_004`，当前 packet `attempt_033` 需新的明确授权。
locale 错误由 `7ad5b2c` 修复，早期失败留痕由 `94d8fab` 修复；资产 roster 层级由 `4a280a2` 修复，运行根推进由 `87d5b86` 修复。第三次授权绑定 packet `attempt_032`、运行根 `attempt_003`，在来源 root-stats dtype 门失败；该授权已消费。`daf6bf6` 接受训练侧 float64 root-stats，并将下一 packet 推进到 `attempt_033`/运行根 `attempt_004`。第四次授权 SHA `b10769b6…4c4b` 绑定 packet `attempt_033`；worker 完成 48 个来源/资产文件读取并完成 1 次 checkpoint CPU load，随后在 `prepare_snapshot_inputs()` 的旧 float32-only root-stats 比较处退出。只读直接对照显示 source mean/std 与 checkpoint mean/std 的 float32 投影逐位相等，因此该次失败是实现契约缺口，不是 source 数据漂移；模型构造、FK/VJP 和参数更新均为 0。修复提交 `d4fefe0`，并将新候选根推进到 `attempt_005`。
locale 错误由 `7ad5b2c` 修复，早期失败留痕由 `94d8fab` 修复；资产 roster 层级由 `4a280a2` 修复，运行根推进由 `87d5b86` 修复。第三至第七次授权的 root-stats、packet runtime evidence、WorkerBudget stage 和 FK source 修复历史保持在前序记录。第八至第十一及第十二次授权、conditioner 输入修复和 packet `attempt_042` 根冲突见前序段落。第十三次授权绑定 packet `attempt_043`、运行根 `attempt_013`；来源/setup/模型构造/strict load通过，conditioner 输入窄化后完成一次预测、损失、两次 decode/FK 与七次 VJP，保存 `h_0..h_6` 和 `g_0..g_6` 数组，参数更新 0。最终 `runtime_guard.finish()` 调用 `r6.verify_runtime_report`，因 admission schema 错用 `w1_d6_shared_gradient_step0_v1/...`、而 D6 verifier 要求 `w1_d6_joint_fit_lr_half_v1/section_6_3_runtime_v1`，以 `real_runtime_report_missing:D6` fail-closed；候选未封口，不将数组裁决为通过。修复 `9e8be76` 对齐原生 schema 并加断言，`89cde79` 将根推进到 `attempt_014`。
**最近授权执行补记（2026-11-22）：**第十三次绑定 `attempt_043`/根 `attempt_013`，前向、损失和七次 VJP完成；runtime schema 与 D6 verifier 不匹配，proof 拒绝封口。第十四次绑定 `attempt_044`/根 `attempt_014`；schema 修正后 tensor-check 清单不匹配。第十五次绑定 `attempt_045`/根 `attempt_015`；相同计算已完成，但 verifier 仍抛通用 `binding_mismatch`，worker 失败工件当时仅记 runtime check 计数、没有 report facts，无法定位具体字段。三次均保留所有梯度数组与失败收据，不作通过裁决，参数更新 0。修复 `0999471` 让 step0 runtime verifier 对每个字段使用明确拒绝代码，并在失败收口保存已生成的 runtime report；正控与字段变异反例覆盖该路径。`002c992` 推进至空根 `attempt_016`；attempt_025 独立复核通过 74 项测试，当前 packet `attempt_046` 未授权。

第十六次明确授权绑定 packet `attempt_046`、根 `attempt_016`；39 文件闭包和全新根预检通过。数值核心仍完成一次预测、损失、两次 decode/FK、七次 VJP，参数更新 0。失败收据首次保留 runtime facts，字段级拒绝为 `step0_runtime_report_binding:tensor_check_points`。report 的 7 个实测检查点是 `fk_chain_built`、`fk_chain_transferred`、`fk_ready`、`loss_ready`、`initial_checkpoint`、`loaded_model`、`loaded_conditioner`；packet 多登记了 4 个 full-D6 trainer construction checks，而该窄诊断的 `construction_check` 回调本来只登记构造次数、不执行 runtime tensor inspection。修复 `a9c1198` 将 policy 精确收窄到上述实际点位；`4c5e415` 绑定未使用根 `attempt_017`，独立复核 attempt_026 的 74 项测试通过。新 packet `attempt_047` 为 `executable=true`、`authorized=false`，仍待针对该包的新一次性授权。

第十七次明确授权绑定 packet `attempt_047`、根 `attempt_017`；runtime report、前向/损失和全部七次 VJP通过，随后 outcome 回执组装 TypeError，候选未发布。修复 `a9d1756` 后，第十八次绑定 packet `attempt_048`、根 `attempt_018`；runtime、前向/损失、七次 VJP、outcome 和全部数组/回执写入完成，参数更新 0，但 outcome 的 `files` 是对可变 `store.records` 的引用，后续 `counts.json`、`call_matrix.json`、`vjp_records.json`、`loss_scalars.json` 写入后混入数组 roster，独立 review 以 `real_outcome_array_roster` 拒绝。修复 `2ddf9fc` 让 outcome 只保存 h/g/loss 数组记录，并加后续 JSON 排除回归；`8788092` 将候选根推进到 `attempt_019`，独立复核 attempt_028 的 76 项测试通过，packet `attempt_049` 未授权。

locale 错误由 `7ad5b2c` 修复，早期失败留痕由 `94d8fab` 修复；资产 roster 层级由 `4a280a2` 修复，运行根推进由 `87d5b86` 修复。第三次授权绑定 packet `attempt_032`、运行根 `attempt_003`，在来源 root-stats dtype 门失败；该授权已消费。`daf6bf6` 接受训练侧 float64 root-stats，并推进 packet `attempt_033`/运行根 `attempt_004`。第四次授权绑定 packet `attempt_033`；source/checkpoint stats 只读投影相等，但旧检查仍失败，修复 `d4fefe0` 后推进 packet `attempt_034`/运行根 `attempt_005`。第五次授权绑定 packet `attempt_034`；worker 完成 source/checkpoint/root-stats 绑定，在 production setup 因缺少 `runtime_evidence` 退出，修复 `ec4ed0b` 后推进 packet `attempt_035`/运行根 `attempt_006`。第六次授权绑定 packet `attempt_035`；runtime 构造时因 WorkerBudget 缺少 `stage=D6` 退出，修复 `ac85c59` 后推进 packet `attempt_036`/运行根 `attempt_007`。第七次授权绑定 packet `attempt_036`；FK source metadata 对账通过，随后 packet runtime evidence 缺失该 source roster，修复 `fdb9e27` 后推进 packet `attempt_037`/运行根 `attempt_008`。第八次授权绑定 packet `attempt_037`；worker 完成 source/checkpoint/FK runtime source hash 检查，runtime identity 因 pinned OMP/MKL=1 与 `cpu_threads=20` 不一致而拒绝，模型构造/FK/VJP/参数更新均为 0。修复 `c99bd90` 令 OMP/MKL 从冻结 runtime identity 派生，推进 packet `attempt_038`/运行根 `attempt_009`。第九次授权绑定 packet `attempt_038`、运行根 `attempt_009`；runtime identity 和 FK source binding 通过，但 worker setup 在 `scene_accesses` 前因冻结矩阵未登记该调用而拒绝，scene 初始化、模型/FK/VJP/参数更新均为 0。修复 `91d7f9c` 将一次 `scene_accesses` 与一次 `scene_cache_misses` 纳入 setup call matrix，并推进 packet `attempt_039`/运行根 `attempt_010`。第十次授权绑定 packet `attempt_039`、运行根 `attempt_010`；setup 通过 runtime/FK binding 及 scene access/cache miss，随后在 `scene_plain_xml_bindings` 处因漏登记 plain XML binding/compile 拒绝，场景编译、模型构造/FK/VJP/参数更新均为 0。修复 `875db5d` 登记两项各一次，推进 packet `attempt_040`/运行根 `attempt_011`。

## Supervisor 合成复核（2026-11-21）

`supervise_process()` 的启动边界、实际 `wait()` 返回、cleanup 完成和最后同步回读分开记录。Worker 在新 session 中启动，因此 PID 与 process-group ID 相等并与 supervisor PID 不同。父进程分别排空两个 pipe，日志最多保留登记上限字节但继续读取以避免 pipe deadlock；超量会使见证不可消费。截止超时会向整个 group 发送 SIGTERM，宽限后 SIGKILL，并实际 wait 主进程、检查 group 与排空线程是否消失。若 cleanup 不能被证明，见证拒绝。

返回观察文件采用一次性创建、文件与目录 fsync、no-follow regular-file 回读和 inode/字节核验。第二次消费回读与 JSON/绑定核验完成后才取得 final monotonic sample；此后不再执行文件 I/O。落盘文件只能是 pending receipt，只有同进程私有见证允许 `verify_external_return()` 消费。覆盖条件同时要求 worker 在工作 deadline 前清理完成，最终 sample 在登记墙钟与 hard deadline 内，exit code 为 0、未超时、group 已回收且日志未溢出。受控启动向 `Popen` 传入 package 冻结的环境映射，不继承 supervisor 环境；解释器绝对路径和 worker script 必须都在授权 launcher 与实现闭包绑定中。

`run_synthetic_attempt()` 仍仅接收显式 `execution_mode=synthetic_test_only` 的包。真实协议入口 `run_authorized_attempt()` 要求 signed authorization、`executable=true`、冻结的一次性额度、精确 real worker 身份和授权绑定的 `argv_template`；在工作起点后先以授权文件 SHA 创建不可重放 ledger directory，再创建 attempt claim。Ledger 采用 O_EXCL 事件文件：`consumed.json`、`launch_started.json`、`candidate_validated.json`、`return_candidate.json`。worker 启动命令只能由已绑定模板及固定 launch-spec 路径生成，不接受调用方临时命令；启动前后实现闭包摘要相等才允许候选进入返回见证。失败后消费 marker 保留，不重置额度或自动重试。

真实协议入口已有 temp-tree fake worker 贯通。独立 CLI 在数值模块导入前验证 bounded/no-follow launch、授权绑定、一次性消费 receipt 和 launch event。十八次真实 worker spawn 均 fail-closed；最近一次 runtime、VJP、outcome 和数组写入均完成，但独立 review 拒绝混入 JSON 收据的数组 roster，参数更新 0。outcome 数组绑定修复经独立复核；当前 packet 未授权。旧 D6/R6 授权不适用于此流程。

## 保留尝试

`real_admission/` 根目录下 `attempt_001` 至 `attempt_049` 全部保留；`attempt_042` 因候选运行根已存在而明确不可用，未获授权、未 dispatch。授权/运行 attempt `001` 至 `018` 均为已消费失败记录。`attempt_049` 是 outcome 数组 roster 绑定修复后的独立复核 packet，绑定空根 `attempt_019`，仍 `authorized=false`；真实运行需针对该 packet 新签一次性授权。已消费授权不重用、不自动续签。
