# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-22 · 两次单次授权 dispatch 均已消费并 fail-closed，在 source/checkpoint payload 读取与数值计算前退出，参数更新 0。独立复核在 `87d5b86` 上通过 70 项定向测试；当前 executable packet `attempt_032` 绑定全新运行根 `attempt_003`，未授权，后续运行需新的明确授权。**

## 结果

生成器读取四份已封口元数据：R6 admission、D6 独立裁决、D6 artifacts index、I/C `attempt_006` CUDA acceptance；另读取并 SHA-256 绑定 39 个当前实现/方案文件。它不打开被索引指向的 `initial.pt`、`probe_inputs.npz`、root-stats NPZ、数据集或生产资产，也不重新哈希历史绑定的 41 个资产。R6 的历史绑定仅作为已封口元数据引用；本轮未重新验证其 payload 当前仍与登记相符。

当前材料目录：`derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/attempt_032/`

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `admission.json` | `2f437b2cb7e1b8a02743dc292a04840d909436e23aac67fe1e99a6b0907eb09f` | `executable=true`、`authorized=false`、独立复核绑定 |
| `authorization_draft.json` | `d9eaee639dce6eaf27ba592b5cf9e5bb6d8bcef5308f396cfed93d8b70f2a5d7` | `authorized=false`、`template_only=true`、六项 checklist 为 null |

Admission canonical metadata SHA 为 `b2fcd5e93335f6dee3a842811ba0fcdacbe7c197bbf9d74b656d7b13dc80d8c4`。它绑定 D6/R6/I/C 固定元数据、六窗与 41 项资产记录、39 文件实现闭包、review receipt `independent_review/attempt_011/review.json` SHA `e0aed353…`（commit `87d5b86`、70 项测试）和 package-pinned worker argv/environment。真实运行根为新鲜的 `attempt_003`，生成时不存在；packet executable 但仍 `authorized=false`。

材料重申单次运行 300s 硬上限（240s 工作+60s 收口）、一次 attempt、参数更新 0。冻结 call matrix 分列来源读取、NPZ 成员、生产 scene/setup、只读 Trainer 构造、数值调用及禁止更新；worker I/O matrix 分列 48 个来源文件、60+2 个 NPZ 成员访问、20 个 NPY 与 8 个 JSON 工件写入、独立复核读回 27 次/哈希 20 次；supervisor matrix 登记消费事件、启动、实际 wait、日志、cleanup 检查上限与返回回读。运行时 worker meter 在生产 setup 调用点逐项计数，数值核心的 hook/VJP meter 复核前向与梯度调用，NPZ decoder、artifact store、独立读盘复核记录实际 I/O；supervisor 对 worker result 中的矩阵、运行时身份和来源 SHA 逐项比对 admission 包。真实值仍未运行；当前证明只来自临时 worker。

## Fail-Closed 边界

`scripts/supervise_w1_d6_shared_gradient.py` 对未 executable 的 packet 仍在授权文件读取与 spawn 前拒绝，退出码 3；对 executable packet 才进入签名授权校验与一次性 worker dispatch。其合成进程监督和 signed-path fake-worker 已通过实际 `wait()`、超时回收、日志定额、ledger 与 pending return receipt 测试。失败发生在 dispatch 后时，CLI 记录 `dispatch_started_outcome_unknown`，不谎报 worker 未启动或授权未读取。

验证：最新独立复核执行 compileall、worker/supervisor signed-path 门和相关 I/C 定向回归 **70 passed**；review receipt `independent_review/attempt_011/review.json` 绑定 commit `87d5b86`、39 文件闭包，复核本身真实 payload reads/worker spawns/updates 均为 0。测试覆盖 CPython isolated-mode 环境一致、worker preflight failure receipt、asset roster metadata 结构及未使用运行根推进。

## 一次性授权执行（两次失败，2026-11-22）

第一次用户明确授权生成 `authorization/attempt_001/authorization.json`，SHA `56aa3ad5…`，绑定 packet `attempt_027`。首次 supervisor 预检因 `attempt_root.parent` 不存在而退出，账本为空且未 spawn；补建父目录后同一授权进入一次 dispatch。Worker 在数值模块导入/源读取前因 locale 环境不匹配退出；账本 `56aa…` 保留 consumed/launch_started/failed，运行根 `attempt_001` 保留失败收据。该授权不可重用。

第二次用户明确授权生成 `authorization/attempt_002/authorization.json`，SHA `e36a7c7a…`，绑定 packet `attempt_030`、一次 attempt、300s/240s 和 0 更新。Worker 在来源读取前以 `admission_asset_source_roster` 退出。失败工件记录 source reads/hashes、checkpoint loads、asset reads、construction 和 updates 均为 0；attempt `002/d6` 为空，无 artifacts。Supervisor wait exit 2、未超时、进程组已回收、日志未溢出；账本 `e36a…` 保留 consumed/launch_started/failed。该授权已消费，不能重试。

locale 错误由 `7ad5b2c` 修复，早期失败留痕由 `94d8fab` 修复。第二次失败发现真实 package 的资产 roster 位于 `source.d6_bindings.production_assets.files`，而 worker 错误读取 `source.production_assets.files`；commit `4a280a2` 修正读取层级。随后 commit `87d5b86` 将真实运行根推进到从未使用的 `attempt_003`。当前 packet `attempt_032` 经独立复核；两份既有授权均已消费，不可重用。后续 dispatch 必须绑定 `attempt_032` 并获得新的明确授权。

## Supervisor 合成复核（2026-11-21）

`supervise_process()` 的启动边界、实际 `wait()` 返回、cleanup 完成和最后同步回读分开记录。Worker 在新 session 中启动，因此 PID 与 process-group ID 相等并与 supervisor PID 不同。父进程分别排空两个 pipe，日志最多保留登记上限字节但继续读取以避免 pipe deadlock；超量会使见证不可消费。截止超时会向整个 group 发送 SIGTERM，宽限后 SIGKILL，并实际 wait 主进程、检查 group 与排空线程是否消失。若 cleanup 不能被证明，见证拒绝。

返回观察文件采用一次性创建、文件与目录 fsync、no-follow regular-file 回读和 inode/字节核验。第二次消费回读与 JSON/绑定核验完成后才取得 final monotonic sample；此后不再执行文件 I/O。落盘文件只能是 pending receipt，只有同进程私有见证允许 `verify_external_return()` 消费。覆盖条件同时要求 worker 在工作 deadline 前清理完成，最终 sample 在登记墙钟与 hard deadline 内，exit code 为 0、未超时、group 已回收且日志未溢出。受控启动向 `Popen` 传入 package 冻结的环境映射，不继承 supervisor 环境；解释器绝对路径和 worker script 必须都在授权 launcher 与实现闭包绑定中。

`run_synthetic_attempt()` 仍仅接收显式 `execution_mode=synthetic_test_only` 的包。真实协议入口 `run_authorized_attempt()` 要求 signed authorization、`executable=true`、冻结的一次性额度、精确 real worker 身份和授权绑定的 `argv_template`；在工作起点后先以授权文件 SHA 创建不可重放 ledger directory，再创建 attempt claim。Ledger 采用 O_EXCL 事件文件：`consumed.json`、`launch_started.json`、`candidate_validated.json`、`return_candidate.json`。worker 启动命令只能由已绑定模板及固定 launch-spec 路径生成，不接受调用方临时命令；启动前后实现闭包摘要相等才允许候选进入返回见证。失败后消费 marker 保留，不重置额度或自动重试。

真实协议入口已有 temp-tree fake worker 贯通。独立 CLI 在数值模块导入前验证 bounded/no-follow launch、授权绑定、一次性消费 receipt 和 launch event。两次真实 worker spawn 均在 source payload 读取前拒绝：locale 门一次、asset-roster metadata 门一次；source/checkpoint/archive/asset payload reads、模型/FK/VJP 和参数更新均为 0。Supervisor 实际 wait 并回收进程组，失败账本不释放额度、不重试。旧 D6/R6 授权不适用于此流程。

## 保留尝试

`real_admission/` 根目录下 `attempt_001/` 至 `attempt_032/` 全部保留。授权/运行 attempt `001` 与 `002` 均为已消费失败记录。当前 `attempt_032` 是 source roster 与 root advancement 修复后的独立复核 packet，仍 `authorized=false`，绑定全新运行根 `attempt_003`。新 dispatch 必须对该 packet 新签一次性授权；已消费授权不重用、不自动续签。
