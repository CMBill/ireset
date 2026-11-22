# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-22 · 用户授权的一次运行已消费，但 worker 在数值/来源读取前因 Python locale 环境契约不匹配退出；修复并复核后生成 executable packet `attempt_029`。未读取 checkpoint/批次/资产 payload、未更新参数；attempt_029 未授权，需新的明确授权。**

## 结果

生成器读取四份已封口元数据：R6 admission、D6 独立裁决、D6 artifacts index、I/C `attempt_006` CUDA acceptance；另读取并 SHA-256 绑定 39 个当前实现/方案文件。它不打开被索引指向的 `initial.pt`、`probe_inputs.npz`、root-stats NPZ、数据集或生产资产，也不重新哈希历史绑定的 41 个资产。R6 的历史绑定仅作为已封口元数据引用；本轮未重新验证其 payload 当前仍与登记相符。

当前材料目录：`derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/attempt_029/`

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `admission.json` | `95e5234efab12c2ac20b0fe305097f8f33ec855de804c8045d8387b8d5029aa4` | `executable=true`、`authorized=false`、独立复核绑定 |
| `authorization_draft.json` | `613b4ea4a52a7c77c5687fd86e385e32464c1648bf9e01d4ef91896e5b9546bd` | `authorized=false`、`template_only=true`、六项 checklist 为 null |

Admission canonical metadata SHA 为 `0f8dac4afafd7d2e831d4c93372296f7eca2c348f050cb80a499830f08abe5bc`。它绑定 D6 裁决 SHA `1de06dae…`、R6 admission SHA `9e54107c…`、D6 artifact index SHA `c5cca9bc…`、I/C acceptance SHA `f05fc505…`、六窗登记路径、六份 D6 工件记录、41 项资产记录、39 文件实现闭包、独立复核收据、固定账本根 `derived/audit/w1_d6_shared_gradient_step0_v1/authorization_consumption_v1` 和 package-pinned worker argv/environment。独立复核 receipt `independent_review/attempt_008/review.json` SHA `524f39c7…` 绑定 commit `048ae3a`，定向测试 69 项通过。它确认 isolated Python 自动注入的 `LC_CTYPE` 已纳入冻结环境，并确认 worker preflight 失败会写失败工件。授权草案仍 unsigned；此前 `attempt_027` 的一次性授权绑定旧 packet，已消费并不可用于当前 packet。

材料重申单次运行 300s 硬上限（240s 工作+60s 收口）、一次 attempt、参数更新 0。冻结 call matrix 分列来源读取、NPZ 成员、生产 scene/setup、只读 Trainer 构造、数值调用及禁止更新；worker I/O matrix 分列 48 个来源文件、60+2 个 NPZ 成员访问、20 个 NPY 与 8 个 JSON 工件写入、独立复核读回 27 次/哈希 20 次；supervisor matrix 登记消费事件、启动、实际 wait、日志、cleanup 检查上限与返回回读。运行时 worker meter 在生产 setup 调用点逐项计数，数值核心的 hook/VJP meter 复核前向与梯度调用，NPZ decoder、artifact store、独立读盘复核记录实际 I/O；supervisor 对 worker result 中的矩阵、运行时身份和来源 SHA 逐项比对 admission 包。真实值仍未运行；当前证明只来自临时 worker。

## Fail-Closed 边界

`scripts/supervise_w1_d6_shared_gradient.py` 对未 executable 的 packet 仍在授权文件读取与 spawn 前拒绝，退出码 3；对 executable packet 才进入签名授权校验与一次性 worker dispatch。其合成进程监督和 signed-path fake-worker 已通过实际 `wait()`、超时回收、日志定额、ledger 与 pending return receipt 测试。失败发生在 dispatch 后时，CLI 记录 `dispatch_started_outcome_unknown`，不谎报 worker 未启动或授权未读取。

验证：独立复核脚本执行 compileall、worker/supervisor signed-path 门和相关 I/C 定向回归 **68 passed**；review receipt `independent_review/attempt_007/review.json` 绑定 commit `7ad5b2c` 和 39 文件闭包，复核本身的真实 payload reads/worker spawns/updates 均为 0。复核新增 CPython isolated-mode 下 `LC_CTYPE` 环境逐位一致正控。当前 `attempt_028` 为 `executable=true`、`authorized=false`；授权并非 packet 的副作用。

## 一次性授权执行（失败，2026-11-22）

用户明确要求授权运行后，生成 `authorization/attempt_001/authorization.json`，SHA-256 `56aa3ad5…`，绑定旧 executable packet `attempt_027` metadata SHA `487f621b…`、一次 attempt、300s hard cap/240s work cap、参数更新 0，六项 checklist 全 true。首次 supervisor 预检在 `attempt_root.parent` 尚不存在时退出，账本为空、未 spawn；补建指定父目录后使用同一授权进入一次真实 dispatch。该次账本已消费并封存：`authorization_consumption_v1/56aa…/consumed.json`、`launch_started.json`、`failed.json`；唯一运行目录为 `derived/experiments/omniretarget_first/w1_d6_shared_gradient_step0_v1/attempt_001/`。

Worker 实际 spawn 并在标准库预检中以 `worker_environment_not_package_pinned` 退出。原因是 CPython isolated 启动在原冻结环境未包含 locale 时自动注入 `LC_CTYPE=C.UTF-8`，与 package 的精确环境 dict 不一致。该检查位于数值模块 import 和 `read_authorized_source_bundle()` 之前；attempt 目录没有 `d6/` 或 `artifacts/`，没有 checkpoint/NPZ/source/asset payload reads、模型/FK/VJP、参数更新。Supervisor wait 返回 exit 2、timed_out=false、group_reaped=true、log_overflow=false；授权事件记录 `parameter_updates=0`，失败收口为 `incomplete`。按一次性协议，该授权与 packet 均不可重用，未重试。

随后在 commit `7ad5b2c` 将 `LC_CTYPE=C.UTF-8` 纳入 package-pinned worker environment，并新增隔离解释器环境正控。又修复 CLI 数值模块导入前失败时漏写 `worker_failure.json` 的缺口，commit `94d8fab`；定向测试扩为 **69 passed**。独立复核 `attempt_008` 在后续文档提交 `048ae3a` 上完成，生成当前 executable packet `attempt_029`。`attempt_029` 尚无授权；此前消费的 `attempt_027` 授权绝不能复用。任何再次 worker dispatch 都需要用户对当前 packet SHA 新签发的一次性明确授权。

## Supervisor 合成复核（2026-11-21）

`supervise_process()` 的启动边界、实际 `wait()` 返回、cleanup 完成和最后同步回读分开记录。Worker 在新 session 中启动，因此 PID 与 process-group ID 相等并与 supervisor PID 不同。父进程分别排空两个 pipe，日志最多保留登记上限字节但继续读取以避免 pipe deadlock；超量会使见证不可消费。截止超时会向整个 group 发送 SIGTERM，宽限后 SIGKILL，并实际 wait 主进程、检查 group 与排空线程是否消失。若 cleanup 不能被证明，见证拒绝。

返回观察文件采用一次性创建、文件与目录 fsync、no-follow regular-file 回读和 inode/字节核验。第二次消费回读与 JSON/绑定核验完成后才取得 final monotonic sample；此后不再执行文件 I/O。落盘文件只能是 pending receipt，只有同进程私有见证允许 `verify_external_return()` 消费。覆盖条件同时要求 worker 在工作 deadline 前清理完成，最终 sample 在登记墙钟与 hard deadline 内，exit code 为 0、未超时、group 已回收且日志未溢出。受控启动向 `Popen` 传入 package 冻结的环境映射，不继承 supervisor 环境；解释器绝对路径和 worker script 必须都在授权 launcher 与实现闭包绑定中。

`run_synthetic_attempt()` 仍仅接收显式 `execution_mode=synthetic_test_only` 的包。真实协议入口 `run_authorized_attempt()` 要求 signed authorization、`executable=true`、冻结的一次性额度、精确 real worker 身份和授权绑定的 `argv_template`；在工作起点后先以授权文件 SHA 创建不可重放 ledger directory，再创建 attempt claim。Ledger 采用 O_EXCL 事件文件：`consumed.json`、`launch_started.json`、`candidate_validated.json`、`return_candidate.json`。worker 启动命令只能由已绑定模板及固定 launch-spec 路径生成，不接受调用方临时命令；启动前后实现闭包摘要相等才允许候选进入返回见证。失败后消费 marker 保留，不重置额度或自动重试。

真实协议入口已有 temp-tree fake worker 贯通。独立 CLI `scripts/run_w1_d6_shared_gradient_worker.py` 在数值模块导入前验证 bounded/no-follow launch、授权绑定、一次性消费 receipt 和 launch event，再进入生产数值核心。它只消费 supervisor 固定 argv/environment，cwd 固定为 attempt root；`attempt_028` 因 `authorized=false` 不会启动它。supervisor 核对矩阵、runtime、来源记录、wait、回收与返回见证；dispatch 后异常以 outcome unknown 收口，失败不释放额度、不重试。**边界：**真实 worker 已启动一次但在数值模块导入和 payload 读取前失败；真实 production runtime、CUDA 数值和资产 payload 未测。旧 D6/R6 授权不适用于此流程。

## 保留尝试

`real_admission/` 根目录下早期非 attempt 文件及 `attempt_001/` 至 `attempt_029/` 的前序 packet 均保留。`attempt_027/` 及其授权记录保留为已消费的 dispatch 失败证据；`attempt_028/` 是加入失败收据前的中间 executable packet；`attempt_029/` 是 locale 修复、失败留痕修复及独立复核后的当前 executable packet，仍 `authorized=false`，未获新授权不能执行。一次失败授权不重用、不自动续签。
