# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-22 · 输入域、失败计数与固定输出工件契约完成修复；独立 worker/supervisor 复核完成；executable packet `attempt_026` 已生成；未读取 checkpoint/批次/资产 payload；无真实执行授权。**

## 结果

生成器读取四份已封口元数据：R6 admission、D6 独立裁决、D6 artifacts index、I/C `attempt_006` CUDA acceptance；另读取并 SHA-256 绑定 38 个当前实现/方案文件。它不打开被索引指向的 `initial.pt`、`probe_inputs.npz`、root-stats NPZ、数据集或生产资产，也不重新哈希历史绑定的 41 个资产。R6 的历史绑定仅作为已封口元数据引用；本轮未重新验证其 payload 当前仍与登记相符。

当前材料目录：`derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/attempt_026/`

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `admission.json` | 待生成 | `executable=true`、`authorized=false`、独立复核绑定 |
| `authorization_draft.json` | 待生成 | `authorized=false`、`template_only=true`、六项 checklist 为 null |

Admission canonical metadata SHA 将在 `attempt_026` 文件生成后登记。它绑定 D6 裁决 SHA `1de06dae…`、R6 admission SHA `9e54107c…`、D6 artifact index SHA `c5cca9bc…`、I/C acceptance SHA `f05fc505…`、六窗登记路径、六份 D6 工件记录、41 项资产记录、39 文件实现闭包、独立复核收据、固定账本根 `derived/audit/w1_d6_shared_gradient_step0_v1/authorization_consumption_v1` 和 package-pinned worker argv/environment。真实 CUDA/I/O 运行仍未授权、未执行。候选真实运行根未创建。

材料重申单次运行 300s 硬上限（240s 工作+60s 收口）、一次 attempt、参数更新 0。冻结 call matrix 分列来源读取、NPZ 成员、生产 scene/setup、只读 Trainer 构造、数值调用及禁止更新；worker I/O matrix 分列 48 个来源文件、60+2 个 NPZ 成员访问、20 个 NPY 与 8 个 JSON 工件写入、独立复核读回 27 次/哈希 20 次；supervisor matrix 登记消费事件、启动、实际 wait、日志、cleanup 检查上限与返回回读。运行时 worker meter 在生产 setup 调用点逐项计数，数值核心的 hook/VJP meter 复核前向与梯度调用，NPZ decoder、artifact store、独立读盘复核记录实际 I/O；supervisor 对 worker result 中的矩阵、运行时身份和来源 SHA 逐项比对 admission 包。真实值仍未运行；当前证明只来自临时 worker。

## Fail-Closed 边界

`scripts/supervise_w1_d6_shared_gradient.py` 仍是明确拒绝的生产 CLI：不打开授权路径、不创建真实运行根、不 spawn 真实 worker，退出码 3。其合成可调用进程原语已实现并复核：独立 session/process group、stdout/stderr 并发排空及定额截断、真实 `Popen.wait()`、超时后的 TERM/KILL 和进程组回收检查、日志文件身份/哈希、pending return receipt 的 fsync 与同步回读。最后单调时钟样本只保存在私有内存 `ExternalReturnWitness` 中；盘上记录标记 `pending_actual_readback` 并保存回读前样本，避免把采样后的写盘时间伪称已覆盖。缺少该私有返回对象、工件漂移、迟返、超时、溢出或未回收均不能得到 `covered`。这些测试不构成真实 worker/授权消费证明。

验证：独立复核脚本执行 compileall、worker/supervisor signed-path 门和相关 I/C 定向回归；测试数、review receipt 和 packet SHA 在最终生成后登记。复核包含 supervisor dispatch 后失败时不谎报“未 spawn”，并确认 consumed ledger/failure event 保留。worker CLI 坏 launch 与 unsigned 快照在导入数值核心前拒绝；临时 fake-worker 正常启动/回收/收口，矩阵漂移拒绝候选；输入重建域反例、worker_failure 留痕、固定工件、账本和返回见证均通过。`attempt_026` 将为 `executable=true`、`authorized=false`，不会因生成而启动真实 worker。真实 payload、资产、CUDA 数值、autograd、参数更新和真实运行根均为 0。

## Supervisor 合成复核（2026-11-21）

`supervise_process()` 的启动边界、实际 `wait()` 返回、cleanup 完成和最后同步回读分开记录。Worker 在新 session 中启动，因此 PID 与 process-group ID 相等并与 supervisor PID 不同。父进程分别排空两个 pipe，日志最多保留登记上限字节但继续读取以避免 pipe deadlock；超量会使见证不可消费。截止超时会向整个 group 发送 SIGTERM，宽限后 SIGKILL，并实际 wait 主进程、检查 group 与排空线程是否消失。若 cleanup 不能被证明，见证拒绝。

返回观察文件采用一次性创建、文件与目录 fsync、no-follow regular-file 回读和 inode/字节核验。第二次消费回读与 JSON/绑定核验完成后才取得 final monotonic sample；此后不再执行文件 I/O。落盘文件只能是 pending receipt，只有同进程私有见证允许 `verify_external_return()` 消费。覆盖条件同时要求 worker 在工作 deadline 前清理完成，最终 sample 在登记墙钟与 hard deadline 内，exit code 为 0、未超时、group 已回收且日志未溢出。受控启动向 `Popen` 传入 package 冻结的环境映射，不继承 supervisor 环境；解释器绝对路径和 worker script 必须都在授权 launcher 与实现闭包绑定中。

`run_synthetic_attempt()` 仍仅接收显式 `execution_mode=synthetic_test_only` 的包。真实协议入口 `run_authorized_attempt()` 要求 signed authorization、`executable=true`、冻结的一次性额度、精确 real worker 身份和授权绑定的 `argv_template`；在工作起点后先以授权文件 SHA 创建不可重放 ledger directory，再创建 attempt claim。Ledger 采用 O_EXCL 事件文件：`consumed.json`、`launch_started.json`、`candidate_validated.json`、`return_candidate.json`。worker 启动命令只能由已绑定模板及固定 launch-spec 路径生成，不接受调用方临时命令；启动前后实现闭包摘要相等才允许候选进入返回见证。失败后消费 marker 保留，不重置额度或自动重试。

真实协议入口已有 temp-tree fake worker 贯通。独立 CLI `scripts/run_w1_d6_shared_gradient_worker.py` 在数值模块导入前验证 bounded/no-follow launch、授权绑定、一次性消费 receipt 和 launch event，再进入生产数值核心。它只消费 supervisor 固定 argv/environment，cwd 固定为 attempt root；`attempt_026` 因 `authorized=false` 不会启动它。supervisor 核对矩阵、runtime、来源记录、wait、回收与返回见证，失败不释放额度、不重试。**边界：**真实 worker 尚未获授权执行；真实 production runtime、CUDA 数值和资产 payload 未测。旧 D6/R6 授权不适用于此流程。

## 保留尝试

`real_admission/` 根目录下早期非 attempt 文件及 `attempt_001/` 至 `attempt_026/` 的前序 packet 均保留。`attempt_021/` 保留输入域、失败计数、固定工件和矩阵修复；`attempt_026/` 为当前 executable packet，仍 `authorized=false`，未签署前不能执行。
