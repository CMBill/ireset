# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-21 · metadata-only 预准入材料生成；worker 与 supervisor 进程原语合成复核；未读取 checkpoint/批次/资产 payload；无真实执行授权。**

## 结果

生成器读取四份已封口元数据：R6 admission、D6 独立裁决、D6 artifacts index、I/C `attempt_006` CUDA acceptance；另读取并 SHA-256 绑定 24 个当前实现/方案文件。它不打开被索引指向的 `initial.pt`、`probe_inputs.npz`、root-stats NPZ、数据集或生产资产，也不重新哈希历史绑定的 41 个资产。R6 的历史绑定仅作为已封口元数据引用；本轮未重新验证其 payload 当前仍与登记相符。

权威材料目录：`derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/attempt_005/`

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `admission.json` | `9cf7d08a5f2ad0ebf65c3495e44cfc29d823daf0e54255a8d0ea092dcd77e1f1` | `metadata_only_materials_drafted_worker_missing`，`executable=false` |
| `authorization_draft.json` | `2eabebff698fc154ae14665b55fb48c749e6fc2d7fbd95dbde284b474e76bbe7` | `authorized=false`、`template_only=true`、六项 checklist 为 null |

Admission 内部 canonical metadata SHA 为 `094413d6e2e87a046cdb040c8b34e25041a4cbe613048ffb86ee795dbbdaa3e7`。它绑定 D6 裁决 SHA `1de06dae…`、R6 admission SHA `9e54107c…`、D6 artifact index SHA `c5cca9bc…`、I/C acceptance SHA `f05fc505…`、六窗登记路径、六份 D6 工件记录和 41 项资产记录。校验器另重算 D6 index 中的 R6 admission 引用、六槽顺序和 D6 launch 与裁决的交叉绑定。候选运行根为 `derived/experiments/omniretarget_first/w1_d6_shared_gradient_step0_v1/attempt_001`，目录未创建。

材料重申单次运行 300s 硬上限（240s 工作+60s 收口）、一次 attempt、参数更新 0。未来输入及 I/O 记录逐项列出 D6 六工件、两个 root-stats 成员和 41 个资产的预期路径/长度/SHA/访问数；完整真实矩阵仍标记 `proposed_only_not_runtime_verified`。I/C 的 32 个 CUDA 同步点只证明合成核心，不是对真实来源读取、scene 构造、worker 监督、实际 `wait()`、进程回收、外层返回观察及收口 I/O 的验收。

## Fail-Closed 边界

`scripts/supervise_w1_d6_shared_gradient.py` 仍是明确拒绝的生产 CLI：不打开授权路径、不创建真实运行根、不 spawn 真实 worker，退出码 3。其合成可调用进程原语已实现并复核：独立 session/process group、stdout/stderr 并发排空及定额截断、真实 `Popen.wait()`、超时后的 TERM/KILL 和进程组回收检查、日志文件身份/哈希、pending return receipt 的 fsync 与同步回读。最后单调时钟样本只保存在私有内存 `ExternalReturnWitness` 中；盘上记录标记 `pending_actual_readback` 并保存回读前样本，避免把采样后的写盘时间伪称已覆盖。缺少该私有返回对象、工件漂移、迟返、超时、溢出或未回收均不能得到 `covered`。这些测试不构成真实 worker/授权消费证明。

验证：本轮共享梯度 worker、adapter、admission 与 supervisor 定向套件 **129 passed**。新增 8 项 supervisor 进程测试覆盖正常 `wait()`/日志/返回见证、超时回收、输出上限、存活后代进程组清理、预算迟返拒绝、receipt fsync/readback 与漂移拒绝、非有限 deadline 在 spawn 前拒绝；生产 CLI 的 refusal 回归仍通过。`compileall` 与 `git diff --check` 通过。测试只启动使用 Python `-I -c` 的合成子进程及临时目录；checkpoint/archive payload、NPZ member、dataset、robot asset、真实模型/FK/autograd/更新、真实 worker 和真实运行根均为 0。当前 `attempt_005` packet 是较早生成的历史 metadata-only 文件，未按本轮新增闭包重发；`executable=false`、`authorized=false`，不得据此授权运行。

## Supervisor 合成复核（2026-11-21）

`supervise_process()` 的启动边界、实际 `wait()` 返回、cleanup 完成和最后同步回读分开记录。Worker 在新 session 中启动，因此 PID 与 process-group ID 相等并与 supervisor PID 不同。父进程分别排空两个 pipe，日志最多保留登记上限字节但继续读取以避免 pipe deadlock；超量会使见证不可消费。截止超时会向整个 group 发送 SIGTERM，宽限后 SIGKILL，并实际 wait 主进程、检查 group 与排空线程是否消失。若 cleanup 不能被证明，见证拒绝。

返回观察文件采用一次性创建、文件与目录 fsync、no-follow regular-file 回读和 inode/字节核验。回读后取得 final monotonic sample，不再写文件；因此落盘文件只能是 pending receipt，只有同进程返回的私有见证记录 final sample 并允许 `verify_external_return()` 消费。覆盖条件同时要求 worker 在工作 deadline 前清理完成，final sample 在登记墙钟与 hard deadline 内，exit code 为 0、未超时、group 已回收且日志未溢出。

该单元尚未接到真实授权/claim/implementation-closure/worker launcher，也没有父级实际返回后的完整消费链。当前 CLI 继续拒绝运行；旧 D6/R6 授权不适用于此流程。

## 保留尝试

`real_admission/` 根目录下早期非 attempt 文件及 `attempt_001/` 至 `attempt_004/` 的前序 packet 均保留。`attempt_003/` 首次加入固定 SHA pin，`attempt_004/` 增加 D6 index 与裁决的交叉绑定；`attempt_005/` 绑定这一最终 24 文件实现/方案闭包，是当前 metadata-only packet，但仍明确不能授权执行。
