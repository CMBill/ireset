# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-21 · metadata-only 预准入材料生成；worker 与 supervisor 进程原语合成复核；未读取 checkpoint/批次/资产 payload；无真实执行授权。**

## 结果

生成器读取四份已封口元数据：R6 admission、D6 独立裁决、D6 artifacts index、I/C `attempt_006` CUDA acceptance；另读取并 SHA-256 绑定 24 个当前实现/方案文件。它不打开被索引指向的 `initial.pt`、`probe_inputs.npz`、root-stats NPZ、数据集或生产资产，也不重新哈希历史绑定的 41 个资产。R6 的历史绑定仅作为已封口元数据引用；本轮未重新验证其 payload 当前仍与登记相符。

当前 metadata-only 材料目录：`derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/attempt_010/`

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `admission.json` | `6f564d056d558e461c795557c36431a151382e3d886db54a20d5e6e0c7938afa` | `metadata_only_real_execution_pending_authorized_worker_cli_review`，`executable=false` |
| `authorization_draft.json` | `4c8904a5a4b87a30a0e26b1c490d1e77ea3e378d06c13418fa712373e34eaaa0` | `authorized=false`、`template_only=true`、六项 checklist 为 null |

Admission canonical metadata SHA 为 `78bf42764b6fdeddb9fea81831f017d07074cfc013ee18545d5aa903b62cd4d7`。它绑定 D6 裁决 SHA `1de06dae…`、R6 admission SHA `9e54107c…`、D6 artifact index SHA `c5cca9bc…`、I/C acceptance SHA `f05fc505…`、六窗登记路径、六份 D6 工件记录、41 项资产记录、实现闭包、固定账本根 `derived/audit/w1_d6_shared_gradient_step0_v1/authorization_consumption_v1` 和拟议 worker argv/environment 模板。当前 worker argv 状态为 `proposed_not_executable`，目标脚本尚未进入可执行准入闭包。候选真实运行根仍未创建。

材料重申单次运行 300s 硬上限（240s 工作+60s 收口）、一次 attempt、参数更新 0。未来输入及 I/O 记录逐项列出 D6 六工件、两个 root-stats 成员和 41 个资产的预期路径/长度/SHA/访问数；完整真实矩阵仍标记 `proposed_only_not_runtime_verified`。I/C 的 32 个 CUDA 同步点只证明合成核心，不是对真实来源读取、scene 构造、worker 监督、实际 `wait()`、进程回收、外层返回观察及收口 I/O 的验收。

## Fail-Closed 边界

`scripts/supervise_w1_d6_shared_gradient.py` 仍是明确拒绝的生产 CLI：不打开授权路径、不创建真实运行根、不 spawn 真实 worker，退出码 3。其合成可调用进程原语已实现并复核：独立 session/process group、stdout/stderr 并发排空及定额截断、真实 `Popen.wait()`、超时后的 TERM/KILL 和进程组回收检查、日志文件身份/哈希、pending return receipt 的 fsync 与同步回读。最后单调时钟样本只保存在私有内存 `ExternalReturnWitness` 中；盘上记录标记 `pending_actual_readback` 并保存回读前样本，避免把采样后的写盘时间伪称已覆盖。缺少该私有返回对象、工件漂移、迟返、超时、溢出或未回收均不能得到 `covered`。这些测试不构成真实 worker/授权消费证明。

验证：worker、adapter、admission 与 supervisor 定向套件 **139 passed**。本轮新增真实协议形态的临时项目树 fake-worker 贯通及账本竞争测试，覆盖正常启动/回收/收口、unsigned 拒绝、一次性消费 marker、跨 run-root 重放拒绝、同一授权并发仅一方消费、授权绑定的 argv/environment 篡改拒绝、worker 失败后授权仍保持 consumed、账本根漂移拒绝，以及实现闭包变化时不发布成功见证。`attempt_010` 与当前 `build_materials()` 输出逐字段一致；worker 启动状态仍 `proposed_not_executable`。正式生产 CLI 实测仍 exit 3，`authorization_path_opened=false`、`worker_spawned=false`、payload reads 和 updates 均为 0。只使用临时 worker 和临时源码；真实 checkpoint/archive payload、NPZ member、dataset、robot asset、数值模型、FK、autograd、参数更新、真实 worker 与真实运行根均为 0。

## Supervisor 合成复核（2026-11-21）

`supervise_process()` 的启动边界、实际 `wait()` 返回、cleanup 完成和最后同步回读分开记录。Worker 在新 session 中启动，因此 PID 与 process-group ID 相等并与 supervisor PID 不同。父进程分别排空两个 pipe，日志最多保留登记上限字节但继续读取以避免 pipe deadlock；超量会使见证不可消费。截止超时会向整个 group 发送 SIGTERM，宽限后 SIGKILL，并实际 wait 主进程、检查 group 与排空线程是否消失。若 cleanup 不能被证明，见证拒绝。

返回观察文件采用一次性创建、文件与目录 fsync、no-follow regular-file 回读和 inode/字节核验。回读后取得 final monotonic sample，不再写文件；因此落盘文件只能是 pending receipt，只有同进程返回的私有见证记录 final sample 并允许 `verify_external_return()` 消费。覆盖条件同时要求 worker 在工作 deadline 前清理完成，final sample 在登记墙钟与 hard deadline 内，exit code 为 0、未超时、group 已回收且日志未溢出。受控启动向 `Popen` 传入 package 冻结的环境映射，不继承 supervisor 环境；解释器绝对路径和 worker script 必须都在授权 launcher 与实现闭包绑定中。

`run_synthetic_attempt()` 仍仅接收显式 `execution_mode=synthetic_test_only` 的包。真实协议入口 `run_authorized_attempt()` 要求 signed authorization、`executable=true`、冻结的一次性额度、精确 real worker 身份和授权绑定的 `argv_template`；在工作起点后先以授权文件 SHA 创建不可重放 ledger directory，再创建 attempt claim。Ledger 采用 O_EXCL 事件文件：`consumed.json`、`launch_started.json`、`candidate_validated.json`、`return_candidate.json`。worker 启动命令只能由已绑定模板及固定 launch-spec 路径生成，不接受调用方临时命令；启动前后实现闭包摘要相等才允许候选进入返回见证。失败后消费 marker 保留，不重置额度或自动重试。

真实协议入口已经有 temp-tree fake worker 贯通。一次性账本根由 admission packet 固定为 `authorization_consumption_ledger`，每个授权 SHA 以独占目录消费，目录内事件只能 O_EXCL 新增；消费发生在 attempt 创建与 spawn 前，失败不释放。启动器必须由 package 的 `worker_launch.argv_template`、绝对 interpreter、实现闭包内的 `script_rel` 和显式环境映射共同冻结；生产环境不继承父进程环境，argv 不经 shell。真实包当前仍 `executable=false` 且 worker 状态为 `proposed_not_executable`，因此 `run_authorized_attempt()` 对当前正式材料 fail-closed。正式 `main()` 仍在读取授权路径前拒绝。尚未实现/验证生产 worker 独立 CLI/permit 初始化、生产 runtime/call-I/O 矩阵及真实资产闭包。旧 D6/R6 授权不适用于此流程。

## 保留尝试

`real_admission/` 根目录下早期非 attempt 文件及 `attempt_001/` 至 `attempt_009/` 的前序 packet 均保留。`attempt_003/` 首次加入固定 SHA pin，`attempt_004/` 增加 D6 index 与裁决的交叉绑定，后续 packet 逐次绑定实现闭包。当前 `attempt_010/` 是包含一次性授权账本根与 package-pinned argv/environment proposal 的 metadata-only packet；仍 `authorized=false/executable=false`，不能签署或执行。
