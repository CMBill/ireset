# W1 固定六窗共享梯度真实准入与监督材料

**2026-11-21 · metadata-only 预准入材料生成；未读取 checkpoint/批次/资产 payload；无真实执行授权。**

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

新增 `scripts/supervise_w1_d6_shared_gradient.py` 是明确拒绝的外层占位入口：当前无真实 worker，不打开授权路径、不创建运行根、不 spawn 子进程，退出码 3。真实入口要启用，需先实现并独立复核 numerical worker 与标准库 supervisor：启动前 t0、一次性 claim、完整实现闭包前后哈希、分阶段调用/运行时/同步/deadline 守卫、硬截止进程组终止和 wait/cleanup 收据、最后同步回读后的外部返回见证。不能把本稿的提议矩阵或旧 D6/R6 授权当作这些证明。

验证：新增 admission/supervisor 测试 3 passed，覆盖元数据读取范围、固定外部 SHA、R6/D6/I/C 交叉绑定、bool 冒充 attempt 计数拒绝、unsigned draft 与 supervisor 不读授权即拒绝。端到端运行只生成上述 metadata packet；checkpoint/archive payload、NPZ member、dataset、robot asset、模型、FK、autograd、更新和真实 supervisor wait 均 0。**当前交付完成的是预审材料骨架，不是可签署的真实准入包；不得据此授权运行。**

## 保留尝试

`real_admission/` 根目录下早期非 attempt 文件及 `attempt_001/` 至 `attempt_004/` 的前序 packet 均保留。`attempt_003/` 首次加入固定 SHA pin，`attempt_004/` 增加 D6 index 与裁决的交叉绑定；`attempt_005/` 绑定这一最终 24 文件实现/方案闭包，是当前 metadata-only packet，但仍明确不能授权执行。
