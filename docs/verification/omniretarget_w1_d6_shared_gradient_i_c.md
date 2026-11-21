# W1 固定六窗共享梯度诊断 I/C 验证记录

**2026-11-21 · 纯合成验收通过；真实来源/检查点未读，真实梯度诊断未授权、未运行。**

## 结果

按[预注册草案 v0.3](../plans/omniretarget_w1_d6_shared_gradient_prereg_draft.md)完成合成 I/C 实现与复核。权威运行是 `derived/audit/w1_d6_shared_gradient_step0_v1/synthetic/attempt_006/`，`cuda_acceptance.json` SHA-256 为 `f05fc5054f27158b5a753ae751c982ea6758c737bc092e3d5aa00ae61ba32d73`，`passed=true`。

| 验收项 | 结果 |
| --- | --- |
| CPU 手算/反例控制 | 6/6 通过；包含等权投影、抵消临界值、方向反转、范数零门及排序敏感性 |
| D6 架构 CUDA 合成矩阵 | 18/18 通过；9 组输入 × 模型初始化 seed `20261008`、`20261122` |
| 参数加性门 | 所有模型/conditioner/joint 及逐参数组通过；最大 `rho` 分别为 `2.24779090e-7`、`1.20235993e-7`、`2.24766225e-7`，均低于合成准入线 `2.5e-4` |
| `h_batch` 槽位门 | 18/18 案例通过；最大误差为固定 `2e-4` 界的 `0.002612` |
| CPU 独立读盘复核 | 18/18 通过；每例重读有限 float32 数组、重算损失/梯度组/投影与裁决，并核验原始具名梯度顺序、返回收据、产物 SHA 和调用矩阵 |
| 单例资源峰值 | 工件峰值 `121,088,256` bytes（低于 128 MiB）；CUDA allocated 峰值 `8,915,526,656` bytes |
| 真实操作 | checkpoint/archive/生产资产读取 0；参数更新 0；优化器、数据集迭代、contact 分支均 0 |

运行时为 RTX 3090、PyTorch `2.14.0+cu126`、CUDA runtime `12.6`、NumPy `2.5.3`、float32、`cuda:0`、1 CPU thread、matmul precision `highest`；TF32、autocast 及 fp16/bf16 reduced-precision reduction 均关闭。每例实测 32 个 CUDA 同步点、7 次配对 VJP、1 次 `Trainer.predict`、1 次生产 `LossAssembler`、2 次 decode 和 2 次合成 FK。合计 126 次 VJP；`Trainer` 构造 18 次，但无 optimizer、loader iterator、dataset read 或 update。

每例精确工件矩阵为 20 个 NPY 写入、5 个 JSON 写入、24 个独立读盘、23 次文件 SHA-256 读回核验；每个用例目录一次写入 25 个文件。17 例在独立临时目录完成读回并清理，保留 seed `20261008`/输入组 0 的完整证据；最终 `cuda_acceptance.json` 单独写入一次。实现闭包 16 个文件在运行前后各读/哈希一次，身份摘要 `f65e074e2eea825933e0b6250674e368ae8bd35afdae0f2ea02e3b175e8feda3`，两端一致。验收 JSON 为每例记录独立复核 SHA、逐例门值和调用计数。

## 实现与测试

- [合成共享梯度模块](../../src/ireset/eval/w1_d6_shared_gradient.py)：使用生产 `Trainer.predict`、`DeterministicG1`、`ConditionEncoder`、`LossAssembler`、`RegisteredWindowDataset` 和 `default_collate`；FK 仅为合成可微替身。参数名册为 130 tensors / 3,739,210 coordinates。模块 SHA-256 `e0cd2bd5e38a0423da7874ed3c152f52e51c015d76cf0ad8113c383bf2800282`。
- [合成验收 CLI](../../scripts/verify_w1_d6_shared_gradient.py)：CPU controls、18 例 CUDA/float32 controls；`--real` 固定 fail-closed，不解析或读取给定路径/授权文件。
- [反例与接口测试](../../tests/test_w1_d6_shared_gradient.py)：18 passed。覆盖计算判据、损失元素 float64 复算、micro 排序、VJP 收据与梯度名册、截止拒绝、工件空间拒绝、真实入口拒绝。
- `Trainer` 仅加 DataLoader 构造完成观察点；默认训练数值路径不变。配套回归 `tests/test_w1_real_training_update_provider.py`：45 passed；`tests/test_w1_de_numerics.py` + `tests/test_omniretarget_w1_de.py`：73 passed；新增测试 18 passed。未运行全项目套件。

纯合成输出的投影裁决仅验证归约和方向判据，不说明真实训练窗是否相消，不证明中间 FK/碰撞、泛化、可行交互或生产资格。

## 被取代尝试

| 尝试 | 状态 |
| --- | --- |
| `attempt_001` | 初始模型 seed 的 `h_batch` 槽位门失败；仅留部分数组，无终态，不消费为验收 |
| `attempt_002` | 18 例数值门通过，但同步点/逐文件读回矩阵尚未准确记录；保留为被取代证据 |
| `attempt_003` | 同步/读回计数已修；但未将实现闭包双端身份写入终局材料，保留为被取代证据 |
| `attempt_004` | 闭包身份及 I/O 门已加入；缺少逐例 `h` 界分数和完整运行时旗标，保留为被取代证据 |
| `attempt_005` | 数值门通过，但未绑定逐例独立复核摘要；保留为被取代证据 |
| `attempt_006` | 当前权威；来源代码闭包双端稳定、运行时/同步/I/O 矩阵及逐例复核摘要完整绑定，18/18 通过 |

真实阶段仍须新的冻结材料、完整真实路径 I/O/runtime/supervisor 矩阵和明确一次性授权；不得重用既往 D6/R6 授权。旧 `attempt_001` 的失败不被删除或改判。
