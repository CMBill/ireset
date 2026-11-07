# W1 N1 最小合成验证与有限数值收口说明

日期：2026-11-07。对应方案：[数值缺口有限收口与受控拟合诊断方案 v0.4](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md) §5。
**状态：N1 已按冻结上限执行一次并完整收口；I/R/D1/D6/F 均未实施。** 纯 CPU 合成算术：无真实网络/原生优化器/checkpoint/数据集；不校准、不宣称 CPU 等价于生产 CUDA/foreach；不改学习率、AdamW epsilon 或阈值；不把参数/梯度升为 float64 充当生产修复。

## 1. 执行事实

- 实现：[src/ireset/eval/w1_n1_synthetic_numerics.py](../../src/ireset/eval/w1_n1_synthetic_numerics.py)（复用已审计的 `adamw_effective_fp32`/`adamw_math_reference`/`propagate_intervals`/`_delta_observability`/`_zero_state_problems`/`_cross_path_flags`/`_inside`，未复制或改写其数值路径），入口 [scripts/verify_w1_n1_synthetic_numerics.py](../../scripts/verify_w1_n1_synthetic_numerics.py)，反例测试 `tests/test_w1_n1_synthetic_numerics.py` **8 passed**（N0 套件 9 passed 另计）。
- 权威运行 `derived/audit/w1_numerics_n0_n1/n1/attempt_001/`：**47 用例全过**（≤48 上限；六类 8/8/8/8/7/8），实测 **0.026s**（预算 900s/工作截止 840s 内，未触发）。实际登记计数（全部非零）：fp32 公式求值 80（≤384）、float64 数学参考求值 16、区间传播 16、合成状态推进 9、用例 47。每例向量≤16 坐标、≤2 步、2 条比较路径。
- 机读交付：`n1_cases.json`（逐用例 expected/observed/passed）与 `n1_summary.json`（分类计数、闭包字段、go/no-go、预算事实）。

## 2. 六类合成验证结果

| 类 | 用例 | 验证内容与结果 |
| --- | ---: | --- |
| same_input_formula_bitwise | 8/8 | 8 组对抗输入轮廓（零/次正规/2的幂/中点舍入/深相消/随机）×2 步：同输入双路径全部节点（theta/m/v/s/b/denom/ratio/delta_ref64）**逐位相同**；并记录相对 float64 数学参考的偏差（仅上下文，不作门） |
| nonzero_write_erasure | 8/8 | 更新量低于写入分辨率时：theta 写入被擦除或"变化了但低于可辨识阈值"，`_delta_observability` 判 obs=False→**precision 语义，不是公式错误**；可辨识增量正常通过；零参考控制通过 |
| one_sided_observable | 8/8 | 同一更新量在不同 theta 量级下单侧可辨识（大 theta 侧写入被擦除）：`_cross_path_flags` 判 `delta_one_sided_observable=True`→**sensitivity 语义**，不误判为公式发散；双侧可辨识/双侧擦除控制正确 |
| state_zero_sign_problems | 8/8 | `nonzero_erased`/`zero_invented`/符号不一致/v<0 各自精确产出预期问题字符串→**direct failure 语义**；深噪声带（±1e-12）与全零控制不误报 |
| cancellation_within_intervals | 7/7 | 1-ULP 梯度差（含 m_none 零出现分歧、次正规边界、三档相消带、两步递推）：跨路径门失败的用例**全部落在保存式区间传播 K_\* 内**→`cross_path_sensitivity_gap` 语义复现；输入差均在冻结梯度门预算内 |
| interval_violation_deviation | 8/8 | 8 种显著公式偏差（h 翻号/去 eps/去 bc2/g 不平方/去 sqrt/lr 翻倍/分母加地板/动量直通）：全部与冻结公式逐位不同**且**越出 K_theta 或 K_delta 区间→证明 N0 的"实现问题候选"检测在偏差真实存在时会触发（N0 零命中的区分力得到正面验证） |

## 3. 有限数值收口说明（按方案 §5 格式）

1. **已验证的同输入公式**：有效 float32 AdamW 节点图在 8 组对抗输入（覆盖零、次正规、2 的幂、中点舍入、深相消带）×2 步上同输入逐位确定；写界=更新后参考 theta 半ULP、obs/ok 判据、区间传播与 N0 重放一致。
2. **未通过的严格跨路径门**：N0 的 189 行 `cross_path_sensitivity_gap` 不因本验证改判——它们是原始相对门在保存传播区间内的相消敏感，本验证以合成用例复现了同一语义（门失败且全部落在区间内）。
3. **未覆盖范围**：长期训练动态、动作/交互质量、CUDA/foreach 生产路径等价性、完整训练更新链——本验证均为合成小向量，不构成上述任何证明。
4. **实现问题候选**：N0 全部 528 行 0 命中，且 C6 证明 8 类显著偏差都会被"逐位公式门+区间"检出——支持"当前保存证据内无实现问题候选"的判断，不等于证明实现无错。
5. **不做的事**：不调阈值/epsilon/lr 追求旧 gap 清零；不宣称 539 项 gap 消除；不恢复 T6 一次性授权；不启动真实拟合。

## 4. go/no-go 建议

`go_for_conditional_fit_diagnostics_review`：在**明确承认 partial 与旧 gap 未消除**的前提下，I（共享 Trainer 诊断实现）具备进入独立审阅的条件；真实拟合（R→D1→D6→F）仍须按方案 §8.3 另立冻结材料与新明确授权，不因本文存在而获准。

## 5. 边界

- T6 保持：`audit_incomplete`、`covered_contract_passed=false`、0 failures/539 gaps、cohort 34 次更新/1347.195507s、一次性授权已消费。
- 全局状态保持：`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。
- N0/N1 均为保存证据/合成算术层面的收口分析；旧报告、旧契约、旧工件未改写。
