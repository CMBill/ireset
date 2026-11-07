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

## 6. 修订记录（2026-11-07 独立审阅后）

独立审阅指出 N1 首轮判据不足以支持检测能力主张（P2-5）且预算封口缺失（P1-3）。全部修复后按修正协议重跑，权威改为 `attempt_002`（attempt_001 保留为被取代证据）。

- **C1**：由"同函数同输入双调用"改为 **audited 公式 vs 独立手写 float32 节点图**（`_independent_fp32_adamw`，按方案节点表独立实现）逐位对照；审计侧记录的中点 fallback 节点显式排除并逐用例登记（本用例集 8 profile × 2 步全部 0 豁免，即全部节点均参与逐位对照且一致）。
- **C5**：通过判据收紧——三个零分歧构造用例（m_none 零出现分歧、v 侧零分歧、次正规边界）**必须**复现跨路径门失败，且全部落在保存式区间传播内；实测 3/3 门失败、7/7 区间内。
- **C6**：8 类显著公式偏差全部改为喂给 **N0 坐标检测器 `compute_row_flags`**（与真实证据同一检测器）：断言 SOURCE_ANOMALY 置位、（区间越界或存在性分歧或非有限）触发、且行分类落入失败/gap 语义——8/8 检出，其中 7 例为分类级 gap、1 例（denom_without_eps 产生 ±inf）为非有限来源异常。
- **预算**：t0 前移至重模块导入前（attempt_002 实测含导入 3.95s）；summary 写盘后最终墙钟封口，越硬截止重写为 `budget_exceeded_after_seal`。
- **attempt 独占创建**；测试 8 项反例全过（新增 C5 门失败下界与 C1 独立对照断言）。

**go/no-go 维持 `go_for_conditional_fit_diagnostics_review`**，其余结论与 §3–§5 边界不变；计数不变（47 用例、fp32 求值 80、float64 参考 16、区间传播 16、状态推进 9，全部非零登记）。

## 7. 修订记录（2026-11-07 第二轮复阅后，attempt_003 权威）

第二轮独立复阅提出 2 P1+3 P2（其中涉及 N1 的为封口口径与 go 建议撤销），全部修复并以修正协议重跑：`attempt_001/002` 保留为被取代证据，权威为 **`attempt_003`**。

- **封口重写（P1×2，与 N0 共享实现）**：`write_outputs` 改用发布-核验循环（已用秒数 vs `wall_cap_s`；通过采样后零写盘；越界撤回）。attempt_003 实测 seal=sealed_within_budget（round 1，3.95s 含导入）。
- **go 建议随撤回同步撤销（P2）**：`N1_WITHDRAW_ON_EXCEED` 在封口撤回时把 `go_no_go` 一并改为 `no_go_budget_seal_exceeded`（原来只撤 all_cases_passed/label_verdict）；CLI 改为以盘上最终 summary 的 `seal.status` 与 `go_no_go` 决定退出码（0=sealed 且 go_*；3=未封口/预算越界；1=合成失败），不再使用写盘前内存值。
- **测试**：N1 套件 11 passed（新增 go 撤回反例与两阶段正例；go 撤回反例同时钉死大 t0 下 901s 必触发 900s cap 的正确口径）。
- 数值结论不变：六类 47 用例全过（计数 80/16/16/9），go=no_go 维持 `go_for_conditional_fit_diagnostics_review`；§3 有限收口说明与 §5 边界全部保持。

## 8. 修订记录（2026-11-07 第三轮复阅后，attempt_004 权威）

第三轮独立复阅提出 2 P1+1 P2（涉及 N1 的为登记成本覆盖与终态时序，与 N0 共享实现），全部修复并以修正协议重跑：权威为 **`attempt_004`**。

- **登记成本覆盖封口与返回（P1×2，与 N0 共享）**：登记成本 = 写前采样 + 尾部预留（固定返回尾部 3.0s + 2×已观测写盘延迟，下限 1.0s），与声明同盘；写后核验覆盖，未覆盖即增大预留重写；终态 `sealed` 仅在非终态 `sealing` 通过双核验后发布；CLI 打印后经 `verify_return_coverage` 独立核验实际返回时间，未覆盖即盘上撤回完成声明**与 go 建议**并 fail-closed 退出 3。attempt_004 实测：写盘前采样 3.87s、预留 4.0s、登记成本 **7.87s**，返回覆盖成立（exit 0）。
- **registration 绑定严格性（P2，N0 侧闭链）**：逐阶段硬性校验（存在/64 位小写十六进制/三值相等），缺失、畸形、不一致均硬失败。
- **测试**：N1 套件 11 passed（两阶段正例升级为同时断言盘上登记成本 54s 覆盖写后样本 40s）。
- 数值结论不变：六类 47 用例全过（计数 80/16/16/9），go=no_go 维持 `go_for_conditional_fit_diagnostics_review`；§3 有限收口说明与 §5 边界全部保持。

## 9. 修订记录（2026-11-07 第四轮复阅后，attempt_005 权威）

第四轮独立复阅提出 2 项 P1（返回核验时序与撤回成本覆盖，与 N0 共享实现），全部修复并以修正协议重跑：权威为 **`attempt_005`**。

- **返回核验时序与绑定证明（P1）**：核验先读盘解析再采样；CLI 输出 flush 后核验；`return_observation.json` 绑定证明（含 summary SHA 与 `checked_after_output_flush`），证明写盘后再核验；CLI 所有退出路径执行核验，未覆盖即 fail-closed（撤回时 go_no_go 一并撤销）。attempt_005 实测：covered=true、elapsed 3.886s ≤ 登记 7.885s、SHA 绑定一致。
- **撤回成本覆盖（P1）**：与 N0 共享——封口循环计费检查前移（exceeded 写盘亦须自身计费覆盖）、撤回路径覆盖成本循环、不能证明时 `cost_coverage_unproven=true` 保持不可消费。
- **测试**：N1 套件 11 passed（CLI 测试新增观察文件断言：covered、SHA 绑定、elapsed ≤ 登记成本）。
- 数值结论不变：六类 47 用例全过（80/16/16/9），go=no_go 维持 `go_for_conditional_fit_diagnostics_review`；§3 有限收口说明与 §5 边界全部保持。
