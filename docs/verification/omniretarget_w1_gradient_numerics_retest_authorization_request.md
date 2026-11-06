# W1 数值复验 T6 十项明确授权请求

日期：2026-11-06。状态：**待用户逐项确认；当前 `authorization_request.json` 为 `awaiting_explicit_user_confirmation`，不是正式授权。**

本请求对应[复验方案](omniretarget_w1_gradient_numerics_retest_plan.md) §9。T5 材料已复阅通过；正式授权记录尚未生成，T6 未启动。

## 1. 绑定材料

| 项目 | 值 |
| --- | --- |
| 实现身份（32 文件） | `c4d4a88926b985ee365732da0d940e28985f86c64d4f828e0bb6c5dca4f01b3b` |
| 复验方案 SHA | `8bb4a79364183231c55dc31c6582ab2622656e9696c0c0724c3e62ce77ebd3e9` |
| 预注册草案 SHA | `e2ba8ef9edb7381dbe410f7e0e290a6fbd702323a4416eeacfee4d2d14796242` |
| 校准 contract | `2a86e06d70e12e44159bef4e4a22e5032b5e2da92fb9dbbe94c547f2c36662df` |
| calibration evidence | `962b625c18080d6c1e08b8769d2a6284161cd86c3d08c0de057e232a6364ccda` |
| calibration canonical | `a9e7b9228189ecb2f1d06ed400e3f4f2aa68b095877cfa5256759f3713e40446` |
| fp32 canonical | `94290e009c5238416a4f24404fefd2b18c1d5efa9f1ecdeef004935de0bd90ce` |
| Gate B v2 | `8157ae0be4bbfed2d06490f4f2e3a119908c49823863105c331c276f65345635` |
| retest acceptance | `af453f7adf4179e77131782a932ccb04bb2ee006d53b373686dbcc8b21d00b36` |
| real admission | `6d986cb392b7bcf06398f6e4762046d6b268938dd27f038edcb8a31c04b0058f` |
| retest admission | `49ae9011a63bc1efd923945c999a96bd1473370073866efeb3093d2e397cad21` |
| history snapshot | `710a85e484411004992795d6bfbd95846ef336b67f8125f971ac1ef1671003b6` |
| binding spec | `357daf3d796276117397d1125e4b89e318cee28ccc9bfcb538dbb608c23e5131` |
| budget revision canonical | `2ceff41a7c72558a1b1c5b1b1914f1108de1fb8ee8ce7c4c2d1dbc767f8246c8` |
| protected manifest | `c6c07c7c747ef4e43de9c79abe9a10fd4b9347b47cc4b0f7bf5d220e617ba22f` |
| closed history head | `af5c797126de82eb1d79b8c319e0f7a5a5cfcbd24b604fb6f4eb35ef36324896` |
| authorization draft | `64fef827f353a29e9d7f61e88e445122d3c6664a904224385be7ac6235e6521e` |
| M5 preflight | `6c27634e0d71f5e02564db4c910540906bcb721515641b348b49dfc5d9a93e43`（passed） |

机器可读请求：[authorization_request.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/gradient_numerics_retest/authorization_request/attempt_001/authorization_request.json)，SHA `4411bbb203d4c339a83e39da717f093ca4f3c0ef322d269abbb4f71cbd685183`。

## 2. 请求授权的十项

| checklist 键 | 本轮范围 |
| --- | --- |
| `new_question_cohort_and_parents` | 同一原根/父锚，承认完整 M6 与六行历史；追加 generation_003 / seq 7–10，不重置 cohort。 |
| `fresh_adamw_and_diagnostic_steps` | 固定 checkpoint 起点，本轮仅新增 16 次诊断更新，累计上限 34；旧 18/32 规则作为历史保留。 |
| `two_continuous_updates_and_common_prestate` | 固定两步/共同前态；u2 保留 u1 的真实 AdamW 状态，不删参数、不清空动量。 |
| `real_network_objective_parameter_backward_fk_clip_adamw` | 本轮 P/I/F 的真实网络、目标、参数 VJP、FK、clip、更新与固定 focus 保存范围。 |
| `call_matrix_and_unsettled_unknown` | 24 类调用矩阵及各阶段 nominal；完整结算前不填名义总数，F 未结清时 null/下界 26。 |
| `budgets_attempts_and_resources` | 累计仍为 3240 秒；R0/P 重分配为 600/540 秒，I/F 900/1200 秒；累计 attempt 3/3/2/2，本轮各一次及资源门。 |
| `synthetic_acceptance_fp32_zeros_propagation` | 最终校准/运行时/四元数源码、Gate B v2 与 retest acceptance；原零规则、可辨识性与传播界。 |
| `zero_fd_ddim_selection_checkpoint_contact` | 七项整数零声明；额外真实探测/分项 VJP/FD/选模/写 checkpoint 不在授权内。 |
| `publishing_three_way_and_withdrawal` | failure/gap/证据完整性分别发布；提交后 revoke、实际返回覆盖、unknown 与不完整结果如实保留。 |
| `partial_coverage_not_ready_no_auto` | partial/null/false/not_ready/不接入保持；一次入口、不自动重试、不自动生产接入。 |

## 3. 固定执行范围

- generation_003：`R0 attempt_003`、`P attempt_003`、`I attempt_002`、`F attempt_002`；seq 7–10。
- 一次入口；失败不自动重试；不启动第二个 invocation 补证明。
- 新增诊断更新 16（P=4、I=4、F=8）；累计 34；旧 18 次保留。
- 累计预算 3240 秒；本轮可用余额 `2672.154590988095` 秒；R0/P/I/F 上限 600/540/900/1200 秒。
- 资源门：GPU peak≤22 GiB、host peak≤32 GiB、同 cohort 工件≤16 GiB、启动空闲盘≥20 GiB；当前约 **33.34 GiB**。
- 原根 83 文件 / 7,835,408,541 bytes 不变；旧 329 failure / 629 gap 未复验。

## 4. 确认方式

请明确回复以下任一种满足方案 §9 的形式：

1. **逐项确认**：列出十个 checklist 键并明确写“同意/确认”；或
2. **整体确认**：明确写“确认上述十项全部同意，并同意按绑定值执行一次 T6”。

仅在收到上述明确确认后，才会生成 `authorized=true / template_only=false`、十项 checklist 全 `true` 的正式 v3 授权记录，并执行一次 T6。此前不写正式授权、不写真实 claim、不启动真实链。
