# W1 真实参数梯度与更新链路首次执行报告

日期：2026-11-02。冻结依据：[预注册 v0.9](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md) 与 [执行计划 v0.11](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_provider_execution_plan.md)、Gate B attempt_012。

本次按用户明确授权启动真实链一次，结果为 **`audit_incomplete` / exit 3**。Gate C 已授权，Gate D 的 R0 完成，Gate E 在 P 的扩散模型首次前向之前停止；I/F 未启动，未自动重试。

用户原指令：

> 同意草案十项，授权按固定预算执行一次真实 R0→P→I→F（3240 秒、16 次诊断更新），不自动重试。

[正式授权记录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/authorization.json) SHA `199bfb30f11502aaf57acf38f1c49525ac58e511f870f230f223cb12b8525d78`，十项 checklist 全为 true；绑定的授权草案 SHA `66d1261d5f0ff93a49d44861cc4983830223015e5749f879a7ade3672612226f`。预算、调用上限、资源及七项整数零声明沿用冻结草案。

运行根：[20261102_2704709_w1_real_training_update_audit](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit)。启动器在 uv 和项目 import 前记录时钟，并将同一起点传给冻结 CLI；[启动记录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_launch.json) 与 [退出记录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_exit.json) 保存完整命令、起点、进程和退出码。未修改执行入口或 25 个绑定文件。

| 阶段 | attempt | 终态 | complete | 登记成本（秒） | 预算（秒） |
| --- | --- | --- | --- | ---: | ---: |
| R0 | attempt_001 | done | true | 58.599905234 | 240 |
| P | attempt_001 | incomplete | false | 8.502827990 | 900 |
| I | 未创建 | 未运行 | — | 0 | 900 |
| F | 未创建 | 未运行 | — | 0 | 1200 |
| 合计 | 首次入口一次 | audit_incomplete | false | 67.102733224 | 3240 |

预算函数以 `measured_wall_s` 求和；`wall_s` 仅为一位小数展示。登记成本含封口时计入的保守 seal-tail，外层实测 66.205312315 秒另列，不拿外层耗时覆盖账本。

R0 的元数据准入 passed/problems=[]，checkpoint payload 解码、NPZ 数值读取、模型构造及 MuJoCo 编译均为 0；仅完成 8 个 CPU draw unit。R0 registration SHA `6abab029788f0e87859a560e3e9866c9958758f5a8765b7092f34ca5d53848aa`。P 已在登记及 reservation 后读取父数组、完成 `input_admission`（passed/problems=[]，SHA `b1dbefcff2323a2b12b148df92f9b9ced12ca8f6269d4010656c7891fffcb041`）、加载两份 checkpoint、构造两套 model/conditioner/optimizer，并执行真实生产目标/FK/反向及裁剪。

| 实际累计计数 | 数量 |
| --- | ---: |
| checkpoint loads / model constructions / conditioner constructions / native optimizer constructions | 各 2 |
| conditioner forwards / backbone forwards / production loss assembler / parameter backward / micro units | 各 8 |
| production float32 FK batch | 16 |
| native global clip / native AdamW steps / diagnostic parameter update events | 各 2 |
| CPU draw units（R0=8，P=0） | 8 |
| scene setup mj_forward | 1 |
| native add_noise / explicit reference updates / independent VJP / reference mj_forward、jacSite | 0 |

实际 `training_updates=2`，不是计划全链的 16 次。按执行顺序，两次原生更新发生在 deterministic；P 的独立参考与 F 重放尚未发生，因此不能宣称这两次更新的参数链路通过验证。FD、DDIM、验证/选模、生产 checkpoint 写入和 contact 分支未启动；R0 的七项零计数已落盘，P 的完整报告缺失，不补造其未保存明细。

停止原因保存在 [原始输出](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_stdout.log)（SHA `24c3c31a329e2c37aa151615f3a0fd309c1fbc5964901547f59a3c36cb7abb29`）：

```text
p_stage_exception:ValueError:production diffusion alpha schedule differs
```

[provider 检查](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_provider.py:1672) 在 strict load 之后，把真实 DDPM `alphas_cumprod` 提升为 float64，与 `effective_alpha(self.alpha)` 作 `np.array_equal` 比较；不一致即拒绝返回该 trainer，扩散前向未启动。[独立构造](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py:700) 使用 float64 linspace、平方与 cumprod，再于 effective-alpha 层转 binary32；生产路径由 Diffusers DDPMScheduler 构造。本次未在账外重建 scheduler 或计算差值，具体坐标/ULP 和修复方式尚未实测确认。

[R0/P 索引](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit/attempts_index.json) 与 [成本账本](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit/costs.json) 的终态、complete、成本及调用余额一致；两份 reservation 的 consumed 均已结算，逐类满足 `after-before=consumed`，且未超原预留与 caps。不存在本地 running/publishing 或 consumed=null 的未结算记录。

P 缺少 `p_report.json`、`p_rows.json`、`p_diffusion.npz`。部分 `p_deterministic.npz` 实际存在，但未列入 P 的封口 artifacts，不能消费或当作完整成功 P；原文件保留。CLI 在 P 的提前返回路径没有写入 `audit_result.json`，也未产生 F final，未事后补造正式报告。

[执行观察记录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_observation.json) SHA `54ca9d2ad155fdac77c1d4d60acd3037cfb8ec289af3d31f492ebd72ab1394e4` 仅从 JSON、源码文本、文件字节摘要和已哈希 stdout 整理本次事实，明确 `is_formal_stage_report=false`。它保存 10 个原运行文件的 SHA/bytes、完整 24 类调用、异常和缺失工件。原 cohort 共 466,350,106 bytes（约 0.434 GiB），未删除失败记录或部分数组，未重载 checkpoint/NPZ 数值，未追加数值探针、校准或 pytest。

25 个绑定实现文件及 6 份主要冻结材料字节复核无漂移；预注册、执行计划、Gate B attempt_012 与正式授权原字节保留。本次只在授权范围执行一次，执行后另写观察记录与文档，未修补生产或审计代码。

`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 保持。旧 U0/UR/原 F/r6/旋转融合结论保持。

后续先在纯合成范围定位真实 scheduler 与冻结 alpha 的有效求值差异，修复后重新冻结受影响的 Gate B 材料；不得通过直接放宽等值门、改 dtype/阈值或重置 run 名绕过当前结果。真实恢复还须处理既有 cohort 的实现/授权绑定，并保留已消耗的 2 次更新、P attempt 和成本；本次授权的一次运行已结束，不自动 `--resume`。
