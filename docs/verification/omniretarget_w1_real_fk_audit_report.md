# W1 真实训练解码、FK 几何与输出空间导数审计报告

日期：2026-10-28。按用户对[预注册方案 §9.2](../plans/omniretarget_w1_real_fk_geometry_derivative_audit_prereg_draft.md)的明确授权，真实 **R0→G→D→F 各执行一次**，退出码 0，结论为 **`real_saved_fk_covered_contract_passed`**。`complete_evidence=true`、`covered_contract_passed=true`，`failures=[]`、`gaps=[]`、`verification_problems=[]`；四阶段均为可消费的 `done/complete=true`。

本轮覆盖固定保存输入的训练解码、15 点 FK 几何、FK 损失及其输出空间梯度，未观察到该覆盖层的实现偏离。整体仍为 `objective_coverage=partial`、`full_objective_contract_passed=null`；归因为 `attribution_insufficient`，未选择干预因子，`auto_execute=false`、最终交互 `not_ready`、生产接入 false。

## 1. 授权与执行身份

运行根：`derived/experiments/omniretarget_first/20261028_2083249_w1_real_fk_audit/`。启动时 HEAD 为 **`208324943326360bf6afc14d84f3a430223b3f58`**，工作树 clean；授权、准入包与注册绑定的 11 个文件一致。

| 记录 | 路径 / SHA-256 |
| --- | --- |
| 明确授权 | [authorization.json](../../derived/audit/w1_real_fk_audit/authorization/attempt_001/authorization.json)：`846b16422f2be8a731914f4aeffd471154d5cdd13d17a972fb99a1ae9ae04ba4` |
| 最终准入包 | [attempt_014/admission_package.json](../../derived/audit/w1_real_fk_audit/admission/attempt_014/admission_package.json)：`f113887b87ad27757e105a58a42807fe90d3d639c68168b9e55a345ef9b116c8` |
| 冻结方案 v0.2 | `e27f882a49896209f0e447c56c24e4433db915435d2713fa2bfd62fccd64cc7a`；执行后不修改该快照 |
| 冻结校准文件 | `e70fa5b51e0be94cd6c0f809c606648af787cd2fbe4f6ac52d39ce37557c2392` |
| 校准证据 | `8f46fbe1830d4576f61ff85c8384a78e057ff3f5a900299e5c8c3440e3df0ed9`；未放宽阈值 |
| 执行注册 | [registration.json](../../derived/experiments/omniretarget_first/20261028_2083249_w1_real_fk_audit/registration/attempt_001/registration.json)：`93a7096fd58139d52351763befc673f9be49c42882fc898ac063f9582a33d6bb` |
| 终局 | [fk_audit_final.json](../../derived/experiments/omniretarget_first/20261028_2083249_w1_real_fk_audit/finalize/attempt_001/fk_audit_final.json)：`c1b8e0e7136b053dbed6d9710993fae3112896d8560571237aefb4c3a35ab674` |
| 独立复核 | [independent_verification.json](../../derived/experiments/omniretarget_first/20261028_2083249_w1_real_fk_audit/finalize/attempt_001/independent_verification.json)：`9432985de7cb6f3dc19ab6b19876d897846bd9f7373df9fc45bed456530bb647` |

授权记录引用用户原指令，九项 checklist 全为 JSON `true`；四阶段预算、预留与最多两次尝试精确冻结，十项禁止新增的调用/更新计数均声明为 JSON 整数 0。新 cohort 为 `w1_real_fk_geometry_derivative_audit_v1`；启动前固定根历史为 known、0 run/成本/attempt/调用，未借用父审计或旧实验额度。

父运行 `20261027_b45cd7c_w1_real_objective_audit` 仅提供不可变保存输入，七份来源工件逐文件哈希一致，父 R0/P/F 消费者状态均可消费。四窗为 `sub10_largebox_063`（T=133）与 `sub10_largebox_085`（T=148），各 `[0,60)`、`[30,90)`；两段均属 sub10，不声明跨受试者覆盖。

## 2. 执行前启动计时修复

执行前发现原 R0 起点晚于解释器/模块导入，与方案 §6.1 的计费口径不符。提交 `2083249` 在任何真实 cohort/attempt/数组读取/计算之前补齐：官方入口接收父启动器记录的单调时钟，R0 事前登记、截止与成本使用同一起点；未提供父时钟时使用 CLI 的首次导入起点。非法起点在建目录前拒绝；过期启动在读取真实数组前停止。问题、矩阵、阈值、cohort、预算与调用上限均未改变。

修复后模块 **77 passed（51.04s）**，另存完整纯合成证据 `synthetic/20261028_startup_accounting/`（`synthetic_contract_passed`，登记成本 11.5438s），再生成最终准入包与授权。旧 `review11`、旧准入包与校准保持原字节。

正式启动命令仅执行一次：

```bash
uv run --locked python derived/audit/w1_real_fk_audit/authorization/attempt_001/launch.py
```

[执行记录](../../derived/audit/w1_real_fk_audit/authorization/attempt_001/execution.json)保存官方 `--real --device cuda` 参数、线程环境、起止时刻及日志哈希。北京时间 19:52:28.505–19:52:49.493，外层进程实耗 **20.9881s**；R0 注册前已发生的 **2.6591s** 启动成本包含在 R0 内。

## 3. R0 准入与运行时

R0 的全部 13 个检查通过，涵盖身份、父来源/消费者、固定窗口、有效损失权重、输入 schema/域、环境、校准与键数。

- 资产闭包：4 个锁定文件及 37 个引用，全部可读且哈希绑定；闭包摘要 `46cb786a27522a8f0cc14625114a1295ab4e8b2cc36f39f2b0be2ff04f41f2b2`。G/F 实际编译锁定资产；参考 robot-only 模型 `nq=36/nv=35`，29 关节与 15 site 的顺序/地址/位姿检查通过。
- 输入：父保存预测、target、FK 点、有效 float32 常数及 root/关节限位 schema 均通过；预测两模型各 6,960 个关节归一化值在 clamp 内部、边界/外部均为 0，四窗 6D 旋转无无效域。真实预测并未覆盖 clamp 边界，边界拒绝语义由合成反例保留。
- 生产 CUDA/float32，参考 CPU/float64，CPU 数值线程 1；Python 3.12.13、Torch 2.14.0+cu126、CUDA runtime 12.6、NumPy 2.5.3、SciPy 1.18.1。
- 父/子均实际应用并复核：TF32 off、matmul `highest`、autocast off、`deterministic_algorithms=false`；未静默改变确定性设置。

## 4. G：解码、几何、标量与输出梯度

**122/122 主表行通过**。其中的 20 条 FK 梯度行另存明细并由 F 对账，不能再加作 20 条独立覆盖。以下仅汇总已封口行内保存的误差，不在 F 后重新读取 NPZ 数值计算。

| 比较族 | 行数 | 保存的最大绝对误差 | 最大 `bound_max_ratio` |
| --- | ---: | ---: | ---: |
| decode | 48 | 4.2718e-7 | 0.17766 |
| production float32 FK | 16 | 6.5308e-7 m | 0.11998 |
| CPU/float64 FK 核 | 12 | 5.5944e-7 m | 0.45665 |
| 父 FK 点绑定 | 16 | 逐位一致 | 逐位门 |
| FK 标量/归约 | 10 | 1.4504e-9 | 0.01656 |
| FK 输出空间梯度 | 20 | 1.0646e-9 | 0.07992 |

decode 行涉及不同原量，不将其跨单位最大值解释为单一物理误差。`bound_max_ratio` 为逐行冻结容差的保存值；梯度另要求分组 relative-L2/cosine、非零/零向量及域判据通过，不能仅靠该比例宣称通过。FK 标量使用 live 生产值，核对元素平方距离、点数分母、窗口/batch 归约和权重；梯度仅针对保存 prediction leaf，不涉及网络参数。

## 5. D：固定状态有限差分

**32/32 固定状态、4,672/4,672 行通过**，每个状态同时覆盖 35 维切空间与 38 维归一化输入，两个固定步长均完整；D 的 9,344 次姿态求值已计入调用账本。

| 族 / h | 行数 | 最大绝对误差 | 最大 `bound_max_ratio` |
| --- | ---: | ---: | ---: |
| tangent35 / 1e-5 | 1,120 | 3.7164e-11 | 1.6220e-5 |
| tangent35 / 1e-6 | 1,120 | 3.2862e-10 | 1.4051e-4 |
| normalized38 / 1e-5 | 1,216 | 2.2245e-10 | 4.5627e-6 |
| normalized38 / 1e-6 | 1,216 | 3.5515e-10 | 3.0295e-5 |

这些是状态空间 FD，不是参数 FD；`parameter_fd=0` 保持不变。

## 6. F：磁盘重读与独立进程重放

F 重读 G/D 行表和三份 NPZ，分别核验哈希/schema，再在新 Python 进程中从父保存输入重算 G+D。worker pid **3587323**、ppid **3586981**，`start_method=spawned_python_subprocess`、`exit_reason=ok`。工件读取、父后端准入、worker 后端准入、geometry replay、FD replay 五单元全部 admitted。

122/4,672 键集、完整行字段、独立梯度明细、保存数组与 fresh 重推均通过冻结判据，`verification_problems=[]`；完整 worker delta 与其自身终态 counts 一致，F 预留已结算。F 的独立性来自新进程/新后端/磁盘重读；参考数学实现与 G 共用，不宣称第三套数学证明。

## 7. 成本、尝试与调用账本

| 阶段 | 登记成本（s） | 累计阶段预算（s） | 阶段收口预留（s） | 尝试 |
| --- | ---: | ---: | ---: | ---: |
| R0 | 4.085243 | 180 | 30 | 1/2 |
| G | 5.150468 | 420 | 60 | 1/2 |
| D | 4.204449 | 600 | 90 | 1/2 |
| F | 10.807986 | 600 | 90 | 1/2 |
| 合计 | **24.248145** | **1,800** | 不互借 | 各一次 |

登记成本来自 index/ledger 完整精度字段，包括启动、真实计算、读写与每阶段至少 1s 的保守 seal-tail 计费，因此高于外层实耗 20.9881s。阶段收口预留与 seal-tail 计费不是同一字段。固定根重算历史 known、仅本 run；盘×索引×台账键集一致，无 running/publishing，三处调用预留均 `consumed_complete=true`。

| 实际调用 | G | D | F | 累计 / 上限 |
| --- | ---: | ---: | ---: | ---: |
| LossAssembler | 2 | 0 | 2 | 4 / 8 |
| 生产 float32 FK batch | 4 | 0 | 4 | 8 / 16 |
| CPU/float64 FK batch | 3 | 0 | 3 | 6 / 12 |
| production VJP | 2 | 0 | 2 | 4 / 8 |
| 参考 mj_forward | 1,200 | 9,344 | 10,544 | 21,088 / 42,176 |
| mj_jacSite | 7,200 | 0 | 7,200 | 14,400 / 28,800 |
| 其中 FD pose eval | 0 | 9,344 | 9,344 | 18,688 / 37,376 |
| scene setup forward（另计） | 1 | 0 | 1 | 2 / 4 |

R0 上述调用全零；G/D/F 增量与预注册名义矩阵精确相等，索引与台账逐阶段使用同一累计快照，全部未超各阶段/cohort 上限。十项零计数均为 JSON 整数 0：checkpoint 加载/写入、conditioner/model、参数 backward/FD、optimizer step、training update、DDIM、contact 分支。真实资产/FK/输出梯度已执行，网络与训练未执行。

## 8. 封口后只读核验与证据保护

[postcheck.py](../../derived/audit/w1_real_fk_audit/authorization/attempt_001/postcheck.py)仅读取 JSON、目录和文件字节哈希；结果 [postcheck.json](../../derived/audit/w1_real_fk_audit/authorization/attempt_001/postcheck.json)为 **`passed=true/problems=[]`**，SHA `1fbbd75ec81a6caff885e3ad9d7704ea81785e650f293e3e97c09f3907400422`。

核验包括四阶段消费者、精确成本/调用快照、固定根次数/历史、三处预留结算、启动起点/截止、授权/注册/报告绑定、运行时、行计数、worker 身份及工件哈希；388 个执行前保护文件、七份父来源和 11 个绑定文件全部未变。28 个真实运行文件另存 [real_run_manifest.json](../../derived/audit/w1_real_fk_audit/authorization/attempt_001/real_run_manifest.json)，SHA `0076d362665a5801f273a5b68fe889464de6c12301d01613166fb7ee21aa647d`。

注册的 `tolerance_contract_sha256=ff6b1ea4…` 按规范化 JSON 计算，授权中的 `e70fa5b5…` 是文件字节摘要；运行内副本与冻结校准原件逐字节相等。元数据核对初版直接混比这两种摘要，初版脚本/结果保留为 `postcheck_initial.py/json`；按注册源码口径修正后通过，真实工件未改写、未增加数值调用。封口后文档修改的保护复验另存授权目录 `post_documentation_integrity.json`。

启动计时补齐后的模块测试为 **77 passed（51.04s）**；旧相关回归 107 passed（35.40s）保留为此前源码的历史结果，本轮没有冒充重跑。正式 F 封口后不再运行 FK/FD/VJP、真实资产相关测试或全项目套件。URDF 解析器关于 collision `name` 的提示保留在 stderr，未形成失败或缺口，也未修改资产。

## 9. 裁决边界

本轮通过补齐了父真实目标审计的保存输入解码/FK 几何及 FK 输出导数证据，未提供选择单因素干预的偏离。网络参数梯度、梯度累积/裁剪、AdamW 与更新循环、选模、DDIM、端到端动作质量仍未在真实数据上审计；四窗也不代表全数据或跨受试者覆盖。

不自动启动这些后续工作。`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 均保持；U0 incomplete、UR no_quality_gain、原 F not_ready、旧预测 r6 不可验收及旋转融合不接入的结论不变。
