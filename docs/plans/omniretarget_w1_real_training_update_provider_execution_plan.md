# W1 真实网络参数梯度与优化更新链路——真实 provider 接入与执行计划

日期：2026-10-31。版本：v0.7。状态：**`gate_a_implemented / gate_b_acceptance_passed / gate_c_not_authorized / gate_d_e_not_run`**。

本计划承接 v0.9 的纯合成 I/C 交付，目标是实现真实 provider、完成只读准入和 R0 元数据登记，并在取得新的 §10.2 明确授权后执行一次真实 **R0→P→I→F**。v0.7 承接 v0.6 的 Gate A/B 实施并按独立审阅修复 5 P1+4 P2，重新冻结校准、完整合成、合成接受与真实 metadata-only admission：真实 provider 与参考后端、续跑/显式 attempt 选择、F 候选预检/结算后终检/封口后最终选择、真实 admission schema、CLI 分派均已落地。Gate C（新的明确授权）、Gate D（真实 R0）与 Gate E（真实 P→I→F）仍未执行；本轮不读取真实 checkpoint payload 或源 NPZ 数值，不编译真实资产，不创建真实 cohort/reservation/experiment run，也不改变 v0.9 的历史工件。

下文 Gate A–E、测试、校准、准入及真实运行中的已执行部分以 §6 的 v0.6/v0.7 记录为准；未执行部分仍为后续待办。文档状态不代表真实授权或真实链已通过。

## 1. 固定依据与边界

当前冻结依据如下。它们在 provider 接入期间只读；哈希变化会阻断准入。

| 项目 | 当前冻结值 |
| --- | --- |
| cohort | `w1_real_training_update_audit_v1` |
| v0.9 方案 | `docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md`，SHA `e2ba8ef9edb7381dbe410f7e0e290a6fbd702323a4416eeacfee4d2d14796242` |
| tolerance contract | `derived/audit/w1_real_training_update_audit/tolerance_contract.json`，SHA `7d899d88955e2ef9660471be83bcde5c2387ddef73302bf20a8c650f7e290f72` |
| calibration evidence | `aa4bd83167ea11a7c1cb159e494269b7d843f8233bbbc272558caf9c59e0a9ed` |
| fp32 contract（canonical JSON 摘要） | `6d7a24d0cd077454120dfbaaad37fcc378b10572e4a33a26df8ca09f7e0003e1`；文件字节 SHA 另记，不与 canonical 摘要混用 |
| v0.9 synthetic registration | `43a52e9699760fb3aa4b1ebb39d98904f0ad786aa4fa30df55da3dd973f9b3e4` |
| v0.9 admission package | `admission/attempt_012/admission_package.json`，SHA `44e229f1052823a92b8376b36ea8d79aba0854db3780264b3699e16319d855be` |
| 父运行 | objective `20261027_b45cd7c_w1_real_objective_audit`；FK `20261028_2083249_w1_real_fk_audit`；逐文件 SHA 以 v0.9 方案 §1 和 R0 registration 为准 |
| checkpoint | deterministic/diffusion 两份 `formal/*/attempt_001/best.pt`，均 step-4000；逐文件 SHA 以 v0.9 方案 §1.2 和 R0 registration 为准 |
| 资产闭包 | FK 父 registration 的 `locked_assets`：`derived/robots` 下四个锁定文件及其 37 个引用文件，详见 §1.1；`resources/manifest.json` 仅补充来源信息 |
| run root | `derived/experiments/omniretarget_first/<fresh-run>`，必须是 `HISTORY_ROOT` 的直接子目录 |

真实 provider 只覆盖已注册的四个 sub10 窗、两份 step-4000 checkpoint、两模型、两次连续诊断更新、四 micro 累积、联合裁剪、AdamW 和独立 F 重放。模型选择、DDIM、epoch/loader 等价、端到端质量、第二套参数求导、参数有限差分和生产接入继续保持 `not_audited`。

数值门、gap 分类、有效 float32 AdamW 图、调用矩阵和零计数沿用 v0.9。真实数据不能用来放宽门；若必须改变输入、参考定义、阈值、更新次数或调用矩阵，必须另立 v1.0 方案，保留 v0.9 字节并重新校准、合成验收和准入。

### 1.1 实际计算资产

生产 `G1TorchFK`、`G1Spec` 与独立 MuJoCo 参考读取以下派生资产。准入及 R0 必须先核验 FK 父 registration 的冻结 SHA，再从其 `locked_assets` 继承完整预期路径/摘要并实时核对；不能以当前文件或重新生成的 manifest 自报值替代预期值。

| 文件 | FK 父 registration 冻结 SHA-256 |
| --- | --- |
| `derived/robots/g1_29dof_locked.urdf` | `83a9341689683721285660a8e3f6419a4eb91ba5a99dca6541f7335cd3ab45b9` |
| `derived/robots/g1_29dof_locked.xml` | `5b0e619d482ef4b5559d5181985ce451c207d28181a7d97746eedcaa1c79f717` |
| `derived/robots/g1_29dof_locked_w_largebox.xml` | `0ec64b96e078672e3b676cd526415454a1cf319eb4506e0e51fde650858cebfb` |
| `derived/robots/lock_record.json` | `dda2aa612a6ab24e4074eef6cead7f8263d006b026c2d7715f715d4e69904e42` |

另须逐一核验父 `references` 的 37 个引用文件（含 G1 meshes 和 `resources/robots/largebox/largebox.obj`）、引用集合、解析后的路径及软链接目标。父 `closure_sha256=46cb786a27522a8f0cc14625114a1295ab4e8b2cc36f39f2b0be2ff04f41f2b2` 仅汇总引用文件，不能替代上表四文件的逐项核验。缺失、增加、摘要或引用目标漂移均拒绝。准入只读文件字节/文本以核验闭包，不编译模型；真实构造前按同一预期重新核验。

### 1.2 接入后的身份冻结

新 `implementation.files` 必须绑定本执行计划、冻结的 v0.9 预注册方案、新 provider/测试文件、CLI/审计器，以及实际调用的 Trainer、loss、conditioner/backbone、diffusion、tokenizer、FK/旋转/来源适配 helper。同时冻结 `pyproject.toml`、`uv.lock`、Python/Torch/CUDA/MuJoCo 等版本、默认 foreach 实现源码摘要和生产 runtime 设置；缺失文件、未列入的新依赖或实现漂移均拒绝。

Gate B 的合成证据、真实准入包、授权、R0 与 F 必须消费同一份完整文件清单和 runtime identity，不能只补 provider 而漏掉本计划或导入的计算 helper。本计划或实现任何字节在授权后变化，都须重新生成受影响的合成证据与准入包并取得新绑定的授权；v0.9 预注册及父工件仍保留原字节。

预注册方案与本执行计划分别记录路径/SHA；扩展 `IMPLEMENTATION_FILES` 时将 `PREREG_DRAFT` 固定为原预注册路径，不再由列表最后一项推断。授权模板/validator须核验新增执行计划和Gate B证据绑定，10项checklist的键集不变。

## 2. 阶段门与交付物

### Gate A：真实 provider 的最小实现

新增独立的 `RealTrainingProvider` 适配层（建议放在 `src/ireset/eval/w1_real_training_update_provider.py`），让现有审计器只依赖 provider API。合成 provider 与真实 provider 分开，真实模式禁止回退到合成 fixture。

provider 分成两个生命周期：

1. `metadata` 生命周期只读取路径、NPZ 键/数组头、配置/登记 JSON、文件字节摘要和资产引用文本，不调用 `torch.load` 或解码 checkpoint payload，不载入 NPZ 数组值，不构造模型/优化器，不编译 MuJoCo/G1 场景。哈希核验的字节读取如实记录，不能称为“文件未读”。roster 使用 Gate B 合成证据冻结的静态记录，不在准入/R0 调用当前会随机构造完整模块的 `build_parameter_roster`。
2. `materialized` 生命周期只在 R0 已封口、父消费者状态可消费且授权/准入绑定通过后，作为**已登记并预留调用的 P/I/F 阶段内单元**建立。`materialize()` 负责该阶段的四窗数组、常数及来源检查；checkpoint strict load 和模型构造由受相同守卫的 `build_trainer` 按模型惰性完成，不能在 materialize 和 factory 中重复加载/构造。每阶段首次读取、传输、资产编译及失败收口均计入该阶段，R0 封口与 P 登记之间不做这些工作。

provider 必须实现现有 P/I/F 所需的最小接口：真实配置和 runtime、完整 named-parameter roster、四窗 `micro_batch`、固定 noise/alpha、真实 loss/FK 上下文、`build_trainer`、参数前态/AdamW 状态恢复、计数器绑定和磁盘工件路径。构造后的真实 named-parameter roster 必须与静态记录逐项一致。不得在 provider 中写回源数据、checkpoint、父报告或资源软链接目标。

P/F-P 使用 `with_optimizer=true`，每模型独立新建空 AdamW；I/F-I 使用 `with_optimizer=false`，仅恢复共同参数/显式参考状态，不构造原生优化器或执行生产 LossAssembler。生产 loss/FK/箱体静态场景在各 P/F-P 阶段内共享一次初始化，保持 `scene_setup_mj_forward` 的 P=1、I=0、F=1；构造和加载计数以 v0.9 §7.2 各阶段列为准，不把 materialization 当作额外免费调用。

**必须连同 provider 完成的接线：**

| 现有位置 | 后续实现要求 |
| --- | --- |
| CLI、admission builder/validator、授权模板/validator、`run_registered_audit`/R0 | 按真实/合成 mode 和 schema 分派并核验完整绑定；真实元数据门不沿用合成 fixture 的数值检查或默认校准重算，详见 Gate B/D |
| `ReferenceContext` 的 target/evaluate | 改由 provider 的独立参考后端执行；真实路径不得调用 `synthetic_fk_np`、`synthetic_fk_jac_np` 或要求 `SyntheticChain` |
| `run_f_stage`/`_worker_main` | spec 绑定真实 mode、provider 身份、实现/源文件/输入准入摘要；子进程独立重建真实 provider，禁止继续硬编码 `SyntheticTrainingProvider/load_fixture` 或依赖父进程活对象 |
| admission/full-chain 预算及 F 父子两侧矩阵检查 | 统一按 mode 生成预期；当前 F/admission 的 scene 常量 0 只适用于 synthetic。真实 scene 的 R0/P/I/F 为 0/1/0/1、全链2；synthetic 全为0，二者分别校验 |
| 续跑 CLI、`stage_state`、父消费与 F spec | 实现 §2.1 的显式 attempt 选择及父工件绑定；允许已结算失败记录与唯一可消费 attempt 共存，不再要求整个阶段只有一个终态 attempt |
| `prior_call_counts`、`budget_for`、full-chain 检查与结果字段 | 按 §2.1 分别重推选中链矩阵、各阶段及 cohort 历史累计；全链验收不能把含失败调用的累计 counter 直接与单链名义值比较 |
| F 当前 attempt、快照与封口 | 按 §2.2 实现 worker 候选预检、父进程结算后的盘上终检和封口后的消费核验；明确 `candidate_attempts/provisional_chain_counts` 与最终选择的生效时点，不要求 worker 读取自身尚未结算的 `D` |

独立参考接口固定为 `reference_positions(pos,R,q)→float64[N,15,3]`、`reference_jacobians(pos,R,q)→float64[N,15,3,35]`，点与29关节顺序沿用父冻结名单。35列为世界平移、左乘世界旋转增量（`R'=Exp(δθ)R`）、关节弧度；`D_decode=[N,35,38]`，`J38=J35@D_decode`。MuJoCo 原生自由根角速度列到该切空间的转换只在 backend 内执行一次：既有 `RealMujocoReference.jacobians` 已乘 `R.T`，通用 `ReferenceContext` 不得再乘一次；合成 backend 适配到同一接口，保持原数值路径。

真实 `mj_forward/mj_jacSite` 在实际调用入口检查 deadline/`unit_admission`，成功返回后分别加 `reference_mj_forward/reference_mj_jac_site`；不能直接沿用旧 helper 调用前加 `mj_forward` 的计数方式，也不能在 context 外再按 batch 名义量重复加计。每状态、每 site 保留已完成量及失败位置，无法确定完成量时保持 unknown。target 的解码前、解码后 FK 前守卫继续保留；不新增参数/状态 FD、预热或缓存预测以改变矩阵。

严格加载规则如下：

- 父 objective/FK registration、admission、report、数组和 verification 文件逐一按 v0.9 锁定 SHA 核对，并验证父 index/ledger/consumer 状态；不能只信任顶层 `passed`。
- 两份 checkpoint 必须按锁定路径和 SHA 读取，payload 使用 strict state-dict 键集、dtype、shape、有限性和 step/config/root_stats/identity 对账；缺键、额外键、非 finite 或 non-strict fallback 都是硬失败。始终使用 fresh step-0 AdamW，不加载历史优化器状态，不称为历史续训。
- 四窗必须严格为 `W0/W1/W2/W3` 的 demo、slice、条件字段、`raw_root/raw_q/raw_wfc` 和 tokenizer 常数；每窗 `x0=[60,38]`，按固定 micro 槽位堆叠为 `[4,60,38]`。dtype、shape、有限性、四元数/rot6d/joint 域均先检查。只消费冻结父数组，不重跑 dataset/loader；不能补尾窗、换 demo、跨 split 或使用目标真值泄漏字段。
- 真实 FK 使用锁定的 G1/largebox 资产闭包和既有 FK 审计的坐标/四元数约定；资产、XML、mesh、MuJoCo 版本和 runtime 逐项记录 SHA。普通手与 spherehand 不混用。
- 生产和独立路径必须共享同一冻结输入和共同参数前态；独立路径不得读取生产 prediction/Jacobian 缓存作为参考。

Gate A 的完成条件：接口与上述接线齐备，provider 单元测试、用 stub 构造的真实模式输入拒绝反例通过，并能由合成模块生成完整 roster。这里不读取真实权重/源数组、不编译真实资产；随机/空模块的构造在合成报告中如实记录，不冒称零构造。完整回归与校准归 Gate B。交付 provider 源文件、接口说明、测试和实现哈希清单。

### Gate B：合成回归与只读 admission

先以新实现重新验证数值契约，再生成新的 admission attempt；完整绑定范围见 §1.2，不能继续消费 `attempt_012` 的旧实现哈希。provider 测试文件、`--real-admission`、`--synthetic-acceptance` 及验收报告输出已在 v0.6 实现，v0.7 修复后重新冻结。

后续实施时必须运行以下检查。本轮不执行；所有 `<fresh-...>` 路径需先换成未占用的新目录，输出入口须拒绝覆盖已有校准/合成运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked pytest -q \
  tests/test_w1_real_training_update_audit.py \
  tests/test_w1_real_training_update_provider.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked pytest -q \
  tests/test_omniretarget_w1.py \
  tests/test_w1_de_numerics.py \
  tests/test_w1_training_objective_audit.py \
  tests/test_w1_real_objective_audit.py \
  tests/test_w1_real_admission.py \
  tests/test_w1_real_fk_audit.py
uv run --locked python scripts/audit_w1_real_training_update.py \
  --calibrate-synthetic --seed 20261029 \
  --tolerance-out "derived/audit/w1_real_training_update_audit/calibration/<fresh-calibration-attempt>/tolerance_contract.json"
uv run --locked python scripts/audit_w1_real_training_update.py \
  --synthetic --seed 20261029 --device cuda \
  --tolerance-contract "derived/audit/w1_real_training_update_audit/calibration/<fresh-calibration-attempt>/tolerance_contract.json" \
  --out "derived/audit/w1_real_training_update_audit/synthetic/<fresh-synthetic-attempt>"
```

校准必须在本 Gate 实际满足 `tolerance_contract_problems(recompute=True)=[]` 和每个门 `4×observed≤1`，保存重算报告及其实现/环境/contract/evidence/fp32 摘要。现有 `load_tolerance_contract` 默认 `recompute=True` 会运行 `synthetic_calibration`，包含网络、VJP 和 AdamW；新真实准入、`--real` 的入口/R0 元数据检查必须显式使用无计算的结构/摘要核验并绑定本 Gate 重算报告，不得触发默认重算。不以 `recompute=False` 代替本 Gate 的实际重算；真实执行时环境/dispatch 源码或数值契约漂移则退回 Gate B，不在 R0 加做未登记的校准。

上述校准入口须增加实际重算与报告持久化，在新校准目录写 `calibration_verification.json`；完整合成入口在 F 已有重放/原始量重推通过后写 `synthetic_acceptance.json`，绑定重算报告、完整 roster、所有阶段报告/行/数组、实现与环境。接受报告不得增加网络或优化器调用，不以自报 `passed` 代替逐项核验；真实准入通过下列参数显式消费该报告，并校验其引用闭包。

完整合成链的工程验收要求四阶段证据完整且可消费、264 tensor/3864行齐全、全部计数符合 synthetic 矩阵、F 新进程重放/重推通过、failures/problems/verification problems 为空。CLI 可因必测数值 gap 返回 **exit 3 / `audit_incomplete`**，不能一律要求 exit 0，也不能仅凭退出码或 gap 标签白名单放行；必须从原始数组独立重算每条 gap 的适用性、界和分类。相同 seed/runtime/数值路径下逐项比较 v0.9 数值数组和比较事实（历史为392 precision+190 sensitivity=582 gap、0 bound-unverifiable）；身份/时间/路径变化另列。数量相等不能替代逐行证据，新增差异须解释并在固定门下复核，不能抬高阈值。若契约未变，保存“contract unchanged”的逐项证据，而非只比较总 SHA。

§1 的旧 `tolerance_contract.json`、旧 `adamw_fp32_contract.json`、v0.9 合成目录和 `attempt_012` 保持原字节。新校准的 timestamp 可使新文件 SHA 不同，须另存新目录并分别绑定新文件 SHA、calibration evidence 和 fp32 canonical 摘要；不得原地重生成旧文件。后续 synthetic/admission/授权/R0/F 全部使用同一份新校准路径与摘要。

新增真实 admission 入口应明确区分 metadata-only 与 synthetic admission，例如：

```bash
uv run --locked python scripts/audit_w1_real_training_update.py \
  --real-admission \
  --tolerance-contract "derived/audit/w1_real_training_update_audit/calibration/<fresh-calibration-attempt>/tolerance_contract.json" \
  --synthetic-acceptance "derived/audit/w1_real_training_update_audit/synthetic/<fresh-synthetic-attempt>/synthetic_acceptance.json" \
  --out derived/audit/w1_real_training_update_audit/admission
```

新增独立的 `w1_real_training_update_real_admission_v1` schema，明确 `provider_mode=real/scope=metadata_only`，配套 builder、consumer validator 和 CLI/R0/F 分派。旧 `admission_package_problems` 要求 `asset_closure_verified=false`，不能直接消费新包；旧 synthetic schema 的原意保留，禁止靠删掉资产检查或放宽所有模式来兼容。

该入口只核验父锚、checkpoint/config/资产闭包、provider 接口、已冻结合成 roster/重算报告、环境和资源上界，不重跑校准或构造模块；成功/失败均不读取数组值、创建 cohort 或写入 `derived/experiments`。合格新包 `read_scope.asset_closure_verified=true`，另存真实字节/文本核验记录；失败包按实际完成事实填写，闭包未通过不得标为已核验。checkpoint payload 解码、NPZ values、asset compile、模型/优化器构造、forward/VJP/update、cohort/attempt 创建均为0。Gate B 生成 roster 所用的合成/随机模块构造另记，不能合并成“全部构造0”。合格包 `passed=true/problems=[]` 且 `authorized=false/registered=false/run=not_run/section_10_2_checked=false`；通过仅表示本模式的元数据准入。

### Gate C：新的 §10.2 明确授权

授权只能绑定 Gate B 产生的最新方案/实现/contract/fp32/admission 哈希。授权记录必须：

- `authorized=true`，cohort 和 execution identity 与注册一致；
- `AUTHORIZATION_REQUIRED` 的 **10 项** checklist 全为 `true`：新问题与父锚、fresh AdamW/诊断步、连续更新与共同前态、真实目标/参数 backward/FK/clip/AdamW、调用矩阵与 unknown 结算、预算/尝试/资源、合成 fp32/零值/传播门、零 FD/DDIM/选模/checkpoint/contact、publishing 三方撤回、partial/not_ready/no-auto；
- `parameter_fd`、`state_fd`、`ddim_calls`、`validation_calls`、`selection_calls`、`production_checkpoint_writes`、`contact_branch_executions` 全为 JSON 整数 `0`；
- 精确绑定四阶段预算、调用 caps、§1.2完整实现清单、本执行计划与 v0.9（或新 v1.0）方案、Gate B重算/合成证据、tolerance evidence、fp32 contract、新 schema 的真实 admission package 和父 registration；
- 明确真实链结论仍只适用于本方案覆盖范围，不能据此自动训练、选模或接入生产。

在授权记录通过 `authorization_problems()` 前，不得调用 `--real`，不得创建真实 cohort。初次注册只允许一个新的、空的 direct-child run root；不覆盖任何历史 attempt。若需在额度内重试，使用 §2.1 的显式续跑入口，在同 root/cohort 追加 attempt；首次入口的新鲜性检查不能被绕过或拿来覆盖非空目录，不以更换 run 名重获额度，不自动重试。续跑入口及其消费/对账协议同属 Gate A/B 交付和本次授权绑定范围。

### Gate D：真实 R0 元数据登记

首次 `--real` 入口先完成授权/准入结构与绑定、路径新鲜性和历史可归属性检查，再建立 metadata provider。授权缺失/无效、历史 unknown 或无法合法分配 R0 时拒绝在正式根外，不创建 cohort 或 attempt，也不读取真实数组/权重值。授权有效、历史可定且预算允许后，先创建 cohort、写入 R0 running 的 index/ledger 并持久化 reservation，再进行 R0 的真实文件身份和资源核验。启动器在 uv/import 之前记录单调起点，入口和 metadata 准备成本纳入获准 R0。续跑时若 R0 已可消费，按 §2.1 复用其明确身份，不再次分配 R0 或生成噪声；需要重试 R0 时才追加其新 attempt。

R0 全程禁止读取 checkpoint payload、载入 NPZ 数组值、模型/优化器构造、前向/反向、FK/Jacobian、更新和资产编译，也不得通过校准重算间接执行这些操作。metadata provider 使用新准入包绑定的静态 roster 并核对实现身份。

R0 需要完成：

1. 检查当前工作树/实现文件身份、方案和授权/准入绑定、uv/Python/Torch/CUDA/TF32/autocast/matmul/runtime 设置。
2. 核验父 registration/admission/producer/consumer、checkpoint/config、四窗 metadata、资产闭包、磁盘/显存/主机资源和 16 GiB 工件上界。
3. 核验完整 roster/row catalog、固定 micro schedule/noise/alpha、四阶段预算和累计调用余额；reservation 必须在对应 stage 开始前落盘。
4. 写入 R0 registration/admission，封口后再次读取 index/ledger/registration 并做三方调用余额对账。

**元数据与数值输入检查的执行时序细化：**当前 `run_r0/_fixture_schema_problems` 会访问实际数组和独立解码，真实模式必须分离。R0 只冻结窗口键/头部 schema、来源 SHA、静态 roster、micro schedule、8个噪声 draw unit、RNG 前后态及 alpha 配置/摘要。预注册 §3.2 中依赖 `x0` 的独立 float64 `x_t` 保存，延后到已登记的 P 物化准入单元，在任何网络 forward 前完成；输入、有效 binary32 alpha、公式、顺序、阈值及调用数均不变，此时序补充须随本计划一起授权。

P 先核验四窗数值/域、有效 tokenizer 常数与独立编码，并用 R0 已冻结的 t/epsilon 生成独立 `x_t`，不增加随机 draw 或原生 add_noise。结果写入该 P attempt 的不可覆盖 `input_admission` 工件，绑定 R0 registration、源文件和数组/常数摘要，由 P 报告封口；不回写已封口 R0。每模型 strict load 后、该模型首次 forward 前另核对实际 roster/checkpoint root_stats/原生 alpha 等与该输入契约相符。I/F 独立重读/重核源输入和该工件，不以 P prediction/Jacobian 为参考。

噪声计数固定为 R0=8、P=8、I=8、F=24，全链48：P/I 各自原生路径生成8对并逐位对账；F 另重建R0一次8对，再为F-P/F-I各生成8对。构造、materialize、参考 `x_t` 计算不得额外 draw；加载冻结噪声与实际重新生成必须区分计数，不能把额外生成伪记成读取。

已登记 R0 内的父消费者不可消费、实现漂移、资源不足、检查异常或截止超时均保留 cohort/attempt，按实际成本与调用结算并封为 terminal `incomplete`；写盘/结算不能确认时保留非可消费的 publishing/unknown，不假造终态、不删除失败记录。P 不得启动。成功 R0 的名义调用只有 `cpu_draw_units=8`；失败时记录实际完成的 0–8 个 draw unit，其余真实网络/FK/更新调用为 0，不补足名义数。

### Gate E：真实 P→I→F

R0 封口且可消费后，按以下顺序启动 P；I 和 F 同样先登记后读取/构造：

1. 记录阶段 `t0`，核对同 cohort 剩余时间、尝试数、调用和资源余额，分配 attempt。
2. 写入 running index/ledger、父 registration、完整调用 reservation 和工作截止，绑定 counter 与 `unit_admission`。
3. 检查守卫后执行该阶段的 `materialize()`，随后按模型调用 `build_trainer`。不存在 R0/P 阶段之间的免费初始化或 checkpoint 预加载。
4. 读取、strict load、schema、资产编译或构造失败均在当前 attempt 中捕获、保留实际调用并结算封口；不开始前向或后续阶段。无法确认调用余额时保持 unknown。

| 阶段 | 固定工作 | 同 cohort 累计时间预算/预留（已含） | 尝试上限 |
| --- | --- | ---: | ---: |
| R0 | 元数据、父锚、runtime、资源、roster、registration | 240s / 30s | 2 |
| P | 两模型各两次连续原生更新、真实 loss/FK、完整参数梯度/状态/clip/AdamW | 900s / 120s | 2 |
| I | 共同前态的独立输出 VJP、真实 FK/Jacobian、参数 VJP、显式 clip/AdamW 和区间重推 | 900s / 120s | 2 |
| F | 新 Python 子进程，从盘重读并逐位重放 P/I | 1200s / 180s | 2 |

总阶段预算为 3240s；阶段不能互借。`deadline=t0+remaining-reserve` 是唯一工作截止，父进程、worker、`unit_admission` 和每个 FK/VJP/clip/update 单元直接比较该绝对截止，不重复扣 reserve。

资源保持单 GPU/单计算 worker，GPU peak≤22 GiB、host peak≤32 GiB、同 cohort 工件≤16 GiB、启动时磁盘 free≥20 GiB；新元数据、input_admission 和 provider 工件也纳入逐类上界。7.145 GiB单链/14.236 GiB cohort 为 v0.9 已计算上界，接入后按实际新增 schema 重算并证明仍≤16 GiB，不能直接复制旧估算或把上界当作真实峰值。

更新数严格按以下阶段分配，名义全链为 **16 次**（native 8 + explicit 8），P/I 合计 **8 次**；同 cohort 硬上限仍为 32：

| 阶段 | native AdamW | explicit reference | diagnostic/training updates |
| --- | ---: | ---: | ---: |
| R0 | 0 | 0 | 0 |
| P | 4 | 0 | 4 |
| I | 0 | 4 | 4 |
| F-P / F-I 合计 | 4 | 4 | 8 |
| 一次完整链 | 8 | 8 | 16 |

`training_updates=native_adamw_steps+explicit_reference_updates=diagnostic_parameter_update_events`，各值均为严格 JSON 整数。其余调用沿用 v0.9 §7.2 完整矩阵（每 attempt 的调用上限为该阶段一列、同阶段累计及 cohort 上限为对应名义值的两倍）。单链名义值与历史累计的适用范围见 §2.1。每个生产/参考单元执行前检查 deadline、父 registration/cohort/running 状态和调用余额；执行后才增加对应计数。任何已开始的 FK 单元按实际完成调用计数，不能把未执行单元补写为 nominal。

F 必须在独立 Python 进程中完成，pid 与父进程不同且 ppid 正确。worker 在已登记 F 的 reservation 下使用独立本地 counter 和同一绝对截止，不另分配额度；materialization、读取和 P/I 重放均计入 F。它重新读取 checkpoint、输入及 input_admission、RNG、报告、NPZ、rows 和 registration；逐字段对账 loss/parts/gnorm/clip coefficient/RNG digest/array hash/state presence/row multiset/cotangent groups/原始量分类。worker finally 上报全24类及零计数的非负 JSON 整数快照；父进程仅在完整核验、合并成功后 settle，缺失/部分/非法快照保持 `consumed=null`、历史unknown。父子F检查均按真实scene=1，最终全链=2，不复用synthetic的0。检查分工与生效时点按 §2.2 执行，worker 预检通过不代表当前 F 已结算或可消费。F 的来源/重放/证据/封口失败或未知只能封为 `audit_incomplete`；F 成功复核的数值负结果按 §4 分类，不得保留错误的通过声明。

### 2.1 同根续跑、attempt 选择与累计对账

本节为拟实现协议。当前 `stage_state` 在终态 attempt 数不等于1时返回 `stage_not_terminal`，full-chain 检查直接比较累计 counter 与 `CHAIN_NOMINAL`；两者均不能直接用于保留失败记录的续跑，须连同 CLI 和 F 消费端一起修订。

1. **入口与起点。** 拟新增独立的 `--resume --stage <R0|P|I|F>` 入口，要求与初次运行相同的授权、准入、contract、实现/runtime、源输入和既有 root/cohort/execution identity；不重新写 cohort 或重置额度。指定阶段必须是首个未开始或不可消费的阶段，其上游必须已可消费。已完整封口的数值 failure/gap 仍按 §4 进入 F，不能为取得更好数值结果重跑或替换该 attempt。P 的工程重试从冻结 checkpoint、空 AdamW 和诊断步1重新开始，不从中断步的部分状态接着训练；I/F 同样按原阶段整列重做。有效续跑的启动/import、元数据核验及收口成本计入首个新执行阶段，起点在 uv/import 前记录；已完成上游不重新计费或补做调用。
2. **先核验全部历史。** 盘上 attempt 集合、index、ledger、reservation 必须完整一致；所有旧 attempt 均须已有可信终态、有限成本和完整调用结算。已结算的 `incomplete/crashed/timeout` 可保留为历史失败；任一旧 running/publishing、缺失/部分快照、`consumed=null`、身份或三方对账未知，均阻断续跑，不能通过选择另一个成功 attempt 排除。核验剩余时间、attempt、逐类调用及工件额度后，才在原 root 分配下一个未占用编号并落盘 running/reservation。阶段内历史核验只可将自身正在执行且身份/截止/完整预留已核验的当前 attempt 单列，不要求它提前终态或已结算（F worker 对应父 F attempt）；这不使它成为可消费选择，也不豁免身份、调用及预算守卫。当前 F 的候选增量与最终消费按 §2.2 核验，旧 attempt 不享有该例外。
3. **明确选择与父绑定。** `stage_state`/消费者接受显式的所选 attempt，先完成上述全历史核验，再要求该 attempt 为唯一可消费候选（index/ledger 均 done、complete=true，报告/admission 的证据完整且闭包匹配）。允许一个已结算失败 attempt 与一个可消费 attempt 共存；多个可消费候选、错 stage、未封口或被撤回的选择均拒绝，不按目录顺序、最大编号或“最后一次成功”猜测。每个新阶段的 running index/ledger、报告及 F spec 保存 `parent_attempts`：P 绑定选中 R0 的 attempt、registration/admission SHA；I 再绑定选中 P 的报告、全部数组和 `input_admission` SHA；F 再绑定选中 I 的报告/行/数组 SHA。跨边不一致即拒绝。F spec/worker 记录 `candidate_attempts={R0,P,I,F}`，其中 R0/P/I 已可消费，F 为绑定的当前 running attempt，不对自身调用最终 `stage_state` 或提前判为可消费。父进程的 F 最终报告列出 `selected_attempts={R0,P,I,F}` 的明确编号，该完整选择仅在 §2.2 的 terminal 后消费核验通过时生效；F 自身报告摘要由其 index/ledger 绑定，不在报告内形成自哈希循环。旧单 attempt 合成消费保持兼容。
4. **只复用已封口上游。** 重试 I 时沿用已选 R0/P，重试 F 时沿用已选 R0/P/I；旧上游只做元数据/摘要和消费状态复核，不重做其 draw、strict load、forward 或 update。F 原有 P/I 重放仍属于 F 的固定列。新 attempt 不覆盖旧工件或失败原因，不能把已有下游接到不同上游；阶段和 worker 每次读取工件都使用已登记的明确路径/摘要，禁止隐式查找 latest。

设 `D[s,a,k]` 为阶段 s、已结算且可信终态 attempt a 的第 k 类实际调用增量，从 reservation 的 `consumed` 读取，并与 index/ledger 的 `final-before` 逐类相等。新 attempt 在运行登记时持久化 `call_balance_before`，终态两侧保留它及 `call_balance_after`；不得按目录/列表顺序猜测前态。当前 F 尚未封口时的候选增量按 §2.2 核验，不冒充终态 `D`。全部24类和零计数按现有 JSON 整数规则核验；失败 attempt 允许实际部分量，不补齐到名义列。以下三种量须分别保存并独立重推：

| 量 | 定义与验收 |
| --- | --- |
| `selected_chain_counts` | 只对 `selected_attempts` 的四个 `D` 求和；各选中阶段须精确等于其按 mode 生成的名义列，完整链精确等于 `CHAIN_NOMINAL(mode)`：16次更新、48个 draw unit、真实 scene=2 / synthetic scene=0 |
| `stage_call_counts` | 对同 cohort 各阶段的全部历史 attempt 增量求和，含失败、中断和未选中记录；每 attempt 不超过该阶段名义列，每阶段累计不超过该列×2，尝试最多2次 |
| `cohort_call_counts` | 按原固定历史根归属规则，对同 cohort 全部 attempt 的增量求和，逐类不超过 `CALL_CAPS`；不能把每行累计快照再次求和，也不能仅取选中链或以跨 root 的 max 代替历史对账 |

`counter` 的历史余额及结果 `calls`/`training_updates` 如实反映 `cohort_call_counts`；另列完整选中链的 `selected_chain_counts`/`selected_chain_training_updates=16` 和未选中 attempt 的实际增量。summary/coverage/boundaries 等重复字段也须标明范围并从对应计数派生，不能残留将 cohort 更新量固定写为16的声明。尚未形成完整选中链时按已证实的阶段增量报告，未知保持 null/incomplete，不预填16。例：P 的首次失败 attempt 已完成1次 native 更新，随后整列重试成功并完成 I/F，则选中链仍为16次，cohort 累计为17次（native=9、explicit=8、diagnostic=17）；不能因17≠16否决正确的选中链，也不能把17改写成16。历史中断原因与已测数值负结果均保留，适用的负结果仍按 §4 裁决。

续跑启动前，对每个待执行阶段证明“该阶段历史累计+完整新列”不超过其×2上限，并证明“cohort 当前累计+尚需执行阶段的完整列之和”不超过 cohort caps；复用阶段不再次加名义列。时间、attempt 和资源余额也必须分别满足，失败成本、工件和调用不返还、不借用。F 的父子检查按 §2.2 分阶段执行：候选全链预检与终检只比较所绑定四阶段的调用列，终态选中链确认再使用 `selected_chain_counts`；各阶段及 cohort 全部历史上限始终另算。选择正确不能豁免历史 unknown/超额，历史累计合法也不能豁免选中链漏调用。

### 2.2 F 当前 attempt 的预检、结算与封口

F worker 执行时，当前 F 必须仍为 `running/consumed=null`；父进程等待 worker 返回后才能结算。以下候选检查解决这一时序，不把当前 F 当成已可消费阶段，也不放宽任何旧历史要求。

1. **worker 候选预检。** 从盘核验 `candidate_attempts` 的 R0/P/I、父绑定及全部旧历史，要求上游唯一可消费、旧 attempt 全部可信终态且已结算。当前 F 只核验其已登记身份、running/reservation、调用余额和绝对截止。所有计数单元结束后冻结本地最终增量 `L_F[k]`（本地 counter 从0开始），此后不再执行计数单元；以 `provisional_chain_counts=所选R0/P/I的D之和+L_F` 检查按 mode 的完整名义矩阵，同时按“全部旧历史+L_F一次”分别检查阶段/cohort 调用上限和 attempt 上限。当前 F 无须已有 `consumed`、`call_balance_after` 或可消费终态，不能因此报历史 unknown。`provisional_chain_counts` 只作预检，不替代已结算 `D` 或正式 `selected_chain_counts`。finally 将候选编号、预检结果与同一全24类/零计数快照写入 `worker_result.json`；发生异常仍上报实际完整快照及预检未完成原因，不补足名义量。
2. **父进程结算后终检。** worker 返回后，父进程先核验进程/身份/快照完整性、各项 JSON 整数及 reservation/累计上限，成功合并一次后才 settle 当前 F。预检或重放失败但调用快照可信且可合并时，仍结算实际量并保留失败；快照缺失、非法或合并失败保持 `consumed=null`、历史 unknown。结算后将 `call_balance_before/after` 持久化到尚不可消费的 F index/ledger 待封口记录，再从盘重读上游、全部旧历史、当前 F 的已结算 `consumed` 及两侧余额，逐类验证 `after-before == consumed == L_F`，独立重推候选全链矩阵、历史上限、父绑定及 worker 预检事实。此时只暂缓当前 F 自身的 terminal 要求；不要求其已可消费，也不依赖 worker 自报通过。任一工程/证据终检失败按原撤回协议封为 incomplete，完整计数和失败成本仍保留。
3. **封口后确认最终选择。** 候选工程/证据终检通过后，父进程写 F 报告及拟定的完整选择/统计，按既有 publishing→terminal 协议绑定报告、数组、余额与成本。封口后再按通常消费者规则核验当前 F 与三个上游均唯一可消费，并从四个终态 `D` 重推 `selected_chain_counts`；只有该次核验通过，报告/结果中的完整 `selected_attempts`、`selected_chain_counts` 及16次选中链更新才作为终值生效。未通过时执行 pre/post-seal 撤回、重写报告和重绑哈希，候选记录另存，完整选择及其派生终值保持未确认/null，不保留通过声明。完整数值 failure/gap 仍按 §4 保留并分类，不因候选检查改写成工程失败或完整通过。

所有元数据重读、核验、写盘及撤回成本均归当前 F attempt；沿用原工作截止、尾部预留和封口规则。上述预检/终检不新增 worker、draw、forward、VJP 或更新，调用矩阵、预算和资源上限不变。

## 3. 新增测试与反例

真实 provider 接入至少新增以下测试；新增测试使用合成输入、临时文件和 metadata/MuJoCo API stub，不读取真实权重/源数组或编译真实资产：

- 登记前授权/准入/历史拒绝时，零真实调用且不创建 cohort/attempt；初次注册与同root续跑明确区分，续跑保留既有成本/调用/attempt累计，拒绝超额或覆盖旧记录；
- 续跑正向链：R0 或 P 的 `attempt_001` 可信失败封口后，追加 `attempt_002` 成功，保留两条终态且该阶段选中002；P/I/F 消费明确父绑定，F 完成独立复核并可消费。P 首次已更新1次的例子必须保存选中链16次/cohort17次、全部历史成本和工件，不能触发旧 `stage_not_terminal` 或把17误判成单链矩阵失败；
- 续跑复用：分别从 I、F 开始重试，已封口上游不再分配 attempt、身份/数组不变且不新增其阶段调用，后续未完成阶段按原顺序正常登记；P 的部分更新重试必须恢复冻结权重/空 AdamW 再做两步，不续接部分状态。证据完整的数值负结果 attempt 继续进入 F，不能借续跑替换；
- 选择与历史拒绝：多个可消费候选、所选 attempt 未封口/撤回、父 attempt 或报告/数组/input_admission 摘要错配、任一旧失败记录未结算，均在新实质调用前拒绝；选中链齐全但历史超额、历史合法但选中链漏调用分别失败，不能删旧记录、重复累加快照或靠 latest/max 掩盖；
- 已登记 R0 的失败仍无 `torch.load`、NPZ payload read、MuJoCo compile、真实 model construction、FK、VJP 或 optimizer，但必须保留 cohort、失败 attempt、实测成本、实际 draw 计数及 reservation/index/ledger；封口不明时不可消费；
- 真实 admission 的独立 schema/validator 可消费 `asset_closure_verified=true`，legacy synthetic schema 仍维持原义；CLI/R0/F 均拒绝错 mode/schema，元数据入口不触发校准重算/随机 roster 构造，字节读取与payload读取分别如实记录；
- 父 registration/admission/report/array/checkpoint/config/asset 任一字节漂移、缺失、路径穿越、软链接目标漂移都会 fail-closed；
- checkpoint 缺键、额外键、wrong step、wrong dtype/shape、非 finite 或 non-strict fallback 都拒绝；
- 四窗缺字段、重复/缺窗口、slice/shape/dtype/domain/finite 错误、真值泄漏字段和 noise/alpha 不一致都拒绝；
- 普通手/spherehand 或 G1/largebox 资产混用、XML/mesh/版本不一致、FK backend 未绑定都会拒绝；
- 非单位根旋转的解析 stub 验证 J35 世界左乘切空间、J38 组合及点/关节顺序，漏转/重复 `R.T` 均失败；真实 mode 不触及 synthetic FK，逐状态/site慢调用验证截止、调用后计数和部分失败，不按 batch 补足；
- R0 封口且 P running/reservation/counter/deadline 已落盘后才 materialize；慢读取/构造及失败成本计入 P，I 不构造原生 optimizer，F 初始化计入 F；P 超时、父状态变化、调用余额不足、reservation 未结算时不启动后续单元；
- 真实路径不意外调用 synthetic provider；`--real` 缺授权、admission、contract、空 run root 或绑定漂移时零真实调用退出；
- F worker 的真实 provider 重建、pid/ppid、RNG 恢复、报告/数组/行多重集、完整调用快照、unknown 结算和撤回封口保持与 v0.9 反例一致；真实scene P/F各1、synthetic各0，在admission、F父子和full-chain检查均一致；
- F 时序正向：首次完整链及同根 F 重试均在当前 F `running/consumed=null` 时通过候选预检，worker 不要求自身终态 `D`；记录事件顺序，快照落盘/worker 返回先于父合并结算，结算先于父盘上终检，终检先于 terminal 封口，封口先于最终消费确认。最终选中链16次，失败历史另计，不能提前发布完整选择或额外执行计数单元；
- F 时序反例：旧 F 未结算仍阻断；当前完整快照下预检失败仍结算实际量并保留失败，缺失/非法/不可合并快照保持 unknown；worker 预检通过后篡改候选身份、父绑定、`consumed` 或两侧余额，父终检必须否决；封口后消费核验失败须撤回完整选择/通过声明并保留成本，不能跳过终检或把候选16次当作已确认终值；
- R0 只核验 header，P 首forward前的数值/input_admission与独立 `x_t` 拒绝坏值/常数，I/F重新核对，所有noise构造路径仍精确8/8/8/24；
- 证据完整的数值 failure 或 precision/sensitivity/bound gap 保留全行后仍进入 F；来源、域、预算或证据失效才阻断后续阶段；
- 新校准/合成输出拒绝覆盖历史文件，§1 保护文件逐字节不变；阶段更新分别为 P=4、I=4、F=8，全链16，与完整调用矩阵一致；
- 本计划、新provider或任一实际计算helper/依赖身份漂移，均被准入、授权、R0与F绑定检查拒绝；Gate B对exit 3的接受要求完整原始证据与独立分类，不能吞掉测试失败或任意incomplete。

Gate B 运行前列出的专项文件、六个相关文件及新增provider测试，报告实际收集/通过数；55项专项和263项相关回归仅为v0.9历史计数，不要求新增后仍恰好相等，也不在通过后无因重复跑同一套。现有 `test_omniretarget_w1.py` 会通过 `G1Spec` 读取锁定资产，须单列为既有资产回归，不能把整组宣称为纯合成。校准/新增provider测试保持上述stub边界；全项目套件可补充，未运行时明确注明。

## 4. 中断条件、负结果复核与结论

下列任一条件发生，立即停止新的实质计算、记录原因并执行当前 attempt 的结算/撤回/封口；证据不完整时不进入后续数值阶段，不重试到另一个 run 名、不借用其他 cohort 额度、不调整阈值：

- 身份、父锚、checkpoint、输入、资产、config、runtime 或实现哈希不一致；
- provider schema/domain/finite/roster/strict-load/共同前态检查失败；
- deadline、stage budget、attempt cap、call cap、artifact/host/GPU/disk cap 或 reservation 对账失败；
- F 重放 mismatch、报告/数组/rows 漂移、worker snapshot 不完整或 terminal 封口不可验证。

**有限且域有效、来源/证据完整的数值负结果不触发上述提前中断。** P/I 的直接数值 failure，及 `parameter_delta_precision_gap`、`cross_path_sensitivity_gap`、`cross_path_bound_unverifiable` 等数值覆盖 gap，均保留原始量、全部可计算行和原始分类；只要预算和所有准入守卫仍满足，就完成当前固定矩阵并以 `complete_evidence=true` 封口，继续进入独立 F 重放/重推。不得因首条数值 gap 截断矩阵，也不得复制 P 前态来抹去 step-1 failure。遇到非 finite、无效域或无法形成所需证据时仍按提前中断处理。

| F 后裁决 | 必需条件 |
| --- | --- |
| `real_training_update_covered_contract_passed` | 选中四阶段证据完整可消费、无 failures/problems/gaps、选中链矩阵与独立重放通过，全部历史已结算且预算/调用/attempt/资源上限通过 |
| `real_training_update_mismatch` | 证据完整可消费、无必测 gap，存在 F 独立重推确认的直接数值 failure |
| `audit_incomplete` | 任一必测 gap、来源/域/预算/证据失效、重放不一致或不可消费封口；保留已测 failures |

数值 gap 的最终结论为 `audit_incomplete`，不能写成 AdamW 实现错误或完整通过；`complete_evidence` 与数值通过是不同字段。即使数值负结果完整复核，也不得把它写为生产可用。

只有选中四阶段 `complete_evidence=true`、无 failures/problems/gaps、选中链调用矩阵、全部历史上限与 F 逐位复核均通过时，覆盖范围内才可写 `real_training_update_covered_contract_passed`。历史失败原因与已测负结果保留，不能以 attempt 选择清除适用的数值失败/缺口。即便通过，`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 仍保持。

## 5. 计划产物

实现阶段预期产生：

- `RealTrainingProvider` 源文件、CLI wiring、provider/schema/拒绝路径测试；
- 新实现 identity、合成回归/校准重算记录、`calibration_verification.json`/`synthetic_acceptance.json`、新目录的 tolerance contract/fp32 绑定（数值契约不变也另存）；
- 新的 metadata-only admission package（下一个未占用的 admission attempt），以及用户单独提供的授权记录；
- 获授权后才产生的 `derived/experiments/omniretarget_first/<run>/` R0/P/I/F 工件；
- 每阶段明确的父 attempt/摘要绑定，F 的 `candidate_attempts/provisional_chain_counts`、快照/结算后终检、封口后生效的 `selected_attempts` 与选中链矩阵、各阶段/cohort 全历史累计及失败记录；
- 执行后的独立 verification 报告和文档索引更新。

v0.9 的 `synthetic/v0.9_final/`、`admission/attempt_012/`、历史校准和本计划均保留。获新授权前不创建真实cohort、不读取真实checkpoint payload或源NPZ数值；元数据只读核验与既有资产回归的范围按Gate B分别记录。不提前更新“已执行”状态，也不把本计划当作授权记录。

## 6. 文档修订与复审

v0.2 按首轮4 P1+2 P2修订了实际资产闭包、校准独立目录、materialization阶段计费、完整数值负结果进入F、R0拒绝/失败分离和P/I/F更新计数，并把授权checklist订正为10项。

v0.3 的源码对照复审补充了真实admission schema及消费端分派、ReferenceContext/F worker真实后端接线、J35角坐标只转换一次、逐状态/site调用后计数、scene矩阵按mode校验、R0元数据与P数值输入/独立x_t分工、禁止元数据入口隐式重算校准、exit 3负结果验收和完整执行身份绑定。校准重算/合成接受报告须持久化并显式供准入消费；专项/回归计数改为历史参考，既有资产回归和合法续跑的范围分别说明。

**v0.3 静态复审历史（2026-10-31）：当时记录 `plan_review_passed`，本轮已订正。** 当时对照冻结的[v0.9预注册方案](omniretarget_w1_real_training_update_audit_prereg_draft.md)、[I/C验证记录](../verification/omniretarget_w1_real_training_update_audit_verification.md)、现有 CLI/审计器/真实 FK helper及父 registration JSON 复核了生命周期、后端/坐标接口、输入与证据绑定、预算/调用/更新/负结果分类及各门的拒绝条件；随后独立审阅指出同根重试与单终态消费/累计矩阵比较冲突，因此该历史判断不再作为当前版本的通过声明。

v0.3 只读检查历史：24类计数器、10项授权键、四阶段预算/预留、16次更新/48个draw unit/2次真实scene初始化与源码常量一致；四锁定文件预期SHA及37个引用与父JSON一致；两段命令模板仅做`bash -n`语法检查；200个受保护源码/配置/冻结方案与旧证据文件SHA未变。未导入项目模块，未运行pytest、校准、准入、模型/FK或真实实验；Gate A–E均未执行，真实授权/注册/运行状态不变。

**v0.4 审阅修订（2026-10-31）：`plan_review_revised`，修订1项P1。** 补充 §2.1 的显式续跑起点、全部历史核验、唯一可消费 attempt 选择、下游/F 父工件绑定、选中链矩阵与历史累计分开重推，以及16/17次更新的正向验收。对应 CLI、`stage_state`、历史余额与 F 检查列入 Gate A/B 必改接线；预算、名义调用列、16/32次更新和数值门不变。本轮只修订方案与索引，实现和新增验收仍待执行，不能把本次文档修订标为真实链通过或授权。

v0.4 文档静态检查：版本/索引状态、链接/代码围栏/空白、两段命令模板的 `bash -n`、源码常量及16/17次更新算术核对通过；254个受保护的源码、测试、配置、依赖和其他文档文件 SHA 未变（含冻结 v0.9 方案）。本轮未导入项目模块，未运行 pytest、校准、准入或真实实验，未读取真实 checkpoint、NPZ payload 或 G1 资产。

**v0.5 审阅修订（2026-10-31）：`plan_review_revised`，修订1项P1。** 拆分 F worker 候选预检、父进程快照核验/合并/结算后的盘上终检、terminal 后最终消费确认；当前 F 不再依赖尚未产生的终态 `D` 或可消费状态。补充候选编号/临时统计与完整选择的生效边界、正常首次/同根重试时序及结算/封口反例。对应接线与验收仍为 Gate A/B 待办，本轮只修订方案和索引，未实施 provider 或执行真实链；原数值门、名义调用矩阵、16/32次更新和3240s预算不变。

v0.5 文档静态检查：版本/索引、链接/代码围栏/空白、两段命令模板的 `bash -n` 及冻结依据/预算/更新/裁决四组表格对照通过；253个受保护的源码、测试、配置、依赖和其他文档文件 SHA 未变（含冻结 v0.9 方案）。本轮未导入项目模块，未运行 pytest、校准、准入或真实实验，未读取真实 checkpoint、NPZ payload 或 G1 资产；新增时序验收仍待实现后执行。


**v0.6 实施与 Gate B 验收（2026-10-31）：`gate_a_implemented / gate_b_acceptance_passed / gate_c_not_authorized / gate_d_e_not_run`。** 本轮按 Gate A/B 实施：
- 新增 `src/ireset/eval/w1_real_training_update_provider.py`：`RealTrainingProvider` 的 metadata 生命周期只哈希文件/读取 NPZ header/解析 JSON 与资产引用文本；materialized 生命周期在已登记 P/I/F 阶段读取四窗、常数、R0 噪声与父数组，strict load 两份 checkpoint、按模型惰性构造，写入 P 的 `input_admission`。真实 MuJoCo 参考后端按状态/站点检查绝对截止、成功返回后分别计数；合成后端适配同一 `reference_positions/reference_jacobians` 接口，`ReferenceContext` 不再要求 `SyntheticChain`。
- 审计器新增显式 attempt 选择、`_cohort_chain` 调用增量重推、`selected_chain_counts/stage_call_counts/cohort_call_counts` 分离、`--resume --stage`、F worker 候选预检（`provisional_chain_counts`）、父进程结算后盘上终检与封口后最终选择；scene 计数按 mode（synthetic 全 0、real P/F=1、全链 2）。
- 新增 `w1_real_training_update_real_admission_v1` builder/validator 与 `--real-admission`，校准写 `calibration_verification.json` 并拒绝覆盖，完整合成链写 `synthetic_acceptance.json`；授权模板/validator 增加执行计划、校准验证与合成接受 SHA 绑定，10 项 checklist 键集不变。
- Gate B 执行：新 provider 测试 **20 passed**；既有审计模块 **55 passed**；六个相关回归 **263 passed**。校准在新目录重算通过（`recompute_problems=[]`、4× 全部 ≤1），tolerances/rules/tiny-zero floors 与 v0.9 逐项相同，仅环境（内核版本/CPU 线程）使新的 fp32 canonical/evidence SHA 与 v0.9 不同。完整合成链四阶段 done/sealed、264 tensor/3864 行、调用矩阵精确、`training_updates=16`、F 新进程重放；结论 `audit_incomplete`（`complete_evidence=true`、`failures/problems=[]`、582 条必测 gap：392 precision + 190 sensitivity + 0 bound-unverifiable），与 v0.9 逐项一致。`synthetic_acceptance.json` passed；真实 metadata-only admission passed（`asset_closure_verified=true`、`npz_values_read/checkpoint_payload_read/torch_load_calls/model_constructor_calls/muJoCo_compile_calls=0`、父锚/checkpoint/资产闭包/264 tensor/3864 行/真实 scene=2 全过）。最终运行路径与 SHA 以验证记录 §10 为准。
- 首次尝试的中间运行使用 `OMP_NUM_THREADS=1` 且在 `i_accumulated:diffusion:u1:model.net.root_out.weight` 出现 1 条近阈值 failure（582→585 gap）；撤销线程覆盖后 failures=0、gap=582，与 v0.9 一致，故该中间运行不纳入 Gate B 依据，保留为线程环境影响证据。阈值、窗口、lr、dtype 未调整。
- 边界：真实 checkpoint payload、父 NPZ 数值、G1 资产编译、真实模型/优化器构造、真实 R0→P→I→F、真实授权/注册/cohort 均未执行；Gate C/D/E 状态不变，v0.9 校准/合成/准入/方案字节保持不变。

**v0.7 审阅修复与 Gate B 重新冻结（2026-10-31）：`5 P1 + 4 P2` 全部修复，`gate_b_acceptance_passed` 重新成立，Gate C/D/E 仍未执行。**
- **P1 物化前守卫**：`RealTrainingProvider` 现在在 `materialize` 入口、父数组摘要/解码、tokenizer、input admission、MuJoCo 参考后端构造，以及 `build_trainer` 模型构造/`_load_checkpoint` 的 SHA 与 payload 读取前执行同一 deadline+guard；截止或父登记失效时不再启动数组读取、编译或 `torch.load`。
- **P1 源数组实际摘要**：每次 materialize（含续跑复用已封口 R0）先重新计算父 `objective_arrays` 文件 SHA 与 train config SHA，并解码前与冻结值比较；`input_admission.source_sha256` 改为实际摘要。文件漂移直接拒绝，不再携带旧摘要。
- **P1 实现冻结清单**：`IMPLEMENTATION_FILES` 由 15 项扩为 25 项，补入 `model/tokens.py`、`model/deterministic.py`、`kinematics/torch_adapter.py`、`kinematics/rotation_adapter.py`、`geometry/distance_adapter.py`、`embodied/g1_spec.py`、`paths.py`、`eval/training_objective_audit.py`、`pyproject.toml`、`uv.lock`，并用静态依赖测试锁定。
- **P1 合成接受文件闭包**：`synthetic_acceptance.json` 为每个阶段保存 attempt、目录、逐文件路径/字节/SHA，另绑定 fixture 与 run 级 `cohort.json/attempts_index.json/costs.json`；消费端逐文件复核存在性/大小/SHA、阶段必需文件与报告路径，并把 runtime identity 纳入拒绝条件。删除/改写/搬移任一证据文件都会使真实准入失败。
- **P1 运行时身份对账**：fp32 contract 冻结 Python/Torch/NumPy/SciPy/CUDA/设备/MuJoCo 版本与 `torch.optim.adam{,w}` 源码 SHA；校准验证、合成接受、真实准入、R0、F 子进程与 P/I 物化均做无数值计算的身份比较。漂移使旧数值契约不可消费并退回 Gate B。
- **P2 续跑调用量**：`required` 只汇总 `resume_stage` 及其下游阶段的名义调用，已结算的上游与失败 attempt 只留在历史。新增“P 失败 2 次加载→P/I 成功→F 失败已结算 4 次→resume F”反例：10+4=14≤16 合法，旧 10+8=18 的误拒被消除。
- **P2 续跑起点**：`first_start` 只归入 resume 阶段本身；P 之后的 I、I 之后的 F 各自使用新起点，上游执行/封口耗时不再重复计入并提前耗尽下游截止。
- **P2/P2 选择与完成硬门**：显式 attempt 也必须先检查全阶段唯一可消费候选，两个可消费 attempt 下指定其一仍拒绝；`complete=true` 是消费硬条件，`attempt_not_complete` 不再被过滤，两个 `done/complete=false` 也不能被返回为可消费。
- **测试**：provider 模块 **31 passed**（新增 11 项守卫/摘要/文件闭包/身份/续跑/选择反例），审计模块 **55 passed**，六个相关文件 **263 passed**，合并 **349 passed（378.47s）**。Gate B 重新执行写入 `calibration/attempt_007`、`synthetic/attempt_007`、`admission_real/attempt_007`；最终 SHA、调用矩阵、582 gap、16 次更新、F 新进程与工件大小以验证记录 §10 为准。v0.9 方案/校准/合成/准入字节不变。
