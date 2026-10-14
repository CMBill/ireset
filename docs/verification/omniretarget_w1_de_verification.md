# OmniRetarget W1 D/E 实施与正式验证记录

日期：2026-10-13。对应[执行方案](../plans/omniretarget_w1_de_plan.md)。**I0–I4、R0/R1、D、E、val6 和 F 全部完成；有限动作基线的证据交付为 `complete`，最终交互方法为 `not_ready`。** 两模型从头完成各 4,000 个 optimizer updates；val31 主比较中 Teacher raw 通过 2/31，Deterministic raw 和 Diffusion raw 均通过 0/31。完整质量负结果已保留，没有缩分母或修复后冒称 raw。

权威终局为 [F 报告](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/decisions/attempt_001/w1_de_final.json)，`complete=true`、`problems=[]`、`budget_stopped=false`。本记录区分训练完成、比较证据完整和最终可行交互；`motion_baseline=complete` 表示本轮有界诊断完成，不表示预测通过搬运验收。

## 1. 身份、来源与执行范围

只读 SOURCE：`derived/experiments/omniretarget_first/20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8/`，其前置事实见 [A3/B/C 验证 §5R7](omniretarget_w1_a3bc_verification.md)。本轮 RUN：`derived/experiments/omniretarget_first/20261013_be00c50_w1_de/`。下文的产物短路径相对 RUN。

本轮执行身份为 `w1_de_v1`，训练来源为 published original，用途为 `weak_motion_only`，标签协议为 `omniretarget_motion_v1`，接触损失权重为 0。固定 train24、val31、val6，训练不加载 S0 权重。没有新增教师生成、扩名单、rot/trans 增强、test31、续训、修复器调用或按质量筛选的重采样。原 U0 `incomplete`、独立 UR `no_quality_gain` 保持。

运行 HEAD 为 `be00c50e9665dce9978156e780441b9712d16751`，这是方案登记 revision。实际运行包含尚未提交的实现，**不能将 HEAD 单独称为本轮实现提交**。正式注册记录了 5 个已跟踪修改文件和 8 个未跟踪文件的内容身份：tracked diff SHA-256 为 `5cd15d88c2c1812f2f286ce631cc3454a2676c40e3bdf2b3d1f6d8c8267fd86f`。原字节保存在 [implementation_snapshot 清单](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/implementation_snapshot/snapshot_manifest.json)，可在登记 HEAD 上应用 `tracked_diff.patch` 并恢复 `untracked/` 文件还原实现；清单 SHA-256 为 `b9fcf8f26bf2b22da2748fd4c868ef0d115e41cdf5ea1ce225abc32ea8886ebc`。本次未创建 commit 或 push。

[执行注册](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/execution_registration.json) SHA-256 为 `5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e`，执行方案 SHA-256 为 `bed1e62d5c85579e96b94695cbbbb067cb79a820f18b9b692d21bda9c1ec1525`。条件代码 SHA-256 保持 `d73f411d874614d4b6b23f4686b4cd0eaa1d42b6aa33110dfa1facc9c2aebb8d`；训练 YAML、标签协议和 uv.lock 不变。正常数值路径对拍通过，复用已验收 SOURCE 的 A3/B/C。

实际执行时间为 **2026-10-13 12:26:02–15:09:28（+08:00）**。运行期间代码、配置、文档身份固定；串行链末端 `identity_unchanged=true`。运行退出后、编辑本文和导航前，又完成一次注册/三方状态/预算/原始身份核验，见 [terminal_verification](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/terminal_verification.json)。文档登记改变当前工作树身份，但不回写历史注册或权威产物；历史核验应使用保存的实现快照。冻结执行方案首页的 `not_run` 保留为起草时快照，当前状态由本文、[文档索引](../README.md)及 F 产物说明。

## 2. I0–I4 实现与必要验收

| 实施项 | 实际交付 |
| --- | --- |
| I0 | [统一驱动](../../scripts/omniretarget_w1_de.py)：SOURCE 事实复验、独占注册、跨运行根累计预算、scope/盘/索引/台账核验、事前 running、进程组硬超时、原子落盘及最终登记后超限撤销 |
| I1 | [训练入口](../../scripts/train_generation.py)与 [Trainer](../../src/ireset/train/trainer.py)：正式配置/名单绑定、真实 optimizer update 和样本计数、输入至更新的数值检查、完整验证日程、RNG/loader/module-mode 恢复、best/last 与证据保存 |
| I2 | [raw 入口](../../scripts/eval_generation.py)：显式消费选中 checkpoint，完整重建 scheduler/config/model/conditioner，核验权重有限性、内嵌统计及来源/缓存/资产/代码身份 |
| I3 | raw 固定 155 键、Teacher 同布局产物、完整轨迹先保存后测量、逐行失败保留、条件指标 defined 分母、[生成器](../../src/ireset/model/generator.py)限定数值重试 |
| I4 | [val6 detail](../../scripts/omniretarget_w1_de_detail.py)：主产物复用、逐帧曲线及实际 MuJoCo GIF；[F 收口](../../scripts/omniretarget_w1_de_finalize.py)：训练/选择重推、155 产物重新测量、聚合/通过状态复算、固定 checkpoint 重放与状态分列 |

数值拒绝与质量拒绝可结束负诊断；输入不可证、缺键、未测量、超时、未运行或未知历史使比较 `incomplete`。不同 run-root 不重置执行预算。末次保存、内容哈希、索引和台账登记也受截止约束，超限撤销同一 attempt，不追加重复完成行。没有实现精确训练 resume。

完整项目回归命令：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --locked python -m pytest -q tests
```

结果为 **790 passed、1 skipped，1832.06s，exit 0**，包含本轮 152 项新增测试；原基线为 638 passed+1 skipped。已有未注册 pytest `slow` 标记与 OSQP 弃用提醒产生 11,410 条 warnings。测试原输出和登记见 [test_record](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/test_evidence/test_record.json)；日志 SHA-256 为 `ba03c315449a12f804a0e0d97cdb1eff098862b26c60ca957744a630eb825513`。

| 新测试文件 | 数量 |
| --- | ---: |
| [test_omniretarget_w1_de.py](../../tests/test_omniretarget_w1_de.py) | 17 |
| [test_w1_de_driver.py](../../tests/test_w1_de_driver.py) | 16 |
| [test_w1_de_numerics.py](../../tests/test_w1_de_numerics.py) | 45 |
| [test_w1_de_raw.py](../../tests/test_w1_de_raw.py) | 57 |
| [test_w1_de_closure.py](../../tests/test_w1_de_closure.py) | 17 |

实际小夹具执行了 TrainConfig/CLI、两模型从头优化、完整生成回调、checkpoint、raw、detail 单元 CLI/实际 EGL 渲染和 F 重测/重放；整链通过后仅翻转 raw 的通过标记，第二次 F 正确拒绝并登记 incomplete。单独整链验收为 1 passed/230.31s，完整回归也包含它。该夹具使用 CPU、2 步、缩小名单，生产注册拒绝这些短配置；固定 155 键另有反例锁定。真实子进程及后代击杀、末次登记延迟、running 残行、非有限 loss/gradient/weight、缺选模行、foreign checkpoint、数值恢复与接受窗保持均有覆盖。

实现验收发现并修正两项必须记录的事实：31 段均值的 `np.mean` 与 Python `sum/len` 可差 ulp，同源精确对账统一采用注册公式；CPU 和 CUDA 的同 seed 初始噪声不同，E 和 F 均消费每模型注册 device，设备不可用保留 `checkpoint_unavailable`，不静默回退。本轮生产设备为 CUDA。两模型初始化、首成功生成/解码、训练/损失/验证、global/loader RNG 正常路径对拍通过。

## 3. R0/R1 与 SOURCE 边界

[R0 事实报告](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/preflight/attempt_001/preflight_report.json)的 registration/A3/B/C 四项检查均 passed，reasons 为空。SOURCE 旧五门 true 主张与本次事实重算分别入册；350 份展开文件重哈希。R1 独占创建、未复用旧注册。驱动成本 37.536342s，入口进程墙钟 38.938517s。

固定数据为 train24 **4,351 帧/110 窗**，val31 **5,681 帧/145 窗**，55 段共 10,032 帧。train24 顺序清单 SHA-256 为 `8fbfd008b98f56196d31f81352582006ba95c16bdcf2a4da2f5da56eb8267948`。统计只消费 SOURCE train root_stats。源箱尺度、源/目标箱分离、参考日程及推理条件语义保持；教师 G1 根/关节/FK 或真值脚落点不进入推理条件。

主环境为 Python 3.12.13、PyTorch 2.14.0+cu126/CUDA runtime 12.6、MuJoCo 3.13.0，RTX 3090 24GiB；CUDA 前向/反向实测通过。OMP/MKL/OpenBLAS/NUMEXPR=1，DataLoader workers=4、worker 原生线程=1。启动前文件系统约 185GiB、内存约 117GiB 可用，无其他训练进程；环境和依赖未修改。

SOURCE 的已知质量缺陷继续作为前置事实：train19/24、val29/31 strict 失败，均含 penetration，其中 train3/val2 还含 velocities。源几何尺度约 0.680025–0.799525；源→目标 pure +z 全55段，train5.761–10.133cm、val6.052–15.959cm。E 重新测量 Teacher raw，没有用这些旧摘要替代本轮比较，也没有因 strict 缺陷剔除弱动作监督标签。

## 4. D：正式训练与选模

两模型按顺序从头训练，固定 FP32、micro_batch=4、grad_accum=4、有效 batch=16，AdamW lr=1e-4、weight_decay=1e-5、grad_clip=1。主干 d_model=240、4层、6头、FFN×4、时间窗31；动作窗/步长60/30。seed=val_seed=20261008，contact=0。扩散为1000 timesteps、sample/x0-prediction、clip_sample=False、scaled_linear beta 1e-4→0.02，完整生成为 DDIM25/eta0/seed7。

| 事实 | Deterministic | Diffusion |
| --- | ---: | ---: |
| 真实 optimizer updates | 4,000 | 4,000 |
| micro-batches / backward micro-batches | 16,000 / 16,000 | 16,000 / 16,000 |
| 实际窗口样本 | 64,000 | 64,000 |
| train / val 可用窗口 | 110 / 145 | 110 / 145 |
| 完整 loss 验证 | 16次，每次145窗 | 16次，每次145窗 |
| 完整生成验证 | 4次，每次31/31 | 4次，每次31/31 |
| 最终窗口 val total | 0.3876883852 | 0.2504169529 |
| total 首/末100步均值 | 2.34526605 / 0.03432935 | 2.27093108 / 0.03713954 |
| first100 同步耗时均值，ms/step | 975.4 | 964.0 |
| peak CUDA allocated，GiB | 5.47 | 5.47 |
| Trainer 内部墙钟，s | 4,178.8 | 4,444.1 |
| 驱动 measured_wall_s | 4,196.265506 | 4,461.705839 |

每 epoch 为27个整 micro-batch，实际消费108窗，drop_last丢弃2窗；两模型各 started593/completed592 epochs。micro-batch、optimizer update 和样本计数分别登记。两模型均无提前停止、数值失败、保存错误或额外 attempt。total 是加权训练损失，不与完整生成 MPJPE 混作同一度量。

完整 val31 教师拟合 FK MPJPE 演示宏平均，单位 m：

| optimizer step | Deterministic | Diffusion |
| --- | ---: | ---: |
| 1,000 | 0.15791176074810398 | 0.16797982422129645 |
| 2,000 | 0.12285097225281030 | 0.12434786805848430 |
| 3,000 | 0.12998539707574497 | 0.11302992122157389 |
| 4,000 | 0.12204454465921158 | 0.11134758983898290 |

四次验证名单、T、numerical status 和指标均完整，两模型 best 均为 step4000。Deterministic 的 step3000 回退没有覆盖较好的 step2000；最终 step4000 按严格改善更新。窗口 val loss、碰撞或通过率未用于选模。[checkpoint_selection](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/evaluation/checkpoint_selection.json)和 F 从历史/索引/权重重推一致。

| checkpoint | SHA-256 |
| --- | --- |
| deterministic/attempt_001/best.pt | `7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff` |
| deterministic/attempt_001/last.pt | `b3e64f48119044dbe366fe218f270c7a4cef3b1a549db368f395e57867e7d283` |
| diffusion/attempt_001/best.pt | `1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1` |
| diffusion/attempt_001/last.pt | `09c1e1488ebda7d1490ada8edb7affde26c41018fb15bbd9ce2c44c771cffabd` |

两模型驱动成本累计 **8,657.971345s（2.405h）≤43,200s**。历史 S0 的2,561.3s仍属于 C，不计作本轮 D 的步数或费用。训练拟合下降和完整选模成立，不提供最终交互通过证据。

## 5. E：固定裸输出比较

[raw_eval](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/evaluation/raw/attempt_001/raw_eval.json)精确覆盖93个主键和62个扩散附键，共155条。五个 method/seed 组各 expected=attempted=processed=31；没有重复、外来键、缺行、无产物行、checkpoint/标签失败、timeout、error 或 not_run。全部数值有效、保存和测量完成，均覆盖最后一帧；没有额外数值重试。每模型/seed的整段生成共175个窗，多于窗口损失验证的145窗，来自全长生成的尾窗补齐规则。

`retry.model_calls=1` 表示一次生成采样批调用，Diffusion 的这一次调用仍执行25次DDIM去噪，不表示只做一次网络前向。Teacher 的 model_calls=0。两模型原始产物的实际尝试均为1、种子为所用主/附 seed、最终无效窗为0。raw 中 `repair_pipeline_invocations=0`。

### 5.1 主 seed 计数与拟合

`raw_standard_pass = numerical_valid ∧ robot_strict_pass ∧ task_gate_pass`；task 门为逐帧关键点平均误差均≤0.40m，等价于整段 max≤0.40m。p95只作诊断量。下表拟合和任务值是完整31演示宏平均。

| 方法 | 数值有效 | robot strict通过 | task通过 | raw通过 | 教师拟合，m | 人体任务mean，m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher raw | 31/31 | 2/31 | 31/31 | 2/31 | 0，自参考 | 0.06211101 |
| Deterministic raw，seed7 | 31/31 | 0/31 | 27/31 | 0/31 | 0.12204454 | 0.13299047 |
| Diffusion raw，seed7 | 31/31 | 0/31 | 28/31 | 0/31 | 0.11134759 | 0.12123616 |

Teacher通过的两段是 `sub10_largebox_002` 和 `sub7_largebox_032`。E重新测量的模型教师拟合宏与D记录差约2e-16m，符合独立浮点计算容差，不能将所有跨阶段统计都描述成逐位相等。

Deterministic task超界段为 sub12_largebox_073、sub16_largebox_048、sub3_largebox_029、sub7_largebox_038，max分别为0.456781/0.455092/0.485342/0.400475m。Diffusion主seed超界为sub12_largebox_073、sub3_largebox_029、sub7_largebox_038，max为0.424177/0.426903/0.500192m。逐帧首次超界位置和完整失败帧列表在raw行中保存。

### 5.2 穿透、运动与固定场景冲突

机器人最大穿透使用注册 strict 的自适应插值/SLERP；违规比例只按关键帧计算。每段最大值的宏平均与全集合最坏值分别报告：

| 方法 | penetration失败段 | velocities失败段 | 最大穿透范围，mm | 每段最大值宏平均，mm | 关键帧违规比例宏 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher raw | 29/31 | 2/31 | 1.617–42.958 | 14.722 | 0.025603 |
| Deterministic raw，seed7 | 31/31 | 7/31 | 19.951–222.388 | 146.527 | 0.844398 |
| Diffusion raw，seed7 | 31/31 | 5/31 | 36.769–246.746 | 145.797 | 0.812345 |

Teacher最坏穿透段为sub3_largebox_018；两个模型主seed最坏段均为sub7_largebox_038。joint_limits、foot_support、hand_grasp 的记录失败数均为0，这些检查记录不证明源参考的持续抓握或支撑已保持；外部参考对账的接触覆盖仍单独判定。

`box_ground` 按注册从机器人strict排除，另行实测；固定箱体保持原始输入。五个组的场景测量相同：25/31段的最大深度>2mm，范围0–18.117mm，每段最大值宏平均5.067mm。没有移动箱体或改变其他碰撞族来增加通过率。

### 5.3 接触、脚滑与拼接

双手公共参考交集为25个有参考演示、共2,813帧；其余6段的条件指标按零参考语义为null。左右手参考适用演示为27/28；左右脚支撑适用演示为12/31。下面是这些固定适用演示上各段指标的宏平均，不是完整31段的无条件均值。

| 条件指标 | Teacher raw | Deterministic seed7 | Diffusion seed7 |
| --- | ---: | ---: | ---: |
| 左手gap p95，m（27段） | 0.026887 | 0.121776 | 0.148441 |
| 右手gap p95，m（28段） | 0.018777 | 0.102342 | 0.096304 |
| 左手最大箱系锚漂，m（27段） | 0.213467 | 0.410416 | 0.385182 |
| 右手最大箱系锚漂，m（28段） | 0.216534 | 0.433435 | 0.403960 |
| 左脚最坏site速度p95，m/s（12段） | 0.255671 | 0.811822 | 0.810310 |
| 右脚最坏site速度p95，m/s（31段） | 0.221207 | 1.049504 | 0.889107 |
| 双手valid覆盖（25段） | 0.541550 | 0 | 0 |
| 双手certified覆盖（25段） | 0.008524 | 0 | 0 |

Teacher只有4段有非零双手认证交集，最长13帧；两个模型主seed均为0段、最长0帧。valid近接覆盖不能冒充逐区间认证；手认证包含gap、箱系逐site漂移和相对速度，脚速排除首帧占位零。raw/F亦保留逐侧contact F1及其defined演示数。全部模型主/附seed均无双手认证交集，持续双手搬运证据未成立。

拼接最大单帧边界位置差为Deterministic0.026430m、Diffusion0.020887m；最大关节差为0.126347/0.135134rad。明显旋转边界出现在 `sub16_largebox_048` 的末窗起点108（起点列表[0,30,60,90,108]）：Deterministic29.715°、Diffusion99.652°。这段两模型均有velocities失败；记录说明实际生成存在边界异常，尚不能单凭该现象归因于某个预测、表示或融合环节。标准解码/融合在本轮固定，不为提高通过率另加平滑或投影。

### 5.4 扩散固定附表

| seed | processed | 教师拟合宏，m | task通过 | robot strict通过 | raw通过 | 每段最大穿透宏，mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 31/31 | 0.111347589839 | 28/31 | 0/31 | 0/31 | 145.797 |
| 17 | 31/31 | 0.111334925267 | 28/31 | 0/31 | 0/31 | 145.503 |
| 27 | 31/31 | 0.111343597021 | 28/31 | 0/31 | 0/31 | 145.456 |

三seed的拟合宏范围为0.111334925267–0.111347589839m，task失败段相同，penetration31/31、velocities5/31。两个附seed的微量valid双手覆盖不形成认证交集，认证覆盖仍0。所有seed完整保留，附表不参与checkpoint选择，也未选择较好seed替代主结果。

### 5.5 成本口径

下面是31段各行内记录之和；`row_total`含本行生成/检查/保存，不含只记一次的共享准备和阶段级登记。

| 方法/seed | 生成，s | 独立检查，s | row_total合计，s |
| --- | ---: | ---: | ---: |
| Teacher raw | 0.000384（读取已发布轨迹） | 40.767151 | 40.860261 |
| Deterministic/7 | 4.350639 | 56.708244 | 61.157526 |
| Diffusion/7 | 57.971353 | 56.431898 | 114.503393 |
| Diffusion/17 | 57.966028 | 56.461240 | 114.527053 |
| Diffusion/27 | 57.969601 | 56.322288 | 114.391780 |

共享条件构建0.293816s，checkpoint加载合计1.173594s；raw报告内部阶段总耗时512.907327s，预算驱动含入口、哈希和最终登记为 **524.651844s≤7,200s**。GPU边界同步，未另行预热；首个调用的冷启动也计入。Teacher这里是已有教师的读取/回放/检查成本，原始上游生成成本未发布，因此不能从这些数值宣称相对OmniRetarget的端到端加速。质量全拒时也不把较快生成称为可行方法加速。

## 6. val6 与 F 独立终局

[val6 详细报告](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/evaluation/detail_val6/attempt_001/detail_report.json)复用固定六段主seed×三方法的18份产品，expected=processed=n_ok=18、numerical_rejected=0。这里的ok表示产物可核验且测量完成，质量通过数是Teacher1/6、Deterministic0/6、Diffusion0/6。18份逐帧曲线、6份人体/源箱/目标箱参考、2份实际MuJoCo GIF共26个工件哈希均匹配，曲线覆盖末帧并保留首次失败位置。

代表类按固定val6的demo字典序选择，两个类别均存在：

| 代表回放 | 实际含义 | 帧覆盖 |
| --- | --- | --- |
| [pass_case：sub7_largebox_032](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/evaluation/detail_val6/attempt_001/summary__pass_case__sub7_largebox_032.gif) | 仅Teacher raw通过；Deterministic/Diffusion首败帧0/45 | T184，62张，包含末帧183 |
| [fail_case：sub10_largebox_100](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/evaluation/detail_val6/attempt_001/summary__fail_case__sub10_largebox_100.gif) | 三方法都失败；首败帧Teacher27、Deterministic0、Diffusion0 | T73，25张，包含末帧72 |

GIF按原30fps每3帧取样、保留尾帧；正常间隔100ms，末帧请求33ms按GIF10ms量化为30ms。pass_case的名称不表示两个模型通过。pass/fail GIF的SHA-256分别为 `124b351b374ef9b8f01f527ac4c26a7c54c68ea004c6361b3d8186cf077905dd` 和 `ba95ac24888cd8fe5bed846e7e413098559f0ad7f0fd9b95e312a63f30bc3369`。

逐帧工件与GIF在同一detail attempt目录。曲线为 `curves__{method}__{demo}.npz`，参考为 `reference__{demo}.npz`。固定六段的模型教师拟合及robot strict首次失败帧如下；帧号零基，Teacher fit为自参考0。除sub7的Teacher外，18条均未通过raw标准。

| demo | Teacher首败帧 | Deterministic fit，m | Deterministic首败帧 | Diffusion fit，m | Diffusion首败帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sub10_largebox_100 | 27 | 0.103179682 | 0 | 0.098651108 | 0 |
| sub12_largebox_008 | 146 | 0.137305197 | 0 | 0.102970894 | 0 |
| sub16_largebox_052 | 22 | 0.153796758 | 0 | 0.096092538 | 0 |
| sub3_largebox_018 | 0 | 0.097679012 | 2 | 0.109020758 | 2 |
| sub7_largebox_032 | null，通过 | 0.110571712 | 0 | 0.099306806 | 45 |
| sub8_largebox_001 | 26 | 0.089859783 | 0 | 0.093157612 | 0 |

F从完整训练loss/val/formal历史、checkpoint_index、配置/统计/权重重推训练和选择。31个measure unit×5产品，**155/155与raw一致且产品身份/内容核验通过**；重算93主+62附的计数、条件分母、宏平均和通过状态，无缺行、重复或外来键。质量拒绝保留为质量拒绝，没有变成缺证据。

独立checkpoint完整重放按身份规则固定在 `sub10_largebox_100`：两模型均best/step4000/seed7/T73，窗口起点[0,13]，覆盖末帧72。

| 模型 | 最大根数组绝对差 | 最大关节绝对差 | 重放核验 |
| --- | ---: | ---: | --- |
| Deterministic | 0 | 0 | 有限、形状、目标箱逐位绑定、source哈希稳定、末帧均通过 |
| Diffusion | 0 | 0 | 有限、形状、目标箱逐位绑定、source哈希稳定、末帧均通过 |

成功unit结果与F嵌入记录逐位一致。重放容差为冻结finalize实现中的 `REPLAY_TOL=1e-5`；成功JSON未单列该字段，须与实现快照同读。成功JSON也未保留此次独立重放的新retry metadata：原raw产品的一次尝试/无额外重试已核验，但不能写成独立重放retry记录逐项相等。本轮两模型输出差为0，不依赖放宽容差获得通过。

F各状态分列：

| 状态 | 实际裁定 |
| --- | --- |
| training | 两模型completed、4000/4000、evidence_complete=true、best_by_formal_macro |
| evaluation | 主93和附62完整，155个产品独立核验；val6 18条完整 |
| motion_baseline | complete：有限动作学习及质量诊断交付完成 |
| final_interaction_method | not_ready：模型主seed raw0/31，持续双手与主方案最终验收未证实 |
| next_action | attribution insufficient，auto_execute=false |
| 历史来源 | U0=incomplete；UR=no_quality_gain；保持原独立裁决 |

本轮没有追加冻结train24的三例Teacher/model同族缺陷与误差量级诊断，配对train例数=0；`label_principal_bottleneck_supported=false`。已知标签缺陷、模型新增穿透/脱离、预测或拼接/条件问题都是候选解释，目前无法认定主要瓶颈。F建议先定位配对失败族和预测误差，再另立下一项实验；没有自动扩数据、续训、教师处理或恢复修复路线。

## 7. 成本、终端核验与产物登记

六阶段均为attempt_001、exit0、done/complete:true，盘×索引×台账一一对应，成本共6行、无重复scope和running残行。最终驱动成本如下：

| 阶段 | measured_wall_s | 累计预算域 |
| --- | ---: | --- |
| R0/R1 register | 37.536342 | r0r1：37.536342/7200s |
| D Deterministic | 4196.265506 | d，两模型合计8657.971345/43200s |
| D Diffusion | 4461.705839 | 同上 |
| E raw-eval | 524.651844 | e：524.651844/7200s |
| val6 detail | 80.912767 | detail_f：570.839636/3600s |
| F finalize | 489.926869 | 同上 |

驱动实测合计9,790.999168s。外层命令进程墙钟分别为38.938517/4197.607187/4463.043253/526.040174/85.755327/494.724868s，包含驱动开始计时前的进程入口开销，未混作训练内部耗时。成本台账采用逐行measured值作为历史预算锚；四个域均未超限。

[执行回执](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/execution_receipt.json)记录每次实际argv、时刻、exit、日志与整链身份。其SHA-256为 `b754b72e2773ea04e86e92654772e9a496f2ffcb844ec9f82d0dd6e20e0218af`。terminal_verification的10项核验均true、注册问题及六stage_state问题列表均空，SHA-256为 `7ce78a5509cdd8816ff854e7c64d225a5931ae4f034546fa017a2bce46e973e2`。这两份为运行结束后新增的补充证据，未改写正式注册、报告、索引或台账。

| 关键产物（相对RUN） | SHA-256 |
| --- | --- |
| `freeze/execution_registration.json` | `5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e` |
| `evaluation/checkpoint_selection.json` | `a49f0ffea1d51dd103f0aa728794e7c891ae3c42fe59075519d68c452617b9af` |
| `evaluation/raw/attempt_001/raw_eval.json` | `e277cdb5100054ade489c7ddfb4718d2db638724c032df2c9313966a2dac1599` |
| `evaluation/detail_val6/attempt_001/detail_report.json` | `ba744ad3f4eeda6a0d826309738e11b6a22a017f2d9039a258906b840182521c` |
| `decisions/attempt_001/w1_de_final.json` | `7f1a9a595cf7964e99a929c83d3bcaac4aa3482a96e515228921b7b67a576737` |
| `costs.json` | `e79dcdaa4285708983137660aa73c3e82bf97a7d3d0f8e02d2fd84cad5e59475` |

所有大型轨迹、checkpoint、GIF及运行快照位于ignored derived/；仓库迁移需单独保存这些证据及只读资源，文档链接存在不表示产物随代码提交携带。执行方案、父方案、SOURCE和资源原字节保持。本轮790 passed+1 skipped对应运行实现；结束后的验证记录、索引和AGENTS进度登记仅改变文档。

## 8. 执行方案完成清单

- [x] I0–I4实现、真实参数解析与必要反例/完整回归通过。
- [x] R0事实重算通过、R1独占注册落盘。
- [x] 两模型各4k真实更新，完整损失/生成验证与best/last选择重推成立。
- [x] E的93主+62附键精确对账，155条保存轨迹与独立测量齐全。
- [x] val6的18条主产物复用、曲线与两类真实回放完成；未执行train归因三例如实登记。
- [x] F分别裁定训练、比较、有限动作基线和最终交互；后续建议自动执行=false。
- [x] U0/UR、发布标签质量风险及自研路线历史边界保持；未宣称持续搬运、承重、RL跟踪或OmniRetarget端到端加速。

## 9. 收口后独立审计修复轮（2026-10-14）

外部独立审计确认质量负结果成立（97 链接、九份关键哈希、155 产物哈希、31 份 Teacher 内容与冻结标签一致、12 条代表产物重测无差异、选模与四预算域成立），另发现 5 项核验缺口（3 P1+2 P2）已全部修复。**Teacher 2/31、两模型 0/31、`motion_baseline=complete`（有限动作基线）、`final_interaction_method=not_ready`、归因不足的结论不变；本轮纯核验侧加固，未重跑任何生产阶段。**

| 缺口 | 级别 | 修复 |
| --- | --- | --- |
| Teacher 产物未与冻结标签对账即复用自参考 FK | P1 | `recompute_w1_de_product` 对 teacher 行先做 `root/joints` 与冻结 clip 的**逐位对账**（`teacher_product_not_bitwise_frozen_label`），不一致即 `product_invalid`——平移 10cm 并更新哈希的临时副本不再能以自身 FK 洗白拟合 0 |
| 完成台账在预算终检前发布 | P1 | `finish_attempt` 重排：预算裁决**先于**任何终态行发布；盘/索引/台账各恰一次写入。真实子进程 SIGKILL 反例（剩余预算 0.1s、写盘延迟超限）下残行只能为 running 或诚实裁决的 incomplete，`budget_exceeded_after_registration` 撤销也并入同一发布序列 |
| F 只核 detail_f 历史 | P1 | `stage_finalize` 入口**重审全部四个预算域**（r0r1/d/e/detail_f）的盘×索引×台账三方对账与累计上限，计入本 attempt 之前；任一域历史未知（如未入账的 D 尝试）或超限即 `budget_domain_history_unknown/exceeded:<domain>`，closure 不可 complete；detail_f 自身不可用时也落盘 incomplete 记录而非裸退 |
| val6 工件只查五项字段 | P2 | 新增 curves/reference **完整必需字段契约**：curves 57 键按类型（有限/掩码/速度 NaN 定义掩码/区间漂移 NaN=参考掩码外/整型）核形状与掩码一致性（`curves_field_missing/unexpected/undefined_mask/...`）；reference 13 键结构核验+生产入口逐位绑定冻结 clip（`reference_not_frozen_clip:*`）。删 52 字段或删源箱/尺度/教师轨迹的哈希正确残件均拒 |
| 预算单测读到真实运行历史 | P2 | 预算/历史单测显式 monkeypatch `ireset.paths.PROJECT_ROOT` 至 tmp 沙箱（专项回归 151p/1f → 18p/0f）；生产跨运行累计语义不变 |

反例测试：新增真实子进程 SIGKILL 复现（击杀点在第一次台账写盘处，修复前发布 done/complete=true）；合成链测试扩展四项篡改反例（未入账 D 尝试、curves 削字段、reference 削字段、teacher 平移 10cm），closure 夹具 reference 按完整契约真实化。测试链 790→**791 passed+1 skipped**（1791.54s）。

**真实 RUN 复核（新契约下）**：31 份 Teacher 产物与冻结 clip 逐位相等（0 偏离）；18 条 curves 与 6 条 reference 工件完整字段/掩码/冻结绑定核验全部通过；四预算域历史可证且未超限（d 8657.97/43200、e 524.65/7200、detail_f 570.84/3600、r0r1 37.54/7200）；六 stage_state 无问题——权威 F 报告在新代码下结论不变。

本轮工程事实：**自参考捷径的前提是被验证对象与参照逐位相等**（自参考测量只能证明"产物=产物"，冻结标签对账必须先行）；**完成声明必须在最后一个预算检查点之后单次发布**（检查点与发布之间的窗口内被杀，事前 running 行是唯一在盘证人）；**收口审计的预算域必须全覆盖**（只审计即将花费的域会让其他域的未入账尝试静默漂过）；**哈希正确≠交付完整**（产物完整性契约须枚举必需字段集与定义掩码，而非抽检五个键）；**测试不得消费生产台账**（跨运行累计接口在测试中必须隔离 PROJECT_ROOT）；**近似键名字典陷阱**（契约核验首版把 expectations 键写成 `human_kps_world` 漏 t，被合成链测试逮住——完整性契约自身的键名必须与写入端断言对齐，本契约由 `_detail_curves` 的 set 相等断言钉死）。

### 9.1 第二轮复核修复（2026-10-14，1 P1+3 P2）

对首轮修复的二次审计发现 4 项残留缺口并全部修复；结论仍不变（Teacher 2/31、两模型 0/31、`not_ready`、归因不足），真实 31 Teacher/18 curves/6 reference 与四预算域复核通过。

| 残留缺口 | 级别 | 修复 |
| --- | --- | --- |
| 发布后终检仍有 SIGKILL 窗口 | P1 | 首轮只把检查移到发布前，但**登记本身耗时 0.2s** 仍可把剩余 0.1s 耗尽——发布 done/complete=true 后的递归撤销在 SIGKILL 下永不运行，残行为可消费完成声明（计费仅 0.00025s 过早采样）。现 `finish_attempt` 改为**两阶段发布**：先写不可消费的 pending 行（status=running、complete=false、实测成本），终检（覆盖全部登记耗时）通过后才原位发布终态；击杀落在任何窗口都只留下 running 行（消费侧=未知=停止）或检查已通过的终态。反例覆盖两个形态：发布前已超时（残行 incomplete）与**发布前未超时**（在终检写盘处击杀，残行 running、`stage_state` 报 attempt_running、域历史 None） |
| detail_f 未知历史回退崩溃 | P2 | 首轮回退把 `historical_s=None` 传入共享 `_finish`，`historical_s + elapsed` 抛 TypeError 且异常出口重复失败——报告虽落盘但索引/台账残 running。现 `_finish` 对未知历史记 `budget_consumed_cumulative_s=None` 并正常登记本次可测成本与 incomplete 终态；反例（decisions 下未入账 stray attempt）验证 finalize 落盘 terminal incomplete 且 `stage_state` 干净 |
| 参考箱尺度只查正数 | P2 | `source_box_similarity_scale` 从真实 0.7995248 改 1.0 并更新哈希仍过生产核验。现 reference 工件尺度与**冻结 derived npz 的 similarity_scale 逐位对账**（`reference_similarity_scale_not_frozen`，_load_clip 已哈希锁定 derived 源） |
| NaN 掩码允许 Inf 混入 | P2 | `nan_velocity/nan_seam/nan_mask` 用 `~isfinite` 对定义掩码，首帧合法 NaN 换 ±Inf 仍过。现**先拒 ±Inf**（`curves_field_infinite`）再用 `isnan` 对定义掩码——未定义位置是 NaN 且仅是 NaN，Inf 是坏测量不是未定义槽 |

测试链 791→**792 passed+1 skipped**。第二轮工程事实：**发布前检查不够——检查必须覆盖登记自身的耗时**（两阶段 pending→终态发布使"检查点之后"与"最后写入之后"合一，原子替换保证击杀落在窗口内只留 pending）；**未知历史的收口登记本身不能依赖已知历史**（回退路径的记账字段必须容忍 None）；**正数≠冻结值**（尺度这类从冻结源复制的标量同样需要逐位对账，与数组字段同一标准）；**`~isfinite` 不是 NaN 掩码**（Inf 与 NaN 同属非有限，未定义槽的合法形态必须显式限定为 NaN）。

### 9.2 第三轮复核修复（2026-10-14，1 P1：终态发布后的击杀窗口）

对第二轮修复（`abb8c02`）的复核确认三项 P2 成立，但 **P1 发布窗口仍未关闭**：终态（done/complete=true）写盘本身仍在最后一次截止检查之前——独立子进程复现（剩余预算 0.1s、终态 done 台账写盘延迟 0.2s、落盘后立即 SIGKILL）得到实际耗时 ~0.220s 而记录成本仅 0.003s、索引与台账均 done/complete=true、`stage_state` 无问题、域历史被接受；发布后的递归撤销在 SIGKILL 下永不运行。两阶段设计只把检查移到 pending 登记之后，其后的终态写盘仍可耗尽预算。

**修复：三阶段发布（pending → publishing → seal）。** `finish_attempt` 现按以下序列发布：

1. **Phase 1（pending 预登记）**：running/complete=false 行（不变）。
2. **Phase 2（终态内容发布，不可消费）**：终态内容（含 done/complete=true）以专用状态 **`publishing`** 写入索引与台账并携带实测成本——此时最后一次截止检查尚未覆盖这次写盘，任何消费腿都不得将其视为终态。
3. **终检（在终态发布之后）**：截止判定覆盖发布写盘自身的耗时；未超时但发布越界的完成声明被撤销为 `budget_exceeded_after_publication` 并以 publishing 重新发布（方向安全）。
4. **Phase 3（seal）**：终检通过后原位翻转为真实终态并刷新实测成本（覆盖发布写盘耗时）；seal 写盘是进程自身无法观测的最后尾巴。

消费侧拒收：`stage_state` 对 publishing 行报 `attempt_final_check_pending`；`_local_domain_history_cost_s` 对索引 publishing 行返回 None（未知=停止）；detail 的 `_load_indexed_report` 把 publishing 纳入显式非完成状态集合。真实 RUN 的历史行不携带 publishing 状态（legacy 兼容，与既有"缺声明字段的 legacy 行不构成矛盾"教义一致），复核不受影响。

**反例测试**：新增真实子进程击杀反例（**击杀点在终态写盘落盘之后**——预算宽裕起跑、终态写盘延迟 0.2s 越界、落盘后立即击杀）：盘上确有 done/complete=true 内容但状态为 publishing、实测成本 0.003s<0.1s（写盘自身耗时未入账），`stage_state` 报 `attempt_final_check_pending`、域历史 None——**未完成最后截止检查的状态不可消费**；新增进程内互补反例（终态发布写盘越界且未被击杀）：终检撤销完成声明、seal 为诚实 incomplete（`budget_exceeded_after_publication`），单行索引/台账、实测成本 ≥0.03s 且作为诚实预算历史可消费。原有两形态反例（发布前已超时/终检写盘处击杀）随三阶段语义更新断言（残行均 publishing 或 running，均不可消费）。

**真实 RUN 复核（新消费腿下）**：六 stage_state 无问题、四预算域历史可证且数值逐位不变（r0r1 37.54/7200、d 8657.97/43200、e 524.65/7200、detail_f 570.84/3600）——权威 F 报告结论不变（Teacher 2/31、两模型 0/31、`motion_baseline=complete`、`final_interaction_method=not_ready`、归因不足）。测试链 792→**794 passed+1 skipped**（1971.24s）。

本轮工程事实：**"检查点之后"与"最后一次写盘之后"无法由两次写盘合一**——终态内容必须先以显式未封口状态落盘、终检覆盖该写盘、通过后才原位封口；击杀落在封口前留下的是**自我声明未完成检查**的行（publishing），消费侧按显式非完成集合拒收（与 running 同一教义：不可捕获终止没有生产侧出口，事前行是唯一在盘证人）；seal 写盘自身是任何登记协议都无法事后观测的最后尾巴——这是不可捕获终止下预算记账的固有时钟边界，其成本量级由实测写盘延迟界定（本轮反例中 0.2s 延迟被终检捕获，残行不可消费）。另实测：子进程首次调用 `finish_attempt` 时函数内延迟导入（`ireset.data.a2_integrity`/`ireset.paths`）耗时 ~0.37s 且计入 t0 起算的预算——反例子进程必须先暖导入再采样 t0，否则入口检查即误判超时。
