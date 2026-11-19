# W1 固定六窗：§8.3 授权前复核

2026-11-19。**已到授权停止点。** 本轮补齐 [§6.3 运行时硬门](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md#63-继承已审计数值运行时的硬门)及[外层修复记录](omniretarget_w1_d6_joint_fit_outer.md)列出的新旧身份绑定，不重跑 [唯一 R6](omniretarget_w1_d6_joint_fit_r6.md)，不写正式授权，不执行真实 D6 或 F。

## 实现与验证

- D6/F 调已审计的 `configure_production_runtime()`，核对配置前、就绪和阶段终点的真实进程元数据。绑定 helper 原字节及 FK 缓冲区检查所依赖的已安装源码。D6/F 完整报告必须包含对应阶段的运行时观察；D6 缺证据即不启动 F，外层只读核验同样检查。
- 模型、conditioner 和 FK 在构造或转移返回后先检查实际浮点参数/缓冲和精确设备，再准入下一单元。`Trainer` 的新增可选检查不改变默认路径、初始化顺序或原生 AdamW。原 CPU 初始化和 CPU 参考架构明确检查 float32/CPU；真实生产转移、loss/FK 及重放目标检查 `cuda:0`/float32。F 两份 checkpoint 预检仍先于 fresh，逐模块 strict load 前对账源与目标键集/shape/dtype，返回后立即检查，再允许下一次加载或前向。无隐式精度修复、数值试跑或新增训练/FK 调用。
- 独立回归：R6/真实适配/双身份 **77 passed**（7.61 s），真实入口外层替身 **49 passed**（14.79 s），I6 CPU 夹具 **90 passed**（139.83 s），旧单窗原生前缀及修复定向 **19 passed**（30.82 s）。覆盖 dtype/device/掩码漂移、迟返停止、缺运行时证据零 F、旧包或草案不放行、当前身份与材料漂移。未运行全项目套件。
- 只读重新核对 421 个来源、449 个保护条目、RTX 3090/T6 运行时元数据；空闲磁盘约 22.47 GiB，要求 ≥7 GiB（仍须启动前复查）。未解码真实数组值、未加载真实 checkpoint、未构造真实模型或 FK、未编译真实场景或做 GPU 数值探针。不能据此保证显存、训练在时限内完成、达到均值门或两端逐位复现。

## 材料与授权范围

1. [最终审阅材料](../../derived/audit/w1_d6_joint_fit_lr_half_v1/preauthorization_v2/review_materials.json) SHA-256 `4067acde7b1620eed3ce12f7d2903bf88712810aca439c489abd0309d17e2d31`。此件记录旧、现两套身份，不把 R6 改判为新版准入。唯一 R6 以其原准入包和独立实际退出见证复验，仍为 **3.958656 s ≤ 登记 8.307731 s ≤ 120 s**。新版 88 文件 SHA-256 `64b5b079436eb66728237bcb0b275801731e040ba53fa5546d29890a2b879daf`；新契约 SHA-256 `fca7d037d423f0bddea42e49f4a123172a5d08314c18ba721e3b806e7086eb98`。两套材料均按其自身身份精确核验。
2. [未签授权草案](../../derived/audit/w1_d6_joint_fit_lr_half_v1/preauthorization_v2/authorization_draft.json) SHA-256 `185f062f079570b7915bc519e1d106d9a4dd554e158a2c257c4441f842b2c3dc`。新 schema `w1_d6_joint_fit_lr_half_v1/section_8_3_v2`，`authorized=false`、`template_only=true`、五项清单全 `null`；独立回读和只读门验证过，不能用它启动。旧未签草案与旧真实结果是历史，不作新版许可。
3. 所审阅的是给定配置内六个固定训练首窗能否同时达到四项平均门，不是对旧 T6 539 项缺口改判，不是六窗恒定 LR 对照或生产资格。仍仅 55/55 已绑定迁移日程、原 `train24` 根统计、共享确定性模型 1000 步和第 501 步前减半；真实调用矩阵 SHA-256 保持 `d4f79600f8167e45a198a00bc9406b5858f42bda93807dde8d889d8507868b42`。D6 `3600/120 s`、F `600/60 s`（硬上限/内含收口预留）、整链 ≤4200 s 且末 60 s 仅外层，受管峰值 ≤2 GiB；F 在 D6 实际关账后有条件接续，fresh 仅重放 initial/last。停止不自动重试，不运行 D1、DDIM、正式 val/test 或生产接入。

**下一步仅在用户复阅 [§8.3](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md#83-新授权必须明确) 五项并对上述审阅材料给出新的一次性明确授权后进行。** `includes` 必须精确为 `["D6","F"]`；当前 `authorization.json` 和真实实验根均不存在。
