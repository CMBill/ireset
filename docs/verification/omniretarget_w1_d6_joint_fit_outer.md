# W1 固定六窗：真实入口外层计时修复（纯合成验收）

2026-11-19。只修 `w1_d6_joint_fit_lr_half_v1` 的启动、实际退出和存档计费。不改[冻结方案 v0.2](../plans/omniretarget_w1_d6_joint_fit_diagnostic_plan.md#51-阶段与整链的实际截止)，不重跑[唯一 R6](omniretarget_w1_d6_joint_fit_r6.md)，不签发授权或启动真实 D6/F。

## 计费与消费边界

- 旧入口在最终输出之前判定完成。内存时钟对照中原先为 `4144.75 s ≤ 登记 4145 s`；仅把末次输出推迟 60 s，模拟返回到 `4204.75 s > 4200 s`，仍给出 `fit_pass` 和退出码 0。
- [v2 入口](../../scripts/run_w1_d6_joint_fit_real.py)由已经存活的监督父进程在主线程调用。父进程在导入入口和启动整链前取 `time.perf_counter()`，将同一 `start` 传给 `launch_authorized(auth_sha, start=start)`。授权和 ≥7 GiB 磁盘门先于独占创建及同步 `outer_launch.json`。父进程持有 collector 的实际 `wait`；collector 建独立会话，observer、D6/F worker 各自使用会话内的进程组。实际等待、进程清理和两路定额 collector 日志同步均被计费。
- 父进程复核阶段收据、内层存档和日志原字节。`outer_observation.json` 绑定同次 launch、实际 collector wait、内层存档和日志摘要；它始终是 `pending_actual_readback`。文件及目录同步、回读并核对后才采最后一次样，不延长原登记成本。完成路径不再进行成功输出或写盘，父调用方须直接保留返回的 `ExternalReturn`；普通 CLI 不提供启动后丢弃见证的入口。调用方将原见证交给只读 `consume(root, actual)` 独立核对，不能从 run 树重建见证、拿裸退出码或本地候选代替。若调用方另包进程、传输或写盘作为必需交付，不能用本协议的最后样本替代那层实际覆盖。
- 晚返回、丢失或错绑证据均为 `incomplete`，不发布最终拟合通过/不足。未运行重放为 `null`，实际观察到 `replay_mismatch` 为 `false`；已验证的局部事实不等于外层完成。失败在链硬截止内尽力保留实际 wait 和成本下界，不重写阶段收据、不重开目录。D6 的 `3600/120 s` 与 F 的 `600/60 s` 硬上限/预留不变，F 从 D6 实际关账接续，`T0+4140 s` 后仅供外层收口。受管上界仍为 2 GiB；写前计入两路新增日志和两个独立失败收据预留，按工件名册计算的设计峰值上界为 212,361,216 bytes（约 202.5 MiB，非实测峰值）。

## 验收结果

[定向测试](../../tests/test_w1_d6_real_outer.py)只在 `/tmp` 使用替身报告，实际运行父进程、collector、observer 和 worker 的进程、日志、台账和存档路径。数值阶段用假报告替换，不导入 Torch、NumPy、MuJoCo 或生产数值模块。

- 贯通正控：实际 collector exit 0、清理完成，从启动前到最后回读 **0.988215 s ≤ 登记 5.738499 s ≤ 4200 s**；持有实际见证的只读消费者一致，工件字节不变。单窗未过的完整夹具仍能经 F 和外层收口给出 `fit_insufficient`。
- 只延迟内层封口后的真实 collector 退出：仍 exit 0，实际 wait **5.927320 s > 登记 5.796177 s**，判 `incomplete`，无见证。另覆盖遗留进程组的回收、定额溢出、日志同步失败、证据缺失或改字节、串绑起点/PID/摘要、低磁盘和重入。假时钟覆盖回读或同步迟返阻断后继；末次收口延迟 60 s 即使留下 pending 收据，`4204.75 s` 也不发见证或完整裁决。
- 使用锁定环境分进程回归：新外层 **48 passed**（14.11 s）、原 R6/真实适配 **34 passed**（7.22 s）、I6 **90 passed**（139.91 s）；未跑全项目套件。I6 回归是独立 CPU 夹具，不是新的真实拟合或等价性证据。

## 身份与仍需解决的启动前条件

- 修复候选的 85 文件身份 SHA-256 为 `665a13b909a76e433da070a4f0af5d1e031a0caa5dc4268611ec6536dff78b46`，候选契约规范 JSON SHA-256 为 `3359e4902a97d74b7195aaa6fe231e18ad40b8c0ff2ec1adce9e6b1551db527c`。契约仅变更 `commands`、`reports`、`space`；`call_matrix(False)` 仍为 `d4f79600f8167e45a198a00bc9406b5858f42bda93807dde8d889d8507868b42`。六个首窗、55/55 迁移日程、原 `train24` 根统计、数值路径及 500/501 日程不变。421 个来源和 449 个冻结/历史保护条目均通过字节核对。
- 历史 R6 身份仍为 `b2afcb4ce26f2c436dc42b9bbc123fb83aeb4c8f2f066cb6c1d90849b6041607`（84 文件），原包、独立见证及未签草案保持原哈希。当前 `verify_archive` 在封存字节及返回链复验后，在新版契约相等校验处拒绝旧包。这**不撤销原 R6 对旧实现的准入，也不把它自动绑定到新实现**。建议另行复阅版本化的双身份材料：按旧身份只读复核历史封存及实际返回，独立冻结最终新实现/契约及变更清单，逐项对账保持相同的数据、运行时要求和调用矩阵；由新授权明确绑定两方。此轮未实现该版本桥接，未放宽等值门，也没有再次消费 R6。
- [主方案 §6.3](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md#63-继承已审计数值运行时的硬门)仍有单元顺序缺口：D6 的 [Trainer 构造适配](../../src/ireset/eval/w1_d6_joint_fit_real.py#L280)尚未逐构造在下一独立单元前验证实际参数/浮点缓冲 dtype/device；F 的 [fresh 重放](../../src/ireset/eval/w1_d6_joint_fit.py#L656)在两模块都构造或都 strict load 后才一起查。应先补逐单元硬门与纯合成反例，再复阅和冻结真实材料。数值等价、显存、真实训练时长与端点逐位可复现仍未证明。

**真实实验根和 `authorization.json` 均未创建。真实执行仍须最终材料上的新 §8.3 一次性明确授权，`includes` 精确为 `["D6","F"]`。**
