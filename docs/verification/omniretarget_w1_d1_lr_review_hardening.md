# W1 固定单窗 LR：真实实验后的工程复阅修复

2026-11-19。

本次只修复复阅提出的 2 项 P1、2 项 P2。已封存 [真实单窗结果](omniretarget_w1_d1_lr_schedule_diagnostic.md#8-真实-b-执行与最终比较2026-11-18) 保持 `fit_pass` / `single_window_pass`。没有改写方案、真实 A/B、R′、旧合成记录或任何授权；没有启动新的真实阶段或 D6。

## 修复与拒绝边界

- **AdamW LR（P1）**：日程驱动在 preflight 和调用原生更新的最后入口拒绝实例及全局 step 前/后钩子，并绑定实际 AdamW 实例及原生分派。最后入口才记录各组 LR；没有改 `Trainer`、模型、损失、日程或状态更新算术。F 重读新版证据，不把两端重放当成训练步的证明。
- **外部实际等待（P1）**：链的父采集器启动并回收整组，实际等待观察器退出。`chain_observer_actual_wait.json` 绑定启动 ID、PID、起点、封口、观察写盘文件与实际退出；`chain_wait_archive_observation.json` 绑定收据 SHA 和写盘同步后采样。等待和存档都必须在登记成本内，采集器还检查最后一份观察写盘；失败不发布可消费比较。新消费调用 `consume_comparison(attempt, collector_return_code=实际采集器退出码)`，仅传旧的 observer exit 0 不再有效。
- **前缀截止（P2）**：共同前缀收据同步返回后立即重查工作截止，再进入优化器状态读取和日程切换。慢写到达截止时零晚读、零后续更新。
- **2 GiB 工件硬上限（P2）**：B 的受管运行树在写入前核算字节，原文件与替换临时文件同时计入；JSON/NPZ 按序列化后的长度准入，checkpoint 用基于张量的预留加 Linux 文件大小硬限制，受管 stdout/stderr 各有定额。预留 1 MiB 收口空间，内含 4 KiB 失败证明保留量。没有空间就拒写；D1′ 不完整时 F′ 不做 fresh 重放。门约束的是此次运行的工件峰值，不是操作系统或依赖缓存的累计写入量。

## 验证与边界

定向测试使用隔离 CPU 夹具；包括切换步晚注册钩子、实际等待/两次写盘越线、收据缺失与串绑、前缀慢写、低上限下拒写 NPZ/checkpoint、替换峰值与日志限额。正例从真实子进程等待贯通至只读消费，两端逐位重放。复用的旧常量路径及 N0/N1 写盘行为不启用新工件门。

**结果：日程与复阅 73 passed（190.90s）；旧诊断/review16/外层 80 passed（245.17s）；准入与 N0/N1 91 passed（18.22s）。合计 244 项，未跑全项目套件。** 原 59 项继续通过；新增 14 项，包括实例/全局 × pre/post 四种晚注册钩子拒绝、慢前缀写盘、三个受限工件边界，以及外部等待/存档控制、原子替换与日志上限。原监督器贯通用例另外扩展了缺收据、串绑/超时收据及旧裸退出码拒绝。

执行记录在 `/tmp/opencode/w1_lr_review4.er8OV4/`：

- `final_acceptance/` 是本轮最终代码身份的全 CPU 合成 A→B→F 贯通。外部父进程从启动前至实际采集器退出为 **28.400814s ≤ 登记31.297936s ≤ 2400s**；观察器实际退出28.290772s，收据同步后采样28.309373s。collector exit0、F 两次 fresh probes 逐位重放，运行树1,283,770 bytes。与旧 `acceptance_002` 的全部7份 NPZ（5 probes＋输入＋教师）及4行 LR 历史逐字节相同。完整输出见 `final_acceptance_verified.json`。
- 最终合成实现清单为35文件，SHA `3a7faed3e64f84e7fbaafc7d8535f60a9c3bdfe347635bb9903b78781a295446`；相对旧33文件新增空间模块和原先未纳入闭包的 N0 封口写盘模块。原方案字节未变，新增定义只用于新身份/材料。
- `slow_observer_exit/` 通过真实子进程 `atexit` 注入退出清理延迟：观察器实际exit0，但父进程等待 **31.779366s > 登记31.459689s**。采集器返回3，落 `chain_capture_failure.json`，不发布成功等待收据。受控时钟用例另覆盖收据写盘或最后观察文件写盘越线，即使成功候选文件已存在也不能消费。
- 低上限测试在首个教师NPZ、首个probe和初始checkpoint分别于写前拒绝，零 optimizer 更新、零 F fresh probes；报告/台账/不可比较摘要仍能使用收口预留落盘。原子替换测试确认旧文件与 `.tmp` 同时占用、拒写保留旧字节；子进程日志超定额被截在保留槽并使流程失败。
- 本轮开始登记383个受保护的真实A/B、授权、准入和旧合成文件，修复后逐一重哈希不变；冻结源码下旧 `verify_saved_b.py` 的完整输出与旧 `verified_outcome.json` 逐字节一致，后者SHA仍为 `b40117137c02f1f65278749c12861262d9ca774350afb2ba995851909b255ac5`。原核验器同时复核其393历史文件和52旧合成文件保护清单。

开发期间的失败没有被删去：首轮低上限夹具未建输出根，暴露新空间扫描入口要求；在隔离 fixture 路径建立根后修复。另一轮期待停在checkpoint但预算实际上先拒了输入NPZ；测试改为从实际序列文件大小推导可用工作空间，使三个拒写位置各自被精确覆盖。相关日志保留在上述目录。

```bash
uv run --locked pytest tests/test_w1_d1_lr_schedule.py tests/test_w1_d1_lr_review_regressions.py -q
uv run --locked pytest tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_smallset_fit_review16.py tests/test_w1_i_outer_return.py -q
uv run --locked pytest tests/test_w1_smallset_fit_r_admission.py tests/test_w1_n0_coordinate_flags.py tests/test_w1_n1_synthetic_numerics.py -q
```

历史 `verify_saved_b.py` 绑定原实现身份，不能直接用本次新代码消费旧收据。可用 `git archive e96e0adea1573eade8c943233ae56f42ec6120ce src scripts tests configs` 解出原代码，在只读 Linux `bwrap` 中将这四个目录覆盖到原项目路径后按[原命令](omniretarget_w1_d1_lr_schedule_diagnostic.md#84-可复查的证据)复验。保留原路径是因为收据绑定了绝对目录；不需要切换、回退或覆盖工作区。下一次真实实验必须另立方案、重新冻结材料并获新授权；不是重开本次的固定根或已消费额度。
