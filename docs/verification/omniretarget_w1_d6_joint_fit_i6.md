# W1 固定六窗联合拟合：I6 纯合成实现与复阅

2026-11-19。范围是[六窗诊断方案 v0.2 §6 第 2–3 项](../plans/omniretarget_w1_d6_joint_fit_diagnostic_plan.md#6-实施顺序与放行前置)。本记录不改写方案的执行前快照，不是 R6 准入包，也不授予真实运行权限。

## I6 实现边界

- 新入口 `scripts/run_w1_d6_joint_fit.py` 只接受两种 CPU 合成夹具，并独占全新的 `/tmp` 输出根；旧授权、含 D1 的许可及真实参数均不接受。不运行 D1、不读取历史 A/B checkpoint 或数组值、不构造真实 G1 FK、不重建缓存。合成许可只对当前启动有效，`real_authorized=false`。
- D6 对从封存描述文件重建的六窗取证，复用现有 W1 窗构造、`Trainer` 的原生更新单元、整批 probe 和原四门度量。校验实际 collate 及设备批的八个数值键、窗位置和洗牌；conditioner 只收到五个合法条件键。前缀 JSONL/NPZ 在生成时登记且同步保存；独立读取校验后才发布切换收据，从写盘返回先查截止，再验证实际 AdamW 状态并原地修改各组 LR。训练算术与旧入口的默认行为不改。
- F 在读取数组值和 checkpoint 之前校验本次启动、许可、报告、台账、实际等待、工件字节和覆盖。通过后从保存量验证两端状态、完整曲线及全部窗位。两份 checkpoint 各只读一次，同一已验输入快照只转换一次，fresh 模块不再读输入或权重路径。仅 initial/last 各做一次整批重放；中间与教师 FK 状态关联、尖峰判门仍为 `not_evaluated`。
- 独立等待者记录 D6/F 的真实子进程退出与日志同步。F 起点就是 D6 关账证明同步并复核后的时钟；任何阶段不使用整链最后 60 秒。观察器、采集器和只读消费端分离：终态还要求实际采集器返回、最后存档的 SHA 与写后覆盖，并将最终工件和 D6 已封索引再次比对。阶段报告仍是 `pending_F` / `pending_outer`。
- `[D6,F]` 的合成调用矩阵和各类文件上限从同一代码定义进入许可与执行门。受管树按 2 GiB 峰值执行，启动前检查 7 GiB，计入日志槽、原子替换并存字节、临时 checkpoint 和收口预留。代码另列真实计划固定维度的推导矩阵（4000 micro、24,000 窗次、606 窗行、D6/F FK 8612/12），不把它当作真实 SOURCE/资产/CUDA 调用已经冻结或实测的证据。

## `7b28bf6` 初版贯通结果（历史，不作为修复后的验收）

初版两条独立 CLI 链使用同一 78 文件实现身份 `529da24008bf5495ed9f5a75550c854a308c83cf912046ed8ea4b7a1b5c61ba2`，与 `7b28bf6` 的相关文件逐项一致。环境为 CPython 3.12.13、锁定的 PyTorch `2.14.0+cu126`、NumPy 2.5.3；实际仅 CPU/float32，工作进程单线程，未用 CUDA。初版数据以其 `acceptance/` 为准，不消费 `cap_fixed/` 等更早版本的证据。

| 夹具 | 裁决 | 末次通过窗 | `complete_evidence` / `replay_passed` | `mean_fit_passed` |
| --- | --- | ---: | --- | --- |
| `fit-pass` | `fit_pass` | 6/6 | `true` / `true` | `true` |
| `one-window-fails` | `fit_insufficient` | 5/6 | `true` / `true` | `false` |

负控的根位置、根旋转、关节角和教师 FK 宏平均分别为 `0.004908921524042243 m`、`0.0016603710090031191 rad`、`0.0002864697473571745 rad` 和 `0.004947470068086916 m`，四项都在门内；但 `i6_fixture_05:0:60` 的根位置 RMSE 为 `0.022177299921612378 m`（门限的 2.218 倍），教师 FK MPJPE 为 `0.02206635848183772 m`（2.207 倍）。两处超门都属于同一窗，不会被其余五窗掩盖。

- **D6：**每条链各 6 次窗物化与编码，1 套共享模型/conditioner/AdamW，4 次原生更新、16 个整批 micro（96 窗次），5 次整批 probe（30 窗行）；conditioner/backbone 各 21 次，训练及取证的 FK 调用 68 次（32 次损失、36 次教师和 probe）。所有更新行的实际 LR 为 `[1e-4,1e-4,5e-5,5e-5]`；2 行历史、3 个整批 probe 与 18 窗行先完成前缀取证，切换在第 3 步 `begin_update` 前，切换前后 AdamW 状态摘要相同并与相邻更新衔接。另以控制器夹具验证 500/501/502 和 500/51/306 前缀边界，未做千步训练。
- **本地与 F：**每阶段保存量核验各计 16 个工件字节快照/哈希、7 个数组档案（468 个成员）、2 次 checkpoint 预检读、36 次规范解码和 36 次生产解码、30 条指标归约。F fresh 另计 1 套模块、4 次 strict load、1 次输入批转换、2 次整批 forward、12 次预测 FK 和 12 条归约。两端原始 `x0`、生产解码和预测 FK 逐位一致；D6/F 报告 `counts` 与 `call_matrix(True)` 逐项相等。

### 实际返回与存档覆盖

表中时间为秒；D6 与 F 各从自己的登记起点计，外层各项从 D6 起点计。D6 和 F 的 `actual_wait.json` 均为实际退出 0 且进程组已清理，两个 `post_sync.json` 经回读再采样；F 的 `t0` 与 D6 的 `closed_at` 精确相等。观察器正常退出并清理后，采集器写入、同步和复读 `archive.json`，随后写入、同步并复核 `archive_post_sync.json`；其实际退出和清理也在外层登记成本内。只读裁决凭外层见证消费，不把阶段或观察器 exit 0 当作通过。

| 实测或登记项 | `fit-pass` | `one-window-fails` |
| --- | ---: | ---: |
| D6 实际关账 ≤ 登记（硬上限 3600） | 39.405981 ≤ 43.336628 | 41.604854 ≤ 45.525489 |
| F 实际关账 ≤ 登记（有效硬上限 600） | 7.275712 ≤ 11.197869 | 7.275996 ≤ 11.259483 |
| 观察器实际 wait | 46.726707 | 48.931901 |
| `archive.json` 同步并复读后的采样 | 46.820466 | 49.021482 |
| 采集器实际 wait（含最后同步证明的写入及回读） | 46.864213 | 49.059291 |
| 采集器退出及清理 ≤ 登记（上限 4200） | 46.925072 ≤ 51.705487 | 49.129677 ≤ 53.904940 |

每条链的最终受管文件合计分别为 3,240,503 / 3,240,927 字节（这是收口后的大小，不是峰值实测）。受控时钟另验证：D6 在 `T0+3600s` 关账时，F 在 `T0+4080s` 禁止新工作，`T0+4140s` 后不接受封口；未用完名义阶段或整链额度不构成放行。

### 会话证据与回归

本机证据目录：`/tmp/opencode/w1_i6.o4HqEE/acceptance/`。`fit-pass.json` 及 `one-window-fails.json` 各自包含父进程实测的 `external_collector_wait`。在独立进程中以该见证调用 `consume`，两例返回与 CLI 裁决完全相等，未导入 Torch/NumPy，无补做数值调用或改写收据。只读复核摘要 `verified_acceptance.json` 的 SHA-256 为 `3aa882f3c26b30f2b9994904e35c98c6bc654bead6f04c88911623873e3d71cd`。两个运行根各自的 `archive_post_sync.json` 摘要分别为 `d99289859aafbc68ab4aa27ec6874e48ca1b0d0734c28b8501aab58c966f2d7e` / `96a2c5f481ae3ca2d8618f4ba4ab29f44f96cb02167eafd501e0e5a8f6c45083`（按正、负控顺序）。

- I6 **69 passed**（123.78 s，`i6_final.log`）：含两个完整子进程对照和载荷、顺序、学习率、前缀、优化器、F、计时、空间及独占反例。坏训练批不进入该 micro 的前向；坏前缀不进入第 501 步，收据同步越线后也不读优化器状态；F 硬门失败零装置构造，坏重放输入/末次权重零 fresh 构造和加载；预检后替换路径不改变 F 运行内快照，但终态拒收漂移字节。延迟返回、慢存档、裸退出码与串绑见证均不发布完整证据。
- 单窗 LR **73 passed**（202.83 s，`lr_regression.log`）；旧拟合与外层返回 **80 passed**（255.60 s，`fit_regression.log`）；准入/N0/N1 **91 passed**（18.48 s，`legacy_meta.log`）。复现保持三个独立 pytest 进程：早期混合执行 `legacy.log` 为 240 passed / 4 failed，四项均为旧准入夹具的 `runtime_identity:cpu_threads` 不匹配，在独立进程通过。未运行全项目套件。
- `protected.sha256` 的 **449** 项逐文件复核零漂移，含 v0.2 计划原文（`c6e9150bdbba95d443b5d2150cc466fb5adcca377f421a1d324d8f9855370fd2`）、已消费的 A/B 工件、历史 R 来源锚及迁移缓存。拟用的两处真实 `w1_d6_joint_fit_lr_half_v1` 根均不存在。

## 六项复阅发现与修复

`7b28bf6` 的 69 项测试复跑全过，但隔离反例复现了 4 项 P1 和 2 项 P2，初版不通过复阅。本轮修复和复验如下：

1. **实际微批槽序。** 设备批逐槽对照生成清单，同时必须保持 collate 的洗牌槽序。微批记录和消费次数在已验批实际交给原生 Trainer 时入账。六窗整体轮换反例既不启动该 micro 的前向，也不记窗次；合法批与实测原生入口的槽序和载荷记录相同。
2. **第 501 步前缀。** 放行前核对 probe 目录的完整文件集合和类型，再比对生成时登记的摘要。52 份 probe、同名非普通文件、目录读取迟返均拒；后续前缀读取、状态读取、切 LR、micro 和更新均未放行。
3. **生产解码运行时。** D6 本地与 F 都逐次检查保存量生产解码前后的精度及工作截止。CPU autocast 漂移、慢运行时检查和慢解码都不能进入后继调用；F fresh 中单窗 FK 留下的漂移也在下一窗解码前拒绝。
4. **交接与收口计时。** 交接字段逐次读取、哈希、解析；工作截止从编排器传入 worker 并对账台账。阶段与外层的报告、收据、日志和存档按各自截止守卫。慢 worker 元数据读越过工作截止，无后继哈希或报告读取；慢阶段收据在 `T0+4140.1s` 返回，无收据回读或 `post_sync` 写入。外层慢写同样拒绝后继回读。
5. **重放三态。** fresh 形状、dtype、逐位值或派生指标实测不符，记 `ReplayMismatch` / `local_replay_passed=false`，不运行另一端；未重放保持 `null`。本地重放通过而外层未覆盖不发布最终通过。
6. **失败等待事实。** 真实 D6 子进程退出 7 的回归保留 PID、退出码、实际 wait、清理时点和已知成本下界。独立观测字段 `failed_stage_wait.coverage_proven=false`，无阶段通过收据、没有 F，消费端为 `incomplete`。

## 修复后 I6 复阅通过

本轮仅对 I6 纯 CPU 机制给出通过。修复后实现身份（78 个文件）为 `77b0097bbb26f52fc399e4b221ce58b9878d51a07ade3a1d6b283d5a8a11588c`。两条独立 CLI 链从新 `/tmp` 根启动；用各自 `external_collector_wait` 在新进程只读消费，结果逐字段一致，不导入 Torch/NumPy 或追加数值调用，运行根内文件均未改变。

- 正控为 `fit_pass`（6/6），负控为 `fit_insufficient`（5/6），负控四项宏平均仍过门；两条链均 `complete_evidence=true`、`replay_passed=true`。每条 4 更新、16 micro/96 窗次、5 批 probe/30 窗行，D6 与 F 实际计数逐项对上矩阵。LR 仍为 `[1e-4,1e-4,5e-5,5e-5]`，切换前后 AdamW 状态摘要连续。F 从已验快照严格加载两端各自的状态（模型/conditioner 各载两次），重用一份输入，只重放初末端点。每条链的 5 份 probe NPZ、输入/教师 NPZ、训练历史及 probe 日志共 9 份与初版对应夹具逐字节一致。
- 实测 F 起点精确等于 D6 关账时点。下表分别以阶段起点和整链起点计秒，均为实际耗时 ≤ 登记成本；实际等待、清理、写后回读及末次存档摘要都经复核。

| 覆盖项 | `fit-pass` | `one-window-fails` |
| --- | ---: | ---: |
| D6 关账 | 39.565861 ≤ 43.584112 | 39.924959 ≤ 43.958468 |
| F 关账 | 6.779574 ≤ 10.862394 | 6.775379 ≤ 10.874563 |
| 末次存档收据写回后的采集器退出及清理 | 46.580580 ≤ 51.375564 | 46.945159 ≤ 51.723051 |

本轮 CLI 见证及独立只读摘要在 `/tmp/opencode/w1-i6-fix-acceptance.ev5AeO/`；`verified_acceptance.json` SHA-256 为 `ad2961320172eeb013439eb6aa3f532bae0a781c1f304b5eb04b5fad082ed61f`。正、负控各 52 文件，最终受管大小为 3,240,837 / 3,241,250 字节，不当作峰值实测。

新 I6 套件 **86 passed**（142.04 s，`/tmp/opencode/w1-i6-fix-final-tests.dxpJJl/pytest.log`）。隔离的旧单窗 LR、旧拟合/外层、准入/N0/N1 分别 **73 / 80 / 91 passed**（`/tmp/opencode/w1-i6-fix-regressions.bFix3y/`）；未运行全项目套件。冻结计划及历史 **449** 项 SHA 一致，两处真实运行根仍不存在。

## 复现命令

```bash
export PYTHONDONTWRITEBYTECODE=1
test_out=$(mktemp -d /tmp/opencode/w1-i6-test.XXXXXX)
uv run --locked --offline pytest -q --basetemp "$test_out/i6" tests/test_w1_d6_joint_fit.py
uv run --locked --offline pytest -q --basetemp "$test_out/lr" \
  tests/test_w1_d1_lr_schedule.py tests/test_w1_d1_lr_review_regressions.py
uv run --locked --offline pytest -q --basetemp "$test_out/fit" \
  tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_smallset_fit_review16.py tests/test_w1_i_outer_return.py
uv run --locked --offline pytest -q --basetemp "$test_out/meta" \
  tests/test_w1_smallset_fit_r_admission.py \
  tests/test_w1_n0_coordinate_flags.py tests/test_w1_n1_synthetic_numerics.py
```

完整夹具可通过已安装的锁定环境直接启动轻量父进程，使其在两个阶段的 `uv` 启动前采样时钟。用新的空父目录；输出根本身必须尚不存在。

```bash
out=$(mktemp -d /tmp/opencode/w1-i6.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_w1_d6_joint_fit.py \
  --synthetic fit-pass --output-root "$out/fit-pass" > "$out/fit-pass.json"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_w1_d6_joint_fit.py \
  --synthetic one-window-fails --output-root "$out/one-window-fails" > "$out/one-window-fails.json"
```

只读复核现有工件时，以原输出根及随实际等待保存的 CLI 见证为输入，不重新启动阶段：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/scripts:$PWD/src" .venv/bin/python - "$out" <<'PY'
import json
import sys
from pathlib import Path
import run_w1_d6_joint_fit as cli

parent = Path(sys.argv[1])
for case, expected in [('fit-pass', 'fit_pass'), ('one-window-fails', 'fit_insufficient')]:
    recorded = json.loads((parent / f'{case}.json').read_bytes())
    witness = recorded.pop('external_collector_wait')
    witness['wait'] = cli.Wait(**witness['wait'])
    verified = cli.consume(parent / case, cli.CollectorReturn(**witness))
    assert verified == recorded and verified['verdict'] == expected and verified['complete_evidence']
    print(case, verified['verdict'], verified['final']['n_passing_windows'])
PY
```

## 后续边界

I6 只证明机制，不证明真实六窗可拟合、生产 CUDA 可逐位重放或实测能在预算内完成。本次到 I6 止；下一步须另行**明确安排一次仅元数据 R6**。真实 SOURCE/缓存/资产适配与调用、配置、运行时及容量等材料仍须核对冻结，R6 通过后才交未签草案。真实 `includes=["D6","F"]` 仍须 §8.3 新明确一次性授权；旧 A/B、T6 和已消费额度不变。
