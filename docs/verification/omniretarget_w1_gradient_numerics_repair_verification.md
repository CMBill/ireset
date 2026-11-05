# W1 梯度数值稳定性修复实现与验证记录

日期：2026-11-04。状态：**N1–N3 已实现；聚焦数值回归与保存输入局部分项验证通过；Gate B/M5 未重新生成，未启动新的真实 R0/P/I/F，M6 一次授权保持耗尽**。本记录只描述代码、测试和本地只读验证，不构成新的真实执行授权。原 M6 的 329 项 failure / 629 项 gap、`audit_incomplete`、cohort 18 次更新 / 567.845409012s 和 `partial/null/false/not_ready/不接入` 全部保留；本修复不宣称这些计数已清零。

## 1. 起点、边界与实现身份

起点为 28 文件实现身份 `0cc0622c86829b3e4bcbdefc1a3c1edb552ec02b31294dfd98bb70fd563205d8`（M6 授权与最终 Gate B/M5 所绑定字节）。本轮按[修复方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_gradient_numerics_repair_plan.md)实施 N1–N3，并把新方案与新测试纳入实现闭包，闭包变为 **30 文件**，新身份为：

`57bbcae02af67a39d6e2f96147371b16bdf9d7f076b3d1182d1aabe9fc10d933`

原真实运行根保持 **83 文件 / 7,835,408,541 bytes**，本轮只读取；未加载真实 checkpoint payload、未构造训练网络/优化器、未执行真实 forward/backward/FK/AdamW/优化器 step/训练更新、未编译 MuJoCo 场景、未创建真实 R0/P/I/F 或消耗旧阶段预算。Gate B/M5 未重新生成；旧材料绑定的是旧 28 文件身份，不能作为新身份的真实执行依据。

## 2. 实现

### 2.1 N1：稳定矩阵转四元数适配

[tokens.py](/home/c112/codes/bill/ireset/src/ireset/model/tokens.py) 的单一 `_mat_to_quat_torch` 入口改为：

1. 调用已安装的 `pytorch_kinematics.transforms.matrix_to_quaternion`（最佳条件分支，而不是固定 trace-only 分支）；
2. 保持单位范数归一化；
3. 用 `standardize_quaternion` 保持历史公共 `qw >= 0` 符号规范；
4. 保留 `mat_to_quat_torch` 公共别名、wxyz 排列、形状、dtype/device。

依赖冻结：`pytorch-kinematics==0.10.0`；`pytorch_kinematics/transforms/rotation_conversions.py` SHA-256 为 `fa4fae88af30ee99e524b1f0561e5eda7f236a51efc2d7320372f208f3a749ac`，`matrix_to_quaternion` 源码 SHA-256 为 `15848ef8243dbe3addd3c2bbc0378b37d06bcafcb0ec74558d74270cc214c6b3`（由聚焦测试固定）。原 trace-only 实现只保留在测试中作为负对照；其在精确/近 π 输入上的旋转值误差可超过 1.0，单有“归一化后有限”不能证明旋转或梯度正确。

### 2.2 N2：线性 attention 的结构零 key bias

[backbone.py](/home/c112/codes/bill/ireset/src/ireset/model/backbone.py) 的 `Attn.forward` 把 key 投影改为：

```python
key_bias = None if self.k.bias is None else self.k.bias * 0.0
k = F.linear(kv, self.k.weight, key_bias)
```

该处理只适用于当前“共享线性 key bias → QK → 固定 mask → softmax”的数学结构：bias 对所有 key 施加同一行平移，softmax 不变；`bias * 0` 让参数仍在计算图中且梯度存在、逐位为零。Q/V bias、输出投影、conditioner、非线性 key 路径均未改动；未删除参数、未把梯度改成 None、未清空或恢复 AdamW 一/二阶状态。严格 checkpoint 键集、dtype/shape、`requires_grad` 与优化器参数组保持不变。

### 2.3 N3：失败 micro 参数行状态

[w1_real_training_update_audit.py](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py) 的 I micro 参数行不再在比较前写死 `synthetic_verified`。现在 `status` 与 `comparison.ok` 一致：通过为 `synthetic_verified`，失败为 `mismatch`，`failure="micro_parameter"`；汇总、`numeric_passed` 和消费语义不变。原 M6 工件与新代码独立，未被改写。

## 3. 聚焦回归

新增 [test_w1_gradient_numerics.py](/home/c112/codes/bill/ireset/tests/test_w1_gradient_numerics.py)，覆盖：

- 42 个旋转输入：单位、三轴/一般轴精确 π、π 两侧 `±1e-3/±1e-6`、分支交界、固定随机旋转；float32/float64；
- 精确 π/近 π 的旋转作用值、单位范数、`qw >= 0` 符号规范和严格有限性；
- float64 完整旋转作用雅可比对独立直接 R 中心有限差分，最大差 `< 1e-8`；
- 两帧保存的 diffusion u2/m0 r6 输入；
- 依赖版本/源码 SHA 与公共别名；
- self/cross、固定非空 mask、float32/64 的前向等价、零 bias 梯度存在、其他参数梯度一致；
- 非线性 key 反例、非有限输入不获豁免、checkpoint strict-load 与参数名册；
- 带已有非零一/二阶动量的 AdamW 公式投影：零梯度仍产生非零参数更新，状态按 `beta1/beta2` 衰减，step 递增。

结果：**19 passed（3.36s）**。

## 4. 相关回归

| 集合 | 结果 |
| --- | --- |
| 轻量相关集：`test_fk`、`test_generation`、`test_w1_de_numerics`、`test_label_quality`、`test_w1_prediction_consistency`、`test_omniretarget_w1_de`、新聚焦文件 | **184 passed（340.46s）** |
| 计划指定集：`test_w1_real_fk_audit`、`test_w1_real_objective_audit`、`test_w1_training_objective_audit`、provider/audit/migration + 新聚焦文件 | 首轮 **323 passed / 2 failed**；修复两项测试证据后 **344 passed（447.75s）** |

首轮两项失败均不是生产契约失败：

1. `test_calibration_chain_applies_the_4x_rule_to_every_gate` 原先把“逐门统计 > 0.1”作为固定数值代理。N1 改变生产解码后，tiny chain 的逐门最大值变为 theta `0.03249`、state `0.08110`、delta `0.21502`、same `0.21615`，4× 仍全部 ≤1。测试改为比较“逐门统计”与“整行 `classification=passed` 过滤”的最大值，保留原意图并避免把某一数值快照写死；生产容差与可辨识性门未改。
2. `test_legacy_head_matches_the_frozen_real_prefix` 在 M6 执行后读取的是 append-only 的当前 `index/ledger`，整文件 SHA 已因 seq 3–6 追加而变化。测试改为从冻结 snapshot 的 `source/index.json`、`source/ledger.json`、`source/cohort.json` 重建，冻结 legacy head `b499a6ce…` 与常量均未改。

## 5. 保存输入局部分项验证

独立尝试 [failure_attribution_local/attempt_004](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_004)（启动前固定 scope：120s、1 次运动学链、2 次 batch、960 状态、8 次 autograd；训练模型/checkpoint/优化器/更新/MuJoCo 编译上限均为 0）：

| 项目 | 结果 |
| --- | --- |
| 墙钟 | **4.3362s** |
| 计数 | autograd 2；运动学链 1；batch 2；状态 480；训练模型/checkpoint/优化器/更新/MuJoCo 编译 0 |
| 保存输入绑定 | prediction `2aafba93…`、saved P `f4b33c23…`、saved I `c56e0bf0…`、fk elements `199f9415…`，均与 P report / I row 的登记 SHA 相等 |
| 修复前 P vs I | max ratio **22.97634**，5 个超差坐标，max abs `2.13333e-6` |
| 修复后 P vs I | max ratio **0.0154715**，**0 个超差坐标**，max abs `4.95168e-10` |

原 5 个超差坐标（slot/frame/channel）修复前后的冻结输出门比：

| 坐标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 1 / 0 / 4 | 22.97634 | 0.00148165 |
| 1 / 0 / 5 | 1.15037 | 0.00068571 |
| 1 / 0 / 7 | 1.59416 | 0.00121570 |
| 1 / 0 / 8 | 5.85240 | 0.00122131 |
| 1 / 6 / 4 | 1.34628 | 0.00065294 |

修复后 P 相对旧 P 的最大差为 `2.13319e-6`（5 个坐标），说明改变确实发生在原失败位置；完整输出 cotangent 与保存 I 的 0 个超差是本次局部验证的主要结果，而不是只看 loss 标量。

工件：`result.json` `4c00262f1c0b0a0e65375000322ffd7357eef1842aeec6e942e62d3e4a180527`、`local_arrays.npz` `3b00d09321089b4a481b18a93e84152d4376a2ce17365d18b1ed4fe01cc76d56`、`artifact_manifest.json` `fa4df7371a737d61dda521aaa7554785f89e9555cbb92b9ad913b9e9ef96d25e`、`scope.json` `a650cfb1b86b40f7cb677236114ef07eeb012f60ba0070d7efab333ebcc60d96`。

本验证的限制：I 数组是原真实运行保存值，未重新执行独立 MuJoCo FK；只验证输出 cotangent，不重放参数 VJP、`micro2 root_out.weight`、381 precision / 223 sensitivity / 25 bound gap；未构造训练网络或 checkpoint。因此不能把“输出门 0 超差”外推为“329 failure / 629 gap 已清零”。

## 6. 未改变或未执行的事项

- 原 M6 真实工件、授权、校准、Gate B/M5、运行根和旧台账均未改写；原真实根仍为 **83 文件 / 7,835,408,541 bytes**。
- Gate B/M5 **未重新生成**。旧材料绑定旧 28 文件身份，不能用于新身份的真实执行；若未来进入新的真实执行准备，必须重新生成校准、普通/迁移合成接受、只读 admission、snapshot/spec 和授权草案，并取得绑定新字节的新明确授权。
- 未创建新的真实 cohort、预算、R0/P/I/F 或注册；M6 一次授权仍耗尽。
- `audit_incomplete`、329 failure / 629 gap、cohort 18 次更新 / 567.845409012s、`partial/null/false/not_ready/不接入` 与旧质量结论保持。
- 全项目测试套件未重跑；本轮运行的是按变更覆盖面选择的集合（见 §4），不把相关集合的通过冒充全量结果。
