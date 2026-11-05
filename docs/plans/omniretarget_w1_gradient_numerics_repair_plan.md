# W1 梯度数值稳定性最小修复方案

版本：v0.1，2026-11-04。依据[既有工件归因](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_failure_attribution.md)与[局部分项验证](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_local_gradient_verification.md)。本方案确定实现范围；生产修复、Gate B/M5 更新和新的真实 R0/P/I/F 均尚未执行。

## 1. 已有证据与成功标准

保存 diffusion u2/m0 的生产 cotangent 已逐位复现。5 个超差坐标主要来自近 π 绝对根旋转的 trace-only quaternion FK 链；已有库函数干预后全微批根旋转门比降至 `0.00628511`，归一化接口的两帧原 5 个坐标最大门比 `0.00140329`。单位/精确 π/两侧近 π/分支交界等 42 输入×两精度的值、反向和有限差分均已有局部证据。

attention 线性 key bias 数学上是结构零参数，但原 float32 消差残差已进入 AdamW 状态。随机 self/cross 小型 fixture 验证了保留参数、使用 `bias*0` 的前向可产生存在且精确为零的梯度；省略 bias 的 None 语义不同，不能替代。

修复成功首先要求这两个局部机制及接口回归通过，**不以 329 项 failure 或 629 项 gap 全清零作为已被证明的承诺**。旧 `audit_incomplete` 工件保留，原容差、可辨识性门与结果分类不得因修复而放宽。

## 2. N1：替换病态矩阵转四元数适配

修改 [tokens.py](/home/c112/codes/bill/ireset/src/ireset/model/tokens.py:174) 的单一 `_mat_to_quat_torch` 入口，优先调用已安装的 `pytorch_kinematics.transforms.matrix_to_quaternion`，保持 `mat_to_quat_torch` 公共别名、wxyz 排列、形状、dtype/device 和归一化。无需新增依赖，不复制四分支算法。

核对调用方是否依赖当前 `qw>=0` 的符号规范；需要时使用已有 `standardize_quaternion`，并以旋转作用和符号规范分别测试。单位/精确 π 时不能仅要求有限输出；必须同时检查旋转值和旋转作用梯度。已有局部归一化候选没有覆盖所有调用方，仍需实际接口回归。

新增回归覆盖单位旋转、三轴/一般轴精确 π、π 两侧、分支交界、随机正常输入及两帧保存 r6。float64 用独立直接 R/有限差分核验梯度；float32 用冻结输出门和严格有限性核验。记录库版本和函数源码 SHA。局部自写“先选分母”只保留为对照，不作为首选实现。

## 3. N2：在线性 attention 中显式实现结构零 key bias

修改 [Attn.forward](/home/c112/codes/bill/ireset/src/ireset/model/backbone.py:52) 的 key 投影，保留原 `nn.Linear` 参数与所有 state_dict 键，候选为：

```python
k = torch.nn.functional.linear(kv, self.k.weight, self.k.bias * 0.0)
```

该写法在前向去掉对 softmax 行平移无作用的偏置，同时在图中保留 bias 的存在零梯度。适用依据是当前“共享线性 key bias→QK→固定非空 mask→softmax”的数学结构；不按参数名字扫描后清零，不安装梯度 hook，不修改 Q/V bias，也不擦除非线性 key 投影的真实梯度。

保持 bias 参数、requires_grad、optimizer 参数组、checkpoint strict-load 键/dtype/shape、梯度 presence 和步数语义。不能删除 roster、把梯度改成 None、清空一/二阶状态或恢复旧 checkpoint 状态。当前零梯度仍由原 AdamW 公式消费既有状态和 weight decay；这与从零历史重新训练不同。

回归必须覆盖 self/cross、固定 mask、float32/64、零梯度存在性、其他参数梯度一致性、非线性反例、非有限输入不获豁免，以及包含已有非零动量的同输入 AdamW 公式/状态行为。小型合成 fixture 即可验证机制；不为这一步加载真实训练 checkpoint 或执行真实更新。

## 4. N3：修正失败行展示状态

修改 [I micro 参数行构造](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py:3601)，令 `status` 与 `comparison.ok`、`failure` 一致。当前 202 个已失败行仍写 `synthetic_verified`，正式失败汇总已正确拒绝；修复不得更改原 329/629 工件、把失败改判通过或影响成功行。增加一个明确失败和一个明确成功的反例，确认汇总和消费语义保持。

## 5. 实现、回归与身份更新顺序

1. 在旧保护清单和 28 文件身份上记录起点，保留旧运行根与原计划/授权/校准；实施 N1–N3，新增聚焦数值回归文件。
2. 先跑新的适配/attention/状态反例，再跑既有 `tests/test_fk.py`、`tests/test_w1_real_fk_audit.py`、`tests/test_w1_real_objective_audit.py`、`tests/test_w1_training_objective_audit.py` 及 provider/audit/migration 相关测试。由变更覆盖面选择集合，不以重复全量运行代替定位失败。
3. 用保存输入重做局部分项验证，比较完整输出 cotangent，而不只看 loss 标量；保留仍有的 micro2 和 gap 限制。局部验证不创建真实 R0/P/I/F。
4. 把新测试、新方案及实际 import 依赖纳入后续实现闭包；修复会改变原 `0cc0622c…` 身份。若进入新的真实执行准备，重新生成校准、普通/迁移合成接受与只读 admission/binding 材料，不沿用旧身份草案。
5. 新的真实模型重放、参数 VJP 或 R0/P/I/F 必须使用新的明确范围、预算与身份授权。原 M6 一次执行权已用完，本方案不创建或自动使用新的真实执行额度。

现有局部诊断合计 10.3349s、57 autograd 调用/1444 运动学状态，已在独立材料计费；不能作为下一阶段的隐含预算，也不回写原真实台账。后续保存输入局部验证应继续在各自启动前固定 scope、上限和计数，保留失败尝试，不自动扩大成训练网络重放。

## 6. 尚未解决的独立问题

micro2 `root_out.weight` 的输出 cotangent 已通过而参数门失败，不能仅凭 micro0 修复宣布解释完成。629 项 gap 中有 412 条未检测到同参数步梯度 failure；精度可辨识性与允许误差传播须继续逐项保留，不通过上调阈值、缩小参数集合或扩大噪声豁免来减少计数。

全目标覆盖、选模/DDIM/端到端质量、生产接入与新增训练均不在本方案实施范围。`partial/null/false/not_ready/不接入` 在相应独立证据出现前保持。
