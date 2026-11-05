# W1 保存输入的局部分项梯度验证

日期：2026-11-04。承接[失败只读归因](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_failure_attribution.md)，用户随后指令为“继续”。本轮执行保存预测上的局部 loss/FK 输出梯度对照和随机小型 attention 反例，不重放真实训练网络。

> 后续状态（2026-11-04）：本记录建议的 N1–N3 已在生产代码中实施，聚焦回归与保存输入输出 cotangent 验证通过；实现身份由 `0cc0622c…`（28 文件）变为 `57bbcae0…`（30 文件）。实际实现、测试与边界见[梯度数值稳定性修复验证记录](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_gradient_numerics_repair_verification.md)。本页保留为修复前的诊断快照。

**在保存的 diffusion u2/m0 输入上，已确认 trace-only 矩阵转四元数是原 5 个输出坐标超差的主因。** 原生产 cotangent 逐位复现；最大超差坐标的 FK 分项解释 99.99816% 的差异。只替换 FK 根旋转转换后，5 个坐标均回到原冻结门内。该结论限定于输出空间，不能据此把全部 173 项非 key bias 参数失败或 629 项 gap 判为已修复。

最小实现建议见[数值修复方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_gradient_numerics_repair_plan.md)。本轮未修改生产实现、容差或旧工件。

## 1. 工件、成本与边界

独立工件根为 [failure_attribution_local](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local)。[summary_001.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/summary_001.json) SHA 为 `0c246bd585068f16571ce4e1ff351167b048f17e8d90829c93d4de2df57ab86a`。

| 尝试 | 结果与保留事项 | 实测墙钟 |
| --- | --- | ---: |
| attempt_001 | 独立脚本缺少 `src` 导入路径；启动失败，梯度/FK/fixture 计数均 0；原脚本、scope 和 failure 原样保留 | 1.9528s |
| [attempt_002/result.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_002/result.json) | 完整微批 cotangent 分解、转换干预、42 旋转输入×两精度、4 attention fixture、32 保存状态算术投影；20/23 核对成立，3 项未满足见 §5 | 5.2229s |
| [attempt_003/result.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_003/result.json) | 归一化库适配的普通 backward、float64 有限差分、两帧干预及保护核验；6/6 通过 | 3.1593s |
| 合计 | 三次启动分别登记，不改写失败尝试 | **10.3349s** |

每次启动前写入独立 `scope.json`，墙钟上限分别为 360/360/180s；没有消耗旧 R0/P/I/F 的剩余预算。记录的局部计算合计为 **57 次 autograd 调用、3 次运动学链构造、7 次运动学 batch 调用/1444 个状态、4 个随机 attention fixture、12 次有限差分输出调用**。autograd 调用包含批量输出 VJP，不能换算成 57 个标量反传。

真实训练模型构造、checkpoint payload 读取、训练网络 forward/参数重放 VJP、MuJoCo 编译、优化器 step、训练更新均为 **0**。运动学链和随机 attention 的构造单独计数，不把本轮所有“模型构造”笼统写成 0。AdamW 影响仅以保存状态做不回写的 float64 公式投影。

原运行根仍为 **83 文件 / 7,835,408,541 bytes**，路径、大小、mtime 和非 NPZ 摘要未变；28 文件身份仍为 `0cc0622c86829b3e4bcbdefc1a3c1edb552ec02b31294dfd98bb70fd563205d8`。本轮不重复全文件哈希大型 NPZ；继承上一轮全量 SHA 与未变的大小/mtime，同时重新核对读取的预测和两侧 cotangent 数组字节 SHA。使用的 URDF、lock_record 和库函数源码另记 SHA。

## 2. 原生产路径复现与分项分解

固定输入为原 `diffusion/u2/m0` 的 `[4,60,38]` 保存预测及对应目标，保持 CUDA/float32、CPU/float64、20 线程、TF32/autocast 关闭。生产 `LossAssembler` 注入同一 G1 Torch FK 和无接触静态占位，`w_contact=0`，不编译箱体场景。完整微批形状和原 `/4` 梯度累积缩放均保持。

重新求得的完整生产 cotangent 与保存 P **逐位相等，最大差 0**。随后分别求生产六项梯度，按原冻结权重缩放；I 的五个非 FK 分项使用原冻结的 `independent_vjp` 重算。I FK 分项由“保存 I 总 cotangent − 五个非 FK 分项加权和”反解，**它是旧证据的仿射分解，不是一次新的 MuJoCo 独立 FK 测量**。

分项和与保存总差的最大闭合残差为 `1.12578e-10`，来自独立分项反传与总反传的 float32 加法顺序。原 5 个坐标的关键结果如下，索引从 0 开始：

| slot / 帧 / 通道 | 保存 P−I | FK 分项差 | geodesic 分项差 |
| --- | ---: | ---: | ---: |
| 1 / 0 / 4 | 2.13333193e-6 | 2.13329273e-6 | 3.68203e-11 |
| 1 / 0 / 5 | 1.95111967e-7 | 1.95109016e-7 | 5.48594e-11 |
| 1 / 0 / 7 | 3.47764941e-8 | 3.47731220e-8 | 4.60692e-12 |
| 1 / 0 / 8 | −2.18004081e-7 | −2.18016587e-7 | 1.16011e-11 |
| 1 / 6 / 4 | −1.49026647e-7 | −1.49007514e-7 | −2.33287e-11 |

这些坐标的主要差异来自 FK，geodesic 相对角路径不是该局部超差的主要来源。逐坐标 FK 差/总差约为 99.987%–100.006%；略大于 100% 表示其他分项有反向抵消，不是额外独立失败。

详细分项、各坐标和闭合残差见 [cotangent_decomposition.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_002/cotangent_decomposition.json)。

## 3. 只改变根旋转转换的干预

冻结关节局部 site 点，保留原位置、预测、目标、损失权重和分母；转换同时作用于预测与目标根旋转。旧转换的 root-only FK 梯度与完整生产 FK 梯度相等，排除了局部图切分本身消除失败的解释。

| 转换路径 | 检查范围 | 最大原冻结输出门比 | 超差数 |
| --- | --- | ---: | ---: |
| 原 trace-only | 原微批全部 rot6d 坐标 | 22.976338 | 5 |
| 导出的 `pytorch_kinematics.transforms.matrix_to_quaternion` | 同上 | 0.00628511 | 0 |
| 局部候选：先选最大分母再 sqrt | 同上 | 0.00576328 | 0 |
| 直接 R 旋转 site 点 | 同上 | 0.00457416 | 0 |
| 库函数 + 原入口所需归一化 | 保存两帧的原 5 个坐标 | **0.00140329** | **0/5** |

前四行是原完整微批的根旋转输出干预；最后一行是另一次仅 4 状态 FK（预测/目标各两帧）的验证，不能把其最大比值当作全微批最大比值。没有修改原容差 `2e-8 + 2e-4*abs(I)`。

CPU/float64、直接 R 的 URDF 路径与旧 I 反解 FK 根梯度最大差为 `1.52006e-10`。它没有达到诊断脚本额外设置的 `1e-10` 绝对目标，故原核对保留 false；但该残差用冻结输出门绝对项给出的统一门比上界仅 `0.00760030`。该交叉核对支持局部结论，不宣称两个 FK 后端逐位相同。

原两帧处的 trace 消差和小 `4*qw` 分母已由分项差及单变量干预共同确认是这 5 处输出超差的主因。**没有重新运行训练模型或参数 VJP，因此仍不声称已证明全部 173 项参数比较失败会消失。** micro2 的 `root_out.weight` 和其余 gap 仍保留待查。

## 4. 成熟库候选的边界验证

42 个输入包含单位旋转、绕三轴与一般轴的精确 π、π 两侧、分支交界、四分支交界和固定随机旋转；分别运行 CUDA/float32 与 CPU/float64。

原 trace-only 转换在精确 π 案例中旋转作用最大误差为 **2.0**，导数最大差约 **4e12**，虽然结果仍有限。它的既有“归一化后有限”说明不能代表值或梯度正确。

库候选和局部“先选分母”候选在这 42 输入的值与完整 `9×6` 旋转作用雅可比上均通过：float32 最大值差 `2.98023e-7`、雅可比差 `2.38419e-7`；float64 最大差约 `4.44e-16`。未观察到先前静态担忧的未选平方根梯度异常。

进一步对“库函数 + 归一化”运行**普通 backward**，两精度均无非有限梯度；float64 普通雅可比与独立直接 R 中心有限差分（步长 `1e-6`）最大差 **`1.26337e-10`**。dtype/device 保持，最大单位范数误差为 float32 的 `1.19209e-7`。详见 [normalized_library_edge_cases.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_003/normalized_library_edge_cases.json) 与 [local_arrays.npz](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_003/local_arrays.npz)。

因此优先采用已有成熟库接口；局部自写分支只作对照，不作为首选生产实现。四元数符号规范及其他调用方兼容性仍须在实际修复回归中核对。

## 5. 未满足的诊断假设如实保留

attempt_002 的 `verification_problems` 有 3 项，不写成“全通过”：

1. direct-R float64 与旧 I 分解差未达到额外 `1e-10` 绝对目标，实测 `1.52006e-10`；该目标不是原冻结契约门。
2. 预期在 float32 重现库函数未选 sqrt 梯度异常，但未观察到。
3. 同一个预期在 float64 也未观察到。

后两项是被实测否定的故障假设，不是实测到库函数失败。attempt_003 独立记录普通反向与归一化接口验证，6/6 核对通过；未改写 attempt_002 的条件、结果或摘要。初次启动失败 attempt_001 同样保留。

## 6. key bias 的最小稳定路径与状态约束

4 个固定随机 attention fixture 覆盖 self/cross attention、固定非空 mask、CUDA/float32 和 CPU/float64；形状为 B=2、L=7、S=7/11、d=240、6 heads，没有读取真实模型参数。

| 项目 | CUDA/float32 | CPU/float64 |
| --- | ---: | ---: |
| 原 key bias 梯度范数 | 3.78165e-7 / 3.97814e-7 | 9.83807e-16 / 7.85963e-16 |
| `bias*0` 进入同一线性投影 | 两例均为存在且逐位全零 | 两例均为存在且逐位全零 |
| 完全省略 bias | 梯度为 None | 梯度为 None |
| 原与结构零路径输出最大差 | 1.19209e-7 / 8.94070e-8 | ≤2.49800e-16 |
| 其他参数梯度最大绝对差 | ≤9.53674e-7 | ≤2.66454e-15 |

在 key 投影之后加入 `tanh` 的反例中，bias 梯度范数约 `0.82–0.99`，强制结构零导致输出改变约 `0.0020–0.0027`。因此该处理只适用于已经证明行平移不变的当前线性 key 投影，不能按 `.k.bias` 名称推广或挂统一清零 hook。非有限和不合法 mask 不获得豁免。

32 个保存的 key bias 参数步另做不回写的公式投影：u1 从空动量出发，若梯度为零，只有极小 weight decay，float32 写回量为 0；实际保存更新最大 **`3.33674e-5`**。u2 保留旧一阶/二阶状态，即使当步梯度为零，投影写回仍可达 **`1.75182e-5`**。这直接说明不能为了稳定当前梯度而清空旧 AdamW 状态，或把“存在的零梯度”替换成会跳过参数的 None。

以上是 float64 公式与 float32 写回的反事实算术，不是原生 AdamW step 的逐位重放。详见 [attention_fixtures.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_002/attention_fixtures.json) 和 [key_bias_state_projections.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution_local/attempt_002/key_bias_state_projections.json)。

## 7. 交付与未覆盖事项

本轮完成局部分项原因确认和最小修复方向选择；没有修改生产代码、注册新真实阶段或重冻结 Gate B/M5。参数级传播、micro2 `root_out.weight`、381 precision/223 sensitivity/25 bound gap 的修复后变化尚未验证，不能承诺清零。202 个失败 micro 行的展示状态问题仍列入修复方案。

原 `audit_incomplete`、329 failure / 629 gap、cohort 18 次更新 / 567.845409012s、`partial/null/false/not_ready/不接入` 和一次真实授权耗尽保持。后续实现会改变文件身份，必须生成新的适用证据；原授权不覆盖真实重跑。全项目测试套件本轮未重跑，局部证据与未来实现回归分别管理。
