# M0 验证记录

日期:2026-09-11。本文件记录 M0 里程碑中已执行的验证、发现的约定和遗留项,与 `derived/robots/lock_record.json`、`derived/data/audit/m0_pairing_report.json` 配套(均为可重新生成的派生产物)。

## 1. 依赖锁定与运行验证

通过 `uv add` 维护,`uv.lock` 固定;PyTorch CUDA wheel 来源为 `https://download.pytorch.org/whl/cu126`(见 `pyproject.toml` 的 `[[tool.uv.index]]`)。

| 组件 | 版本 | 验证内容 |
|---|---|---|
| torch | 2.14.0+cu126 | RTX 3090 前向/反向 float32/float64;float64 梯度与 CPU 参考相对误差 0,float32 1.7e-7;`cuda.is_available()`、runtime 12.6、cudnn 9.10.2 |
| mujoco | 3.13.0 | 编译 G1 三个 MJCF 变体与 URDF;float64 FK + `mj_jacBody` 有限;`mj_geomDistance` 距离查询;EGL 离屏渲染 |
| numpy / scipy / trimesh / pillow | 2.5.3 / 1.18.1 / 5.1.0 / 12.3.0 | Slerp、Rotation、OBJ 加载与表面采样、PNG 输出 |
| pytest | 9.1.1 | 5 项约定测试全部通过(`uv run --locked pytest tests/ -q`) |

复验命令:`uv run --locked python scripts/smoke_deps.py`。

注意事项:trimesh 5.x `Mesh.sample` 只返回采样点;`mj_geomDistance` 必须传入 `fromto` 缓冲;无显示环境下用 `MUJOCO_GL=egl` 渲染(osmesa 库缺失;EGL 在进程退出时报无害的清理告警)。

## 2. 锁定的派生 G1 资产

`scripts/build_locked_assets.py` 从只读源模型生成(源文件未改动),输出与哈希记录在 `derived/robots/lock_record.json`:

| 派生文件 | 来源 | 内容 |
|---|---|---|
| `derived/robots/g1_29dof_locked.xml` | `g1_29dof.xml` | 29 关节 + freejoint(nq=36),新增 31 个命名 site |
| `derived/robots/g1_29dof_locked_w_largebox.xml` | `g1_29dof_w_largebox.xml` | 同上 + largebox freejoint(nq=43),与教师 NPZ 维度一致 |
| `derived/robots/g1_29dof_locked.urdf` | `g1_29dof.urdf` | 仅修正 `left_pinky_link` 的 `<collsion>` 拼写;MuJoCo geom 数 48→49,碰撞球出现 |

site 约定(定义于 `src/ireset/embodied/g1_spec.py`,随机姿态下 site 与 FK body 原点最大误差 0):

- `kpt_*`:15 个解剖关键点,body 原点,对应上游 holosoma `("smplh","g1")` 映射的目标 body(pelvis_contour、hip_pitch×2、knee×2、shoulder_roll×2、elbow×2、ankle_intermediate×2、ankle_roll_sphere_5×2、rubber_hand×2)。
- `sole_{l,r}_{1..5}`:每脚 5 个碰撞球(r=5mm)球心;1-4 为上游 FOOT_STICKING 集合,5 为 toe。**球心不是接触面**,锁定锚点时需沿法向减半径。
- `hand_{l,r}_{palm|thumb|pinky}`:palm 在橡胶手 mesh 包围盒中心(左手 body 系 (0.0659,-0.0146,0.0102),右手 y 镜像);thumb/pinky 在指尖 body 原点(碰撞球 r=1mm)。

关节顺序(MJCF hinge 顺序,即 G1Spec.JOINT_NAMES):左腿 6 → 右腿 6 → 腰 3 → 左臂 7 → 右臂 7。项目内部位姿统一 `[x,y,z,qw,qx,qy,qz]`,与 MuJoCo freejoint 一致。

## 3. 数据审计结论(M0 配对验收)

脚本 `scripts/audit_data.py`,样本 71 个 NPZ(6 个受试者 × 前 4 个演示的 original + 首个 rot/trans 变体),报告写入 `derived/data/audit/m0_pairing_report.json`;渲染证据在 `derived/data/audit/renders/`(`scripts/render_teacher.py`)。

### 3.1 库存

- 教师 NPZ 共 1,826:`_original` 312 + `_rot_*` 590 + `_trans_*` 924,全部 fps=30。
- 312 个独立演示分布:sub10=79、sub12=57、sub16=53、sub3=34、sub7=46、sub8=43。
- 本机 largebox PT 329 个,其中 17 个无对应教师(清单见报告 `pt_without_teacher`)。
- OMOMO PT 张量为 `[T, 591]` float32;教师 NPZ `qpos=[T,43]`、`human_joints=[T,52,3]`,T 与 PT 一致(抽查全部相等)。

### 3.2 已确认的数值约定

- **教师 NPZ qpos 布局为 `[qw,qx,qy,qz,x,y,z] | 29 joints | [qw,qx,qy,qz,x,y,z]`**(根在前四元数、箱体在 36:43):两种布局假设下单位四元数检验唯一支持该解释(根 z≈0.78m 骨盆高度、箱体 z≈0.19m 半箱高)。**与项目内部约定相反,接入时必须显式转换**(`scripts/audit_data.py::convert_scene_qpos`)。
- PT 物体列 `[318:325] = [x,y,z,qx,qy,qz,qw]`,上游重排约定数值验证成立。
- PT 人体列 `[162:318] → [T,52,3]`。

### 3.3 场景关系(配对可恢复性)

- PT 人 → NPZ human:精确相似变换(全帧残差 max = 0.0),scale = 1.32/身高(实测 0.68–0.80,对应身高 1.65–1.94m)。
- original 变体:同一旋转把 PT 物体姿态映射到 NPZ 箱体姿态(误差 ≤3e-6 度);箱体位置 = 相似变换结果 + 每段常数偏移(纯 z,std=0,例 62mm)。
- 增强变体:human_joints 与 original 完全相同;机器人根与箱体按增强量变换(rot ≤45°、trans 平移),且机器人 z 亦变化(例 +12mm)——教师是**重新优化**的,不是简单变换。
- 结论:**源人场景与目标机器人场景在 NPZ 中天然分离存储,方案 D2 的配对结构得到确认**;参考条件用 `human_joints`(不变),目标箱体轨迹用各变体 qpos 的箱体段。

### 3.4 教师质量(锁定模型回放,FN 数值证据)

71 个样本:关节限位违规 0(顺带验证关节顺序);脚底球最低 z∈[4.0,6.2]mm(球半径 5mm,即贴地);机器人-地穿透 ≤1.7mm、机器人-箱 ≤1.6mm、箱-地 ≤7.4mm;抓取段 palm 与箱面距离最小 -1.6mm(轻微穿透,即"手在箱上")。

**教师存在数毫米级穿透,不能当作无条件可行真值**,后续(§9.1)教师数据需过独立验收器,修正或拒绝超阈值样本。

### 3.5 M0 验收状态

- "至少 10 段可信配对 + 渲染和数值对齐证据":**达成**(24 original + 47 增强变体验证,数值 + EGL 渲染证据)。
- "不能恢复配对时跑通 G1 教师生成":配对已恢复,该路径当前不需要。
- ~~遗留到 M1:接触推断阈值、场景边界、按原始演示分组的划分清单。~~ **已于 2026-09-11(M1 工具就绪后)完成**,依据与产物如下:

**独立分组清单与统计**(`scripts/build_split.py`,种子 20260911,锁定清单在 `derived/data/split/`;受试者哈希用 zlib.crc32——Python 内建 `hash()` 跨进程随机,首次实现曾因此不确定,已修复并双运行验证):

| 分片 | 独立演示 | NPZ 文件 | 帧 | 时长(30Hz) | F=60/stride=30 窗口 |
|---|---|---|---|---|---|
| train | 250 | 1,469 | 258,462 | 2.393 h | 6,431 |
| val | 31 | 178 | 33,038 | 0.306 h | 848 |
| test | 31 | 179 | 32,312 | 0.299 h | 815 |
| 合计 | 312 | 1,826 | 323,812 | 2.998 h | — |

规则:按原始演示分组(全部 rot/trans 变体随演示同分片,不跨集);按受试者分层取 80/10/10;6 名受试者在每个分片均出现(人物重叠为主设置,按人物划分为另设实验);test 31 段 < 100 段目标,按方案 §1.3 使用全部预留演示并如实报告;17 个无教师 PT 记录为 unsupervised_reserve,不进入监督分片。窗口数仅作训练规模参考,不视为独立数据量。

**场景边界**(312 段 original 全量统计,锁定于 `configs/task/box_carry.yaml`):root 高度 [0.305, 0.814] m;箱体高度 [0.179, 1.561] m(静置中位 0.185);根-箱水平距离 [0.144, 1.340] m;人体关键点 z ∈ [-0.031, 1.482] m(脚部网格噪声可低于地面 ~3cm);数据中观测峰值速度:箱 2.26 m/s、root 5.29 m/s(数据范围,不是验收限值)。

**接触推断阈值**(参考侧,`scripts/calibrate_contact_thresholds.py`,18 段 original 对拍机器人侧几何推断):hand enter/exit = 0.04/0.072 m,foot enter/exit = 0.05/0.08 m,宏 F1 0.876(手 0.85-0.88、脚 0.88-0.94;个别片段手部 F1=0,源于参考侧 AABB 代理与 hull 的差异,列入 pilot 人工复核)。已锁定到 `configs/eval/protocol.yaml` 与 `infer_reference_schedule` 默认值。
