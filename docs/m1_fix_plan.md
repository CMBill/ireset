# M1 审阅问题修复方案

日期:2026-09-11。针对 M1 代码审阅发现的 6 个问题的代码级修复计划。

**范围**:接触推断链路、验收门槛、纯优化基线首帧、有限性检查、死配置。不改动已验证正确的部分(FK/雅可比、交互网格、碰撞对注册、坐标系映射)。

**核心结论**:问题 1–3 相互耦合,都会改变 `protocol.yaml` 中已锁定的接触阈值与教师基线数字。这些数字是 M2/M3/M4 全部相对门槛的分母,必须**先修机制、再重新标定、最后重新锁定**,不能只改其中一环。

> 本文中标注 **[实测]** 的数字已在本机验证;标注 **[待定]** 的是提议值,须由 Phase C 的重标定确认。

---

## 0. 修复顺序与依赖

```text
Phase A(独立,可立即合并)
  Fix 4 基线首帧 yaw    Fix 5 有限性门     Fix 6 自适应细分
        |                     |                  |
        +---------------------+------------------+
                              |
Phase B(机制修正,改变 F1 输入)
  Fix 2 脚部表面距离+阈值不变式   Fix 3 参考侧改用 hull   Fix 1 F1 未定义策略
                              |
Phase C 重新标定阈值(扩展网格,新目标函数)
                              |
Phase D 重跑教师质量/基线,重新锁定 protocol.yaml + 文档
```

Phase B 的三项必须一次性合并后再进 Phase C:任何一项单独改动都会让标定结果失效。

---

## Fix 1:接触 F1 在"双方无区间"时返回 1.0

### 根因

[`eval/metrics.py:21-22`](../src/ireset/eval/metrics.py#L21-L22) 在 `tp+fp==0` 与 `tp+fn==0` 时各自回退 `1.0`。当某末端在参考侧和机器人侧**都没有区间**时,`tp=fp=fn=0`,P=R=1.0 → F1=1.0。

**[实测]** 已锁定的 24 段教师报告中,4 个 clip-effector 组合属此情形(`left_hand` 3 个、`right_hand` 1 个),白拿 1.0 计入 0.872 宏平均。

危害不止报告失真:`calibrate_contact_thresholds.py` 用 macro-F1 作**优化目标**,"少产生区间"本身就能提分,存在系统性偏置。方案 §10.1 明确要求拒绝样本记 0。

### 修复

`contact_prf` 在三者皆零时返回 `None`(未定义),宏平均只对已定义末端求均值,并显式报告已定义数量:

```python
def contact_prf(predicted, reference, effector, T) -> tuple[float, float, float] | None:
    """Frame-level precision/recall/F1; None when neither side claims contact.

    A double-empty effector carries no evidence: scoring it 1.0 would reward
    a scheduler for producing no intervals at all (plan section 10.1 requires
    rejected/absent cases to score 0, never a free pass).
    """
    p = predicted.mask(effector, T)
    r = reference.mask(effector, T)
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
```

`contact_f1_all` 同步返回 per-effector 的 P/R/F1(方案 §10.1 要求分别报告)、`None` 占位与 `n_defined`;全部未定义时 `macro_f1=None`,不得填 0 或 1。

补充方案 §10.1 要求的第二个聚合口径(供 M4 使用):

```python
def contact_f1_success_weighted(rows: list[dict | None]) -> float:
    """Demo-macro F1 over the FULL test set; rejected clips (None) score 0."""
    return float(np.mean([0.0 if r is None else r for r in rows]))
```

### 验证

**[实测]** 三种策略下重跑标定(18 段,扩展网格),最优点均为 `hand_enter=0.02`:

| 策略 | 最优 macro-F1 | 最优 hand_enter |
|---|---|---|
| 现状(双空=1.0) | 0.8813 | 0.02 |
| skip(本方案) | 0.8782 | 0.02 |
| zero | 0.8258 | 0.02 |

策略变化不移动最优点(说明偏置未主导阈值选择),但改变报告数值。采用 `skip`:双空是"无证据",记 0 会惩罚正确的"确实无接触";记 1 会奖励空调度。同时在报告中输出 `n_defined`,使双空占比可见。

新增 `tests/test_metrics.py`:双空返回 `None`;宏平均跳过未定义;全空 → `None`;单侧空 → F1=0.0(不是 1.0)。

---

## Fix 2:区间推断阈值宽于其必须满足的验收门槛

### 根因

[`repair/validation.py:41-52`](../src/ireset/repair/validation.py#L41-L52) 存在阈值倒置:

| 项 | 建立区间的阈值 | 该区间必须满足的门槛 | 倒置 |
|---|---|---|---|
| 手 | `hand_enter_m` = 0.02 | `hand_gap_m` = 0.01 | 是 |
| 脚 | `foot_enter_m` = 0.012 | `foot_ground_gap_m` = 0.005 | 是 |

一个真实的 1.5cm 间隙抓取会被推断为接触、然后**必然**判失败——失败源于阈值不自洽,不是轨迹坏。

**脚部还叠加了半径 bug**:sole sites 是半径 5mm 碰撞球的**球心**,[`g1_spec.py:16-18`](../src/ireset/embodied/g1_spec.py#L16-L18) 的 docstring 自己写明"必须沿接触法向减去半径",但 `SOLE_SPHERE_RADIUS_M` 常量**全项目零引用**。`_infer_robot_schedule` 用 `site_xpos[:,2].min()` 当离地间隙,完美贴地的脚读数恰为 0.005 = 门槛值,ratio 精确 1.0,任何噪声即失败。

**[实测]** 半径修正后的值与 MuJoCo 真实表面距离**完全一致**(相关系数 1.000000,maxdiff 0.000000),证实这是纯 bug 而非近似:

```text
sole 球心 z        : min=0.0040  p05=0.0044  median=0.0084
球心 z - 半径      : min=-0.0010 p05=-0.0006 median=0.0034
mj_geomDistance    : min=-0.0010 p05=-0.0006 median=0.0034   <- 与上行逐值相同
```

这很可能是教师 `foot_support` 通过率 0% 的主要人为成因,而该数字已被写入 `protocol.yaml` 当基线。

### 修复

**2a. 脚部离地间隙改用 MuJoCo 表面距离**(而非球心 z 减常数)

球心减半径只在脚掌水平时等于表面距离;`mj_geomDistance` 对倾斜脚同样精确,且与验收器已有的手部查询口径统一。验收器已跑 1108 对/帧,多 10 次查询成本可忽略(**[实测]** 5ms/帧)。

```python
# _infer_robot_schedule: replace sole_z (site-center height) with the true
# sphere-surface gap to the ground geom. Site centers are still used for
# anchor/drift/speed (they move rigidly with the foot); only the *gap* needs
# the radius removed, and mj_geomDistance does it exactly (also for tilted feet).
self.sole_geoms = {
    s: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                          f"{'left' if s == 'l' else 'right'}_ankle_roll_sphere_{i}_link")
        for i in range(1, 6)]
    for s in ("l", "r")
}
...
sole_gap[s][t] = min(
    mujoco.mj_geomDistance(self.model, d, g, self.ground_geom, 1.0, self._fromto)
    for g in self.sole_geoms[s]
)
```

`SOLE_SPHERE_RADIUS_M` / `HAND_TIP_SPHERE_RADIUS_M` 保留为资产元数据,但注明"验收不通过减法使用,几何查询走 MuJoCo",避免再次出现"定义了却没人用"的假安全感。

**2b. 引入阈值不变式,由 `__post_init__` 强制**

```python
def __post_init__(self) -> None:
    """Inference thresholds must not exceed the gates the interval must satisfy.

    An interval created at a distance the acceptance gate already rejects is
    guaranteed to fail for a reason that is about configuration, not motion.
    """
    if self.hand_enter_m > self.hand_gap_m:
        raise ValueError(f"hand_enter_m {self.hand_enter_m} > hand_gap_m {self.hand_gap_m}")
    if self.foot_enter_m > self.foot_ground_gap_m:
        raise ValueError(f"foot_enter_m {self.foot_enter_m} > foot_ground_gap_m {self.foot_ground_gap_m}")
    if self.hand_exit_m < self.hand_enter_m or self.foot_exit_m < self.foot_enter_m:
        raise ValueError("exit threshold must not be tighter than enter (hysteresis)")
```

门槛侧不放宽:`hand_gap_m=0.01`、`foot_ground_gap_m=0.005` 是方案 §10.1/§14.3 锁定的验收容差。改推断侧:

| 参数 | 现值 | **[待定]** 提议 |
|---|---|---|
| `hand_enter_m` | 0.02 | 0.01(= `hand_gap_m`) |
| `hand_exit_m` | 0.06 | 0.03 |
| `foot_enter_m` | 0.012 | 0.005(= `foot_ground_gap_m`) |
| `foot_exit_m` | 0.03 | 0.02 |

**[实测]** 脚部修正(表面距离 + enter 0.005/exit 0.02)的效果——区间数几乎不变,伪失败大幅下降:

| 配置 | 推断区间数 | 违反 gap 门槛 | 最差 ratio |
|---|---|---|---|
| 现状(球心 z,enter 0.012) | 208 | 201(97%) | 5.96× |
| 修正(表面距离,enter 0.005) | 206 | 145(70%) | 3.99× |

残余 70% 是教师真实的贴地滑移/漂浮(方案 §9.1 预期的"教师非可行真值"),即真阳性。

**2c. 漂移/速度检查改为逐 site,不用质心**

`_check_feet` 用 5 个 sole site 的**质心**算漂移,`_check_hands` 用 palm/thumb/pinky 的质心算锚点偏差。脚原地绕竖轴旋转时质心不动而各 site 移动——漏检。方案 §10.1 明确要求"支撑期**所有受约束 site**的世界系速度及累计锚点漂移":

```python
# per-site, not centroid: a foot pivoting in place keeps its centroid fixed
pts = cache["sole_sites"][s][iv.start:iv.end]        # [n, 5, 3]
speed = np.linalg.norm(np.diff(pts, axis=0), axis=-1).max() / dt
drift = np.linalg.norm(pts - pts[0], axis=-1).max()
```

手侧同理:锚点按 site 分别记录与检查,不再用单一质心锚点。

### 验证

新增 `tests/test_contacts.py`(方案目录列出但尚缺):
- 倒置配置构造 `ValidationConfig` 必须抛 `ValueError`;
- 合成"完美贴地"状态(球面切地)通过 gap 门槛,ratio < 1.0(现状恰为 1.0);
- sole 表面距离与 `mj_geomDistance` 逐值一致;
- 原地绕竖轴旋转的脚被 2c 检出、被质心版漏检(回归锁定);
- `hysteresis_intervals` 的进入/退出/最短时长边界。

---

## Fix 3:参考侧手部距离用 AABB 代理,误差与阈值同量级

### 根因

[`contact_sites.py:128-129`](../src/ireset/embodied/contact_sites.py#L128-L129) 默认走 `point_box_surface_distance_numpy`(解析 AABB)。[`distance_adapter.py`](../src/ireset/geometry/distance_adapter.py) 的 docstring 明确写"AABB 仅训练代理,验收一律用 hull"——但参考调度正是用来算 F1 这个头等指标、并用来标定阈值的。

**[实测]** 扫描箱是带圆角的非水密容器,AABB 误差在**接触判定区间**恰与阈值同量级:

| 真实 hull 距离 | \|AABB − hull\| mean | max |
|---|---|---|
| 0cm(表面) | 0.0cm | 0.0cm |
| 2cm | 1.8cm | 2.0cm |
| 4cm | 3.2cm | 4.0cm |
| 8cm | 4.8cm | 8.0cm |

参考侧阈值是 enter 4cm / exit 7.2cm ——误差与阈值本身同量级。**[实测]** 某真实 clip 右手 70% 帧的 AABB 距离饱和为 0(迟滞在该区间毫无分辨力)。

**[实测]** 更关键:代理**移动了最优点**。手部阈值扫描的最优位置在两种度量下不同——

| 度量 | 最优 hand_enter | 该点 F1 |
|---|---|---|
| AABB(现状) | 0.02 | 0.872 |
| hull | 0.04 | 0.836 |

宏 F1 总体量级不翻转(0.599 vs 0.602 的另一组抽样对照),但"锁定 0.04"这个数字与一个误差同量级的代理纠缠在一起,不能作为标定结论。

### 修复

`distance_adapter.py` 新增批量 hull 查询,凸包从编译后的场景 mesh 缓存一次:

```python
_HULL: trimesh.Trimesh | None = None

def largebox_hull() -> trimesh.Trimesh:
    """Convex hull of the compiled largebox mesh (the collision geometry).

    This is the acceptance-grade geometry; the analytic AABB in this module is
    a differentiable *training* proxy only (mean ~1.8cm error at a true 2cm
    gap, which is the regime contact inference decides in).
    """
    global _HULL
    if _HULL is None:
        m = G1Spec().load_scene()
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "largebox_mesh")
        verts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        _HULL = trimesh.Trimesh(vertices=verts.astype(np.float64), process=False).convex_hull
    return _HULL


def largebox_hull_surface_distance(points_world: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
    """Unsigned hull surface distance (0 inside), [T]; batched over frames."""
    geom = world_to_box_geom(points_world, box_pose).reshape(-1, 3)
    return np.clip(-trimesh.proximity.signed_distance(largebox_hull(), geom), 0.0, None)
```

`infer_reference_schedule` 默认改用它;`hand_distance_fn` 覆盖点保留,AABB 版本保留供训练损失(其可微性是它存在的唯一理由)。

hull 映射的正确性已由 [`test_geometry.py`](../tests/test_geometry.py) 验证(MuJoCo hull 与 trimesh hull 差 2e-4 = 探测球 2r,即帧映射精确),本修复复用同一路径,无新增几何风险。

### 验证

`tests/test_contacts.py` 增:参考调度默认度量为 hull;同一状态下 hull 与 AABB 在近表面区给出显著不同的区间(锁定"代理不可用于验收"这一事实,防止回退)。

---

## Fix 4:纯优化基线首帧根 yaw 公式错误

### 根因

[`eval/baselines.py:58`](../src/ireset/eval/baselines.py#L58):

```python
yaw = np.arctan2(-(sh_l[1] - sh_r[1]), sh_l[0] - sh_r[0])  # +x along shoulder line
```

注释的约定本身就错:人形的左肩在 **+y**,不是 +x。**[实测]** 该式给 −0.88°,教师真值 −87°(近 90° 偏差)。

### 修复

```python
# The G1 left shoulder sits at +y in the body frame, so the human shoulder
# line maps to the robot +y axis (not +x).
d = sh_l - sh_r
yaw = np.arctan2(-d[0], d[1])
```

**[实测]** 修正后 −89.12°,对教师真值 −87.07° 误差 2°。

**4b(同函数,较小残差)**:`pos = kpts[0]` 直接把骨盆 **site** 位置当根位置,注释写 "ignore offset"。**[实测]** 首帧根位置误差 0.117m。`kpt_pelvis` 挂在 `pelvis_contour_link` 上,与根 body 原点有固定外参。建议在名义位姿下预计算该 site→root 偏移并减去,消除这个已知常数偏差。

### 效果与影响范围

**[实测]**:

| 指标 | 现状 | 修正 |
|---|---|---|
| 首帧 MPJPE | 8.1cm | **2.6cm** |
| 整段 MPJPE(120 帧) | 4.96cm | 4.91cm |
| 耗时 | 10.4ms/帧 | 10.4ms/帧 |

整段几乎不变(GN 在后续帧能收回),故**已报告的 4.2cm 量级结论不受影响**。但方案 §8.5 规定纯优化回退"首帧使用参考估计的根位姿",M2 的回退种子会直接继承这个错误初值——必须在 M2 前修掉。

新增回归测试:`initial_state` 的 yaw 与教师首帧根 yaw 差 < 15°(在若干 clip 上)。

---

## Fix 5:缺少显式有限性检查

### 根因

`validate()` 目前靠"NaN 比较恒为 False"顺带拒绝 NaN。**[实测]** 关节 NaN 时 `joint_limits`/`velocities` 的 `worst` 报成 `nan`(无诊断价值);箱体四元数 NaN 会从 scipy `Slerp` 抛 `ValueError: Found zero norm quaternions`,即崩溃而非干净拒绝。

方案 §1.3 要求"超容差、**非有限数**、求解失败或时间不连续的结果必须拒绝",§14.2 要求返回状态码。

### 修复

在任何几何/scipy 调用之前加前置门,并给 `ValidationReport` 增 `status` 字段:

```python
def validate(self, scene_qpos: np.ndarray) -> ValidationReport:
    q = np.asarray(scene_qpos, dtype=float)
    if not np.isfinite(q).all():
        bad = np.argwhere(~np.isfinite(q))
        return ValidationReport(
            passed=False, status="rejected_nonfinite",
            checks={"finite": CheckResult(
                False, float("inf"), 0.0,
                location=f"t={bad[0, 0]} qpos_idx={bad[0, 1]}",
                detail=f"{len(bad)} non-finite entries")},
        )
    for name, sl in (("root_quat", ROOT_QUAT_SLICE), ("box_quat", slice(39, 43))):
        n = np.linalg.norm(q[:, sl], axis=1)
        if (n < 1e-8).any() or not np.isfinite(n).all():
            return ValidationReport(
                passed=False, status="rejected_degenerate_quaternion",
                checks={"quaternion": CheckResult(
                    False, float(n.min()), 1e-8, location=f"{name} t={int(np.argmin(n))}")},
            )
    ...
```

`status` 取值:`passed` / `rejected_nonfinite` / `rejected_degenerate_quaternion` / `rejected_tolerance`。方案 §8.5 强调 `needs_review` 不得伪装为 `accepted`,状态码是 M2 `TrajectoryResult` 的前置件,现在建立可避免 M2 重构。

### 验证

`tests/test_validation.py` 扩展:关节 NaN → `status=="rejected_nonfinite"` 且不抛异常;箱体四元数 NaN → 干净拒绝(现状抛 `ValueError`);零范数四元数 → `rejected_degenerate_quaternion`。现有 NaN 测试的 `pytest.skip` 分支可删除。

---

## Fix 6:已锁定 yaml 中的死配置

### 根因

`max_substep_motion_m` 与 `max_subdivisions` 在 [`ValidationConfig`](../src/ireset/repair/validation.py#L53-L56) 和 `protocol.yaml` 均声明,**[实测]** 各仅出现 1 次(即定义处),从未被读取:`_check_penetration` 硬编码单级中点插值。`max_check_interval_m` 算出来后也不与任何限值比较。

`docs/m1_verification.md` §8 承认了这点,但 `protocol.yaml` 呈现为已生效配置——对读者是误导,且方案 §6.3 要求"按运动幅度**自适应细分**的帧间插值"。

另有语义 bug:`motion` 混用米与弧度(`norm(dpos) + max|dq|`),量纲不一致;且 `max_check_interval_m` 记录的是**细分前**的帧间运动,而非细分后的实际最大检查间隔。

### 修复

按运动幅度真正细分,并用笛卡尔量度取代量纲混用:

```python
def _frame_motion_m(self, qa, qb) -> float:
    """Max site displacement between two states (true Cartesian motion).

    Replaces the dimensionally inconsistent ``|dpos| + max|dq|`` metric: joint
    radians and base meters are not addable, and the joint term understates
    motion of distal sites on long links.
    """
    xa = self._all_site_xpos(qa)
    xb = self._all_site_xpos(qb)
    return float(np.linalg.norm(xb - xa, axis=-1).max())

# in _check_penetration:
for t in range(T - 1):
    motion = self._frame_motion_m(q[t], q[t + 1])
    n_sub = int(np.clip(np.ceil(motion / cfg.max_substep_motion_m), 1, cfg.max_subdivisions + 1))
    for i in range(1, n_sub):                       # interior points only
        probe(self._interpolate(q[t], q[t + 1], i / n_sub), f"t={t}+{i}/{n_sub}")
    max_interval = max(max_interval, motion / n_sub)  # post-subdivision interval
```

`max_check_interval_m` 语义改为"细分后的最大检查间隔",并作为**报告项**与 `max_substep_motion_m` 一并写入 report(方案 §6.3:报告最大检查间隔,不宣称形式化零碰撞证明)。若 `max_subdivisions` 上限导致间隔仍超 `max_substep_motion_m`,在 report 中显式标注该事实,不静默通过。

### 验证

`tests/test_validation.py` 增:构造高速帧对 → `n_sub` 达上限且 `max_check_interval_m` < 帧间运动;构造慢速帧对 → 不细分;人为在帧**中间**制造穿透(两端干净)→ 必须被细分检查捕获(现状单级中点在某些运动下会漏)。

---

## Phase C:重新标定阈值

Phase B 合并后,现有标定结果全部失效。重跑 `scripts/calibrate_contact_thresholds.py`,三处改动:

**C1. 扩展网格下界** —— 现网格 `hand_grid = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]`,最优点 0.04 恰在**下边界**,是网格伪影而非真优化点。**[实测]** 扩展到 0.005 起后,AABB 度量下最优为 0.02:

```text
hand_enter:  0.005→0.8616  0.010→0.8592  0.015→0.8576
             0.020→0.8716 (最优)  0.025→0.8699  0.030→0.8686  0.040→0.8655
```

新网格:`hand_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.06]`。

**C2. 脚部阈值退出网格搜索** —— **[实测]** 脚部 F1 在 enter ≥ 0.03 后完全平坦(平台,非优化点):

```text
foot_enter: 0.010→LF 0.577  0.020→0.888  0.030→0.890  0.050→0.892  0.080→0.892  0.120→0.892
```

成因:**[实测]** 人体足高分布 median 0.0084、p75 0.0150,**82% 帧低于 0.02、90% 低于 0.03、99% 低于 0.05**。锁定的 0.05 深在饱和区,该参数实际不起判别作用,"标定得到 0.05"是假精度。脚部阈值应由 **2b 的不变式**(`foot_enter_m ≤ foot_ground_gap_m`)直接确定为 0.005,并在文档中说明其不可由 F1 标定——把它留在网格里只会产生虚假的"已优化"印象。

**C3. 目标函数改为 skip 策略 + hull 参考度量**,并同时记录 `n_defined` 与 exit/enter 比值(现固定 1.8/1.6,属未标定的隐含假设,至少需记录为假设)。

---

## Phase D:重新锁定基线与文档

1. `scripts/validate_teacher.py --n 4 --include-aug` 重跑 → 覆写 `derived/data/validation/teacher_quality.json`。
2. `scripts/run_baseline.py` 重跑 → 覆写 `baseline_quality.json`(首帧 MPJPE 应从 8.1cm 降至 2.6cm)。
3. 覆写 `configs/eval/protocol.yaml` 的 `contact_inference` 与 `teacher_quality_m1_baseline` 两节。**旧数字必须在文档中保留并标注失效原因**,不可静默替换——它们已被 `docs/m1_verification.md` 和 `AGENTS.md` 引用为已验证结果。
4. 更新 `docs/m1_verification.md`(新增"修复后基线"一节,保留原值与差异说明)与 `AGENTS.md` 的进度摘要数字(当前写着 hand 0.04/0.072、foot 0.05/0.08、F1 0.876、教师 F1 0.872)。

### 预期数字变化

| 项 | 现锁定值 | 修复后 | 依据 |
|---|---|---|---|
| `ref_hand_enter_m` | 0.04 | **[待定]** 0.02–0.04 | C1 扩展网格;hull 与 AABB 最优点不同 |
| `ref_foot_enter_m` | 0.05 | 0.005 | C2:不变式决定,非 F1 标定 |
| `hand_enter_m`(机器人侧) | 0.02 | **[待定]** 0.01 | 2b 不变式 |
| `foot_enter_m`(机器人侧) | 0.012 | 0.005 | 2b 不变式 |
| 标定 macro-F1 | 0.876 | **[待定]** ≈0.82–0.88 | 取决于 skip 策略与 hull 组合 |
| 教师 `foot_support` 通过率 | 0% | **[待定]** 上升 | 伪失败 97%→70%,残余为真滑移 |
| 教师接触 F1 | 0.872 | **[待定]** 下降 | 双空不再白拿 1.0 |
| 教师穿透 16.5mm | 16.5mm | 不变或略升 | Fix 6 细分可能发现新穿透时刻 |

**教师全段通过率仍应为 0%**——方案 §9.1 的"教师非可行真值"结论由 16.5mm 穿透(远超 2mm)独立支撑,不依赖脚部阈值 bug。该结论不因本次修复而动摇。

---

## 工作量与风险

| Phase | 改动文件 | 估计 | 风险 |
|---|---|---|---|
| A | `baselines.py`、`validation.py`、`test_validation.py` | 0.5 天 | 低,互不耦合 |
| B | `metrics.py`、`validation.py`、`contact_sites.py`、`distance_adapter.py`、新增 `test_contacts.py`/`test_metrics.py` | 1–1.5 天 | 中:改变全部 F1 数字 |
| C | `calibrate_contact_thresholds.py` + 重跑 | 0.5 天(含计算) | 低 |
| D | 3 个 json、`protocol.yaml`、2 个 md | 0.5 天 | 低,但须保留旧值审计痕迹 |

合计约 3 天,在 M2 关键路径之前完成。

**主要风险**:Phase B 后教师质量数字全面变动,若此时已开始 M2 会造成基线漂移——因此应在 M2 动工前完成。**不修的代价**:M2 修复器会针对被 bug 污染的门槛做优化(尤其脚部 gap 的 ratio 恒为 1.0 边界),M3/M4 的"相对基线下降 ≤5 个百分点"等门槛全部建立在错误分母上。

**范围外(已知但本次不修)**:方案目录列出而尚未建立的模块(`repair/context.py`、`repair/scp.py`、`data/*`、`viz/render.py` 等)属 M2+;`build_split.py:14` docstring 写 "32 independent demos" 而实际 31(manifest 正确,仅注释笔误,可随手改);`hysteresis_intervals` 遮蔽内置 `exit`、`retract_batch` 的 `m` 参数未使用、`world_to_box_geom` 不接受 1-D pose —— 均为整洁性问题,建议随 Phase B 顺手处理。
