# 本地资源登记

检查日期:2026-09-11。本次为文件系统与 XML 静态检查,尚未完成 MuJoCo 编译、FK 对拍或 PyTorch 数据加载。

## 目录与用途

以下路径均相对项目根目录:

```text
resources/
  README.md                         # 资源说明、已验证状态和已知问题
  manifest.json                     # 上游版本、下载校验及资产 SHA-256
  data/
    OMOMO_new -> /home/c112/codes/bill/smrt-data/unzipped/OMOMO_new
    omniretarget/
      README.md                     # 数据发布方原始数据卡
      visualize.py                  # 发布方可视化脚本,尚未运行
      robot-object.zip              # 已校验的原始下载归档
      robot-object/                 # 1,826 个 G1 largebox 教师轨迹 NPZ
      models/
        g1/                         # 与教师数据一起发布的 G1 资产
        largebox/                   # 与教师数据一起发布的箱体资产
        ground_box.sdf              # 场景模型描述文件,不是距离场网格
  robots/
    g1_holosoma/                     # 用户提供的 Holosoma G1 原始模型
      assets/                       # URDF 等引用的几何资源
      meshes/                       # MJCF 等引用的几何资源
      g1_29dof.urdf
      g1_29dof.xml
      g1_29dof_spherehand.urdf
      g1_29dof_spherehand.xml
      g1_29dof_w_largebox.xml
    largebox/                       # 从 Holosoma 补齐的 OBJ/URDF/XML
```

| 资源 | 在项目中的用途 | 使用边界 |
|---|---|---|
| 处理版 OMOMO | 人体参考动作、物体运动、接触推断和演示分组来源 | 属于源数据,通过软链接只读使用 |
| OmniRetarget robot-object | 配对审计后的 G1 监督目标、教师质量评估 | 含增强,不能把 1,826 个文件视为独立原始演示 |
| OmniRetarget 发布模型 | 复核公开教师轨迹所用关节/物体几何 | 与当前 Holosoma 模型分开管理 |
| Holosoma G1 模型 | G1 FK、碰撞检查、接触代理与后续重定向候选资产 | 修正格式差异并完成对拍后才能锁定正式版本 |
| Holosoma largebox | 补齐 G1 搬箱场景的几何与模型引用 | 不能仅凭同名认定与数据发布版缩放一致 |
| manifest.json | 来源追踪、版本复现和完整性校验 | 当前覆盖下载记录和选定资产,不是每个源 PT/教师 NPZ 的逐文件哈希清单 |

相关上游代码位于资源目录之外的 `third_party/holosoma`。其中重定向配置、数据加载代码和 `demo_data/height_dict.pkl` 可用于核对处理版 OMOMO 的字段与人体缩放。

## 自动获取结果

- Holosoma 重定向源码及配套资源已获取到 `third_party/holosoma/src/holosoma_retargeting`,提交固定为 `bccd4d7451640a2800ddc77e469d911a84f91994`;仓库保留 LICENSE。这是稀疏 checkout,不表示整个上游工程已下载或安装。
- 缺失箱体已补齐到 `resources/robots/largebox`,包含 `largebox.obj/.urdf/.xml`,来自上述提交。原有 `g1_29dof_w_largebox.xml` 的全部 mesh 相对路径现在均可解析到文件。
- 用户提供的五个顶层 G1 URDF/XML 与该上游提交对应文件逐字节一致;原始文件未修改,已知 collision 拼写和几何差异仍待处理。
- OmniRetarget `robot-object.zip` 已下载并解压到 `resources/data/omniretarget/robot-object`,数据版本为 `135259130158f1cb1799e4ab9b6566c938285fc9`。归档 SHA-256 与发布方 LFS 记录一致,ZIP CRC 检查通过。
- 实际包含 1,826 个 largebox `.npz`,全部含 `qpos`、`human_joints`、`fps`。通过 NPY 头部检查确认 `qpos` 为 `[T,43]`,fps=30,总计 323,812 帧(约 3.00 小时)。这是包含增强的轨迹时长。
- 按文件名可关联到本机 312 个不同 OMOMO 源文件,无缺失源文件名;尚未证明增强关系、人体坐标/尺度或逐帧时间对齐正确。
- 已下载该数据版本配套的 `models/g1`、`models/largebox`、`models/ground_box.sdf`、数据卡及可视化脚本,保存在 `resources/data/omniretarget`。模型逐文件通过大小及上游 Git blob/LFS 哈希验证。发布版模型与当前 Holosoma 模型分开保存,不假定两者相同。
- 版本、来源及资产 SHA-256 见 `resources/manifest.json`。大型下载和上游 checkout 已加入主项目忽略规则;本地清单与说明保留。
- 本轮磁盘快照约 210GiB 可用(下载前),高于先前约 53GiB 的记录。未下载无关地形数据或受账户授权约束的人体参数模型,也未安装训练依赖。

## 数据

- 项目入口:`resources/data/OMOMO_new`。
- 软链接目标:`/home/c112/codes/bill/smrt-data/unzipped/OMOMO_new`。
- 共 4,421 个 `.pt` 文件,合计 1,920,971,225 字节(约 1.79GiB);其中 `*_largebox_*.pt` 共 329 个。这是文件数,不是已验证的独立演示数。
- 示例:`sub3_largebox_003.pt` 存在,属于 PyTorch ZIP 序列化容器。当前仅检查容器和序列化元数据,尚未确认张量维度、列语义、帧率、坐标约定与配对关系。
- 将链接入口视为只读输入;软链接本身不提供写保护,写入会修改源数据。转换结果、缓存和标签必须写到项目另设的派生数据目录,不得写入本目录。
- 软链接避免复制,目录硬链接不适合作为接入方式。迁移机器后需重新建立链接。

### 字段与配对核查依据

固定提交的 Holosoma `holosoma_retargeting/src/utils.py::load_intermimic_data` 将 PT 读取为张量,从列 `[162:318]` 提取并 reshape 为 `[T,52,3]` 人体关键点,从列 `[318:325]` 读取物体状态并重排四元数。这是上游加载约定,尚未通过本地 PyTorch 数值加载验证,不能直接认定所有文件都符合该布局。

公开教师 NPZ 的实际字段比数据卡更完整:

| 字段 | 已确认内容 | 待确认内容 |
|---|---|---|
| `qpos` | 全部文件头部形状为 `[T,43]` | 关节名称/顺序、根与物体位姿排列、数值范围 |
| `human_joints` | 全部 1,826 个文件中存在 | 具体形状、人体缩放、坐标系、与源 PT 的时间对应 |
| `fps` | 全部为 30 | 重采样来源与增强过程是否改变时序 |

数据卡将 `qpos` 描述为机器人根 `[qw,qx,qy,qz,x,y,z]`、29 维关节角、物体 `[qw,qx,qy,qz,x,y,z]`。实际转换前须与发布版模型和上游加载器交叉验证;项目内部采用 xyz+wxyz,必须显式转换。

当前 312 个源文件名匹配仅用于定位候选配对。训练划分必须按原始演示分组,同一演示的旋转/平移增强不能跨训练、验证和测试集。现有处理版 OMOMO 加载路径直接提取关键点,暂不需要另行下载 SMPL/SMPL-H 参数模型。

## G1 模型

来源:用户从 Holosoma 重定向资源复制到 `resources/robots/g1_holosoma`。现已确认五个顶层模型文件与上述固定提交完全一致;原始文件保留不改动。

| 文件 | 静态检查 |
|---|---|
| `g1_29dof.urdf` | 29 个 revolute 关节、25 个 fixed 关节;标准 XML mesh 相对路径均存在 |
| `g1_29dof.xml` | 29 个 hinge 关节及浮动基座;按 meshdir 解析的 mesh 引用均存在 |
| `g1_29dof_spherehand.urdf` | 29 个 revolute 关节、23 个 fixed 关节;标准 XML mesh 相对路径均存在 |
| `g1_29dof_spherehand.xml` | 29 个 hinge 关节及浮动基座;按 meshdir 解析的 mesh 引用均存在 |
| `g1_29dof_w_largebox.xml` | 箱体 mesh 引用已补齐;按 meshdir 解析后指向 `resources/robots/largebox/largebox.obj` |

### 碰撞小球

- 每只脚有 5 个 `ankle_roll_sphere_*` 碰撞球,半径 5mm;踝部 `ankle_intermediate_1_link` 球半径 10mm。
- 普通手版本 thumb/pinky 标记通常显示为半径 5mm 的 visual 球,但 collision 球半径为 1mm。可视外观大小不能直接作为距离约束半径。
- spherehand 版本包含半球形手部 mesh 和半径 1mm 的 tip 碰撞球,并非仅把普通手改成一个球。
- MJCF 中 `contype=0, conaffinity=0` 的 visual geom 不参与普通自动接触配对;collision geom 另行定义。实际优化是否查询某个球还需核对上游碰撞对筛选。
- 这些几何是局部接触/碰撞代理,不能据此认为离散小球覆盖了机器人全部表面。

### 待处理差异

1. 普通版 URDF 的 `left_pinky_link` 写成 `<collsion>` 而非 `<collision>`,标准解析器不会将其识别为碰撞元素;对应 MJCF 存在该球。后续在可追溯的派生模型中修正并验证,不静默修改原始资产。
2. spherehand URDF 含 4 个 cylinder collision,对应 MJCF 的相关 cylinder geom 被注释。两种格式不能假定碰撞模型完全相同,需选定统一的验收几何。
3. URDF 内嵌 MuJoCo compiler 的 meshdir 与一般 URDF 路径解析规则需通过实际加载验证。文件存在不等于模型可成功编译。
4. 带箱体场景的配套 `largebox.obj` 已从固定上游提交补齐,后续仍需核对它与 OMOMO/公开教师数据中的缩放、原点及位姿约定。
5. 运动学初始候选采用普通 `g1_29dof.urdf/.xml`,与搬箱任务一致性需经上游配置及数据核查后锁定。spherehand 作为不同接触几何候选保留,不混用两套模型生成教师与训练标签。

下一步先完成依赖 smoke test、数据张量语义核查、模型编译与关节/碰撞几何对拍,再锁定正式训练资产与数据清单。

## 来源与维护规则

- [Holosoma 上游仓库](https://github.com/amazon-far/holosoma),固定提交与本地资产哈希见 `manifest.json`;许可证保留在 `third_party/holosoma/LICENSE`。
- [OmniRetarget 数据集](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset),固定 revision 及归档哈希见 `manifest.json`;发布方数据卡保留在 `resources/data/omniretarget/README.md`。
- OMOMO 由用户提供本地处理版目录,具体下载版本/处理来源尚未单独登记;不能由文件名推断许可证或完整处理历史。
- 原始模型、下载归档和源数据保持不变。模型修复、重采样、接触标签、数据转换和训练划分输出到单独的派生目录,记录源路径、版本、参数和校验值。
- 主项目忽略 `resources/data/OMOMO_new`、`resources/data/omniretarget/` 及 `third_party/holosoma/`,因此仅复制项目代码不会带上这些资源。恢复环境时按 manifest 获取固定版本并重建 OMOMO 软链接。
- 更新资源后同步更新本文件及 manifest,明确哪些检查重新执行。依赖与脚本统一使用 Python 3.12 和 `uv run --locked`,安装依赖不等于完成模型/数据验证。
