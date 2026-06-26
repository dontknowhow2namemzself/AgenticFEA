# 悬臂梁 FEA MVP — 工具链验证 Plan (v3.1)

**目标**:在 Apple M5 / arm64 上跑通 `Gmsh → CalculiX (ccx) → PyVista/ParaView` 端到端链路,用线弹性悬臂梁验证整条链的正确性。链路确认无误后,第二步在同一几何上加非线性(NLGEOM / 塑性)。

**本文档范围**:第一步(线弹性 MVP)。第二步钩子预留,不展开。

**版本变更**:
- v2:五层架构;BC 从网格模块拆出;验证层升级为知识库驱动裁判。
- **v3**:验证层从单层扩展为**三层**(物理校验 → 数值验证 → 诊断);**应力退出自动裁判,改由人评判**(原因:固定端应力集中网格依赖发散 + 自由端点载荷假奇异,应力对解析解不收敛);求解输出增加反力(RF);删除数字 score 机制,改清单式检查 + Critical 一票否决。
- **v3.1**:**物理校验前移到后处理之前**——物理校验廉价、不依赖 ParaView,作为快速失败闸门,Critical FAIL 则跳过昂贵的后处理(`.vtu` 转换 + 三张 PNG 渲染)直接进诊断层。新执行顺序:求解 → 物理校验 → 后处理 → 数值验证。

---

## 架构总览(八层职责)

| 层 | 名称 | 职责一句话 | 谁负责 |
|---|---|---|---|
| 1 | **任务理解 (Task Understanding)** | 自然语言 → 明确问题定义 | LLM(语义) |
| 2 | **参数化 (Parameterization)** | 问题定义 → 结构化 `params.py` | LLM 填槽 + 代码校验 |
| 3 | **前处理 (Pre-Processor)** | params → 网格 + BC + `.inp` | 确定性代码 |
| 4 | **求解 (Solver)** | `.inp` → `.frd`(含位移/应力/反力) | CalculiX ccx |
| 5 | **物理校验 (Physics Validation)** | 结果是否符合基本物理规律?(廉价快速失败闸门) | 确定性代码 +(LLM 解释) |
| 6 | **后处理 (Post-Processor)** | `.frd` → `.vtu` → PNG(X/Y/Z 三视图) | 确定性代码 |
| 7 | **数值验证 (Verification)** | 结果是否符合理论解/参考解? | Agent 选依据 + 代码算误差 |
| 8 | **诊断 (Diagnostics Agent)** | 失败了,原因是什么?怎么修? | LLM(纯建议,不执行) |

> 贯穿原则:**LLM 管语义判断,确定性代码管数值正确性。** 数值的对错由代码裁定,LLM 不直接产出或修改数值结果。
>
> 验证三层的分工:**物理校验**抓离谱错误(不需参考解,靠物理律);**数值验证**测数值精度(需参考解);**诊断**在任意环节失败时做根因分析 + 修复建议(不执行)。便宜的粗筛在前,精确的测量在后——错误的结果不值得精确测量它错多少。

---

## 0. 全局约定(所有层共享)

### 0.1 单位制 — SI-mm

CalculiX 无内置单位系统,一致性全靠人保证。全程 SI-mm:

| 物理量 | 单位 | 备注 |
|---|---|---|
| 长度 | mm | |
| 力 | N | |
| 应力 / 杨氏模量 | MPa (= N/mm²) | |
| 密度 | tonne/mm³ | 钢约 7.85e-9 |
| 位移 | mm | 输出 |

> ⚠️ 任何单位换算必须显式注释。解析解对比也用 SI-mm。

### 0.2 命名契约(跨层接口,不可随意改)

| 契约项 | 固定值 | 用途 |
|---|---|---|
| 体积物理组 | `BEAM_VOL` | 单元集,赋材料/截面 |
| 几何面标签 | `face_x_min` / `face_x_max` / ... | 网格层只打几何标签,不含 BC 语义 |
| 固定端节点集 | `FIXED` | BC 层映射生成 |
| 加载端节点集 | `LOAD` | BC 层映射生成 |
| 材料名 | `STEEL` | `*MATERIAL` |

> 关键解耦:**网格层给面打中性几何标签(`face_x_min`),BC 层才把标签映射成 `FIXED`/`LOAD` 语义。** 换 fix/load 位置只改 BC,不动网格。

### 0.3 目录结构

```
project/
├── params.py                     # 参数字典(单一事实来源)
├── schema.py                     # Pydantic 校验模型(参数化层 Tool 用)
├── knowledge/
│   └── reference_solutions.py    # 解析解/参考解知识库(数值验证依据)
├── mesh.py                       # 前处理:几何+网格(打几何标签)
├── boundary_conditions.py        # 前处理:边界条件映射(独立!)
├── build_inp.py                  # 前处理:生成 .inp
├── solve.sh                      # 求解:ccx
├── physics_validation.py         # 物理校验:工程 sanity check(快速失败闸门,在后处理前)
├── postprocess.py                # 后处理:.frd→.vtu→PNG(X/Y/Z 三视图)
├── verification.py               # 数值验证:对解析解(仅位移)
├── diagnostics.py                # 诊断 Agent:根因分析+修复建议(纯输出)
├── run_all.sh                    # 编排,支持 --skip-checkpoints
└── out/
    ├── beam.msh / beam.inp / beam.frd / beam.vtu
    ├── stress_x.png / stress_y.png / stress_z.png
    ├── physics_report.txt
    ├── verification_report.txt
    └── diagnostic_report.txt
```

### 0.4 参数字典结构(`params.py`,带几何分类标签)

```python
PARAMS = {
    "analysis": {
        "type": "static",        # static | fatigue
        "nonlinear": False,      # 第二阶段开
    },
    "geometry": {
        "type": "cantilever_beam",   # ← 分类标签,扩展靠它
        "cantilever_beam": {         # 标签对应参数子集
            "L": 100.0, "h": 10.0, "b": 10.0,
        },
        # 未来: "plate_with_hole": {...}, "free_form": {...}
    },
    "material": {"name": "STEEL", "E": 210000.0, "nu": 0.3, "rho": 7.85e-9},
    "bc": {                          # ← 从几何拆出,引用几何面标签
        "fixed": ["face_x_min"],
        "loads": [{"face": "face_x_max", "dir": [0,0,-1], "magnitude": 1000.0}],
    },
    "mesh": {"elem_size": 2.0, "elem_order": 2},
    "acceptance": {"tol_disp": 0.05},   # 仅位移阈值;应力不自动裁判(见第7层)
}
```

> 注意:`acceptance` 里**只剩位移阈值**。应力阈值已移除——MVP 不对应力自动裁判,原因见第 7 层。

---

## 第 1 层 — 任务理解 (Task Understanding)

| | |
|---|---|
| **负责** | LLM(语义理解) |
| **Input** | 用户自然语言描述 |
| **Output** | 结构化问题定义(喂给第 2 层) |

**1.1 明确维度**(对话补全):计算类型(static/fatigue;linear/nonlinear)、几何、网格、材料、边界条件、载荷、验收阈值。缺失项主动追问。

**1.2 Off-topic 防护**:与本 FEA 任务无关的提问,礼貌拒绝,引导回任务。

**验收**:六维度全部明确(或显式默认补齐并告知),无遗留歧义。

---

## 第 2 层 — 参数化 (Parameterization)

| | |
|---|---|
| **负责** | LLM 填槽 + 确定性代码校验 |
| **Input** | 第 1 层问题定义 |
| **Output** | 校验通过的 `params.py` |

**2.1 做成 Tool(Pydantic schema 约束)**。LLM 不直接写 Python 文件,而是调 `set_params` Tool,只能填预定义槽位。Tool 内部 Pydantic 校验:类型、正负(E>0,L>0)、合理性(L/h 比值警告、nu∈(0,0.5))。校验通过才写入 `params.py`,作唯一事实来源。

**为什么做成 Tool**:把 LLM 输出从自由文本约束成结构化字段,数值正确性归确定性代码,LLM 只管"语义→填槽"。

**几何扩展性**:`geometry.type` 是分类标签,每标签对应独立参数子集。MVP 只实现 `cantilever_beam`;新几何加新标签+新子集+schema 新模型,旧代码不动。

**验收**:通过 Pydantic 校验;`geometry.type` 有对应子集;关键数值在物理合理范围。

---

## 第 3 层 — 前处理 (Pre-Processor)

### 3.1 网格 (`mesh.py`)

| | |
|---|---|
| **Input** | `PARAMS.geometry` + `PARAMS.mesh` |
| **Output** | `out/beam.msh`,含 `BEAM_VOL` + **中性几何面标签** |
| **做什么** | `gmsh.occ` 建 box;给所有外表面打几何方位标签(x_min/x_max/...);整体 `BEAM_VOL`;二阶单元;写 `.msh` v2 ASCII |

> **此模块不含任何 BC 语义。** 只描述几何长什么样、面在哪。

**验收**:`.msh` 非空;单元数合理量级(打印);几何面标签齐全且各面节点数>0;无负雅可比。
**Human 检查点**(可跳过):`gmsh out/beam.msh` 目视确认。

### 3.2 边界条件 (`boundary_conditions.py`) — 独立模块

| | |
|---|---|
| **Input** | `out/beam.msh`(读几何面标签) + `PARAMS.bc` |
| **Output** | `FIXED` / `LOAD` 节点集定义 |
| **做什么** | `PARAMS.bc.fixed` 的几何面标签 → `FIXED` 节点集;`loads` 的面 → `LOAD` 并记录方向/大小 |

> **拆分收益**:换 fix 位置、换 load 面、调载荷方向——全在此模块,不碰网格。

**验收**:引用的几何面标签在 `.msh` 中都存在;`FIXED`/`LOAD` 节点集非空;载荷方向为单位向量。

### 3.3 Human 检查点(可跳过)

前处理完成后停下,允许 human 检查/微调 BC,或 `--skip-checkpoints` 跳过。agent 全自动时默认跳过。

### 3.4 生成求解输入 (`build_inp.py`)

| | |
|---|---|
| **Input** | `.msh` + BC 节点集 + `PARAMS`(E,nu,载荷) |
| **Output** | `out/beam.inp` |
| **做什么** | 写 `*NODE`/`*ELEMENT`(C3D10 或 C3D20R,按 elem_order)/`*ELSET=BEAM_VOL`/`*NSET=FIXED,LOAD`/`*MATERIAL=STEEL`+`*ELASTIC`/`*SOLID SECTION`/`*STEP`+`*STATIC`/`*BOUNDARY`(FIXED 1-3)/`*CLOAD`(LOAD 按方向,总力均分节点)/**输出请求** |

**输出请求(v3 关键)**:
- `*NODE FILE=U`(位移)
- `*EL FILE=S`(应力)
- **`*NODE FILE=RF`(反力)** ← 新增,物理校验层 ΣF=0/ΣM=0 检查的数据来源,缺它物理校验无数据可查

**验收**:节点/单元数与 `.msh` 一致;关键卡块齐全;`*CLOAD` 节点力之和 ≈ 载荷总量;输出请求含 U/S/RF 三项。

---

## 第 4 层 — 求解 (Solver)

| | |
|---|---|
| **脚本** | `solve.sh` |
| **Input** | `out/beam.inp` |
| **Output** | `out/beam.frd`(位移/应力/**反力**) + `.dat` + ccx 日志 |
| **做什么** | `ccx -i out/beam`(不带扩展名);`OMP_NUM_THREADS` 吃满 M5 性能核 |

**验收**:退出码 0;无 `*ERROR`;`.frd` 含 U、S、RF 数据块;提取 max|Uz|、max 应力、反力合用于后续层。

---

## 第 5 层 — 物理校验 (Physics Validation)

> **位置说明(v3.1)**:物理校验前移到后处理**之前**。它廉价(纯数值检查,不依赖 ParaView/渲染),作为**快速失败闸门**:若结果连基本物理律都不满足,**直接跳过后处理**(省下最昂贵的 `.vtu` 转换 + 三张 PNG 渲染),把上下文交诊断层。cheap checks first, fail fast。

| | |
|---|---|
| **脚本** | `physics_validation.py` |
| **负责** | **确定性代码**为主(判定逻辑写死,不让 LLM 判数值);LLM 仅把 FAIL 翻译成工程师能懂的解释 + 兜底未成文的异常 |
| **目的** | 工程 sanity check——抓**离谱错误**,不测数值精度。不需要参考解,靠普适物理律,**对非标准 BC 同样适用** |
| **Input** | **直接从 `out/beam.frd` 提取**(不依赖后处理):位移场、应力场、反力 + 几何、载荷定义、BC、solver 元数据。组装成 `SimulationResult` |
| **Output** | `out/physics_report.txt`:清单式,每条 check 一行 `name / status(PASS/FAIL/WARNING) / message` |

> 关键:应力场提取来自 `.frd` 本身,**不需要先跑后处理的 `.vtu`**——这正是物理校验能前置的前提。

### 检查清单(Critical 标 ★,FAIL 即整体 FAIL)

| # | 检查 | 判据 | 级别 |
|---|---|---|---|
| 1 | **位移方向 ★** | 向下加载 → tip Uz < 0,符号一致 | Critical |
| 2 | **力平衡 ★** | \|Σreaction + Σexternal_load\| / \|load\| < 1% | Critical |
| 3 | **力矩平衡** | 固定端反力矩 ≈ P·L,误差 < 2% | 普通 |
| 4 | **应变能为正 ★** | U > 0 | Critical |
| 5 | **数值有效性 ★** | 无 NaN/Inf,位移场/应力场非空 | Critical |
| 6 | **最大应力位置** | argmax(应力) 在固定端附近,非梁中部 | WARNING |
| 7 | **应力量级** | max 应力在预期量级内(超出 → 网格畸变/奇异嫌疑) | WARNING |

### 判定机制(v3:删除数字 score,改二元结构)

- **Critical 一票否决**:任一 ★ 项 FAIL → 物理校验整体 **FAIL** → **跳过后处理** + 触发诊断层
- **WARNING**:照常列出,**不一票否决、不算分**;提示人注意,常见于应力奇异(预期内)
- 不再有 80 分阈值;清单式呈现,边做边看再决定是否要加权

> few-shot 用途澄清:GPT 给的 few-shot 是用来教 LLM 工程师的 sanity-check **思维模式**(写解释、兜底异常),但**上述每条 check 的判定本身用确定性代码实现**,不让 LLM 去判 ΣF 等不等于 0——否则又掉回"LLM 碰数值"的坑。

**验收**:全部 Critical PASS → 通过,进入后处理;任一 Critical FAIL → 跳过后处理,把上下文交诊断层。

---

## 第 6 层 — 后处理 (Post-Processor)

> **前置条件(v3.1)**:仅在第 5 层物理校验通过后才执行。物理校验 FAIL 时本层被跳过。

| | |
|---|---|
| **脚本** | `postprocess.py` |
| **Input** | `out/beam.frd`(已通过物理校验) |
| **Output** | `out/beam.vtu` + 三张 PNG(`stress_x/y/z.png`,X/Y/Z 视图)+ 位移云图 |
| **做什么** | (a) **转换**:`ccx2paraview out/beam.frd vtu`(PyVista 不能直读 `.frd`);(b) **出图**:PyVista 读 `.vtu`,**von Mises 应力云图**,三个相机方位各出一张;off-screen 渲染 |

**关键依赖**:`ccx2paraview` 需 `pip install`,本链唯一非显然依赖。⚠️
**回退**:headless 出图失败时,ParaView GUI 打开 `.vtu`(human 路径始终可用)。
**记录**:照常记录 max 位移、max 应力(von Mises),供报告汇总。**应力 max 是信息记录,不参与自动裁判**(见第 7 层)。

**验收**:`.vtu` 可被 PyVista 加载;三张 PNG 非空含色标;读出 max 值与第 4 层一致(转换无损)。
**Human 检查点**(可跳过):ParaView 打开确认变形方向、应力分布合理。

---

## 第 7 层 — 数值验证 (Verification) — 知识库驱动裁判

| | |
|---|---|
| **脚本** | `verification.py` + `knowledge/reference_solutions.py` |
| **目的** | 测**数值精度**(不是物理合理性):FEM 结果是否接近参考解 |
| **Input** | `SimulationResult` + 匹配的 `ReferenceCase` |
| **Output** | `out/verification_report.txt`:位移 PASS/FAIL/N/A + 应力信息记录 + 所用依据及适用范围 |

### v3 核心决定:MVP 只自动裁判位移,应力交人评判

**为什么应力退出自动裁判**(必须写进报告,让人看懂"为何只验了位移"):
1. **固定端应力集中网格依赖**:固定端是应力集中区,σxx/von Mises 的 max 随网格加密**单调发散**,理论上不收敛到梁理论的截面值 `Mc/I`——解析解本身不描述角点三维应力。
2. **自由端点载荷假奇异**:集中力加在节点上,该节点周围是**人为数值奇异**(点载荷应力理论无穷大),max 应力在此处也是假的。
3. 结论:应力误差在此配置下**不是有意义的收敛量**,强行设阈值只会让裁判说谎(阈值松=形同虚设,阈值严=频繁假 FAIL)。

**位移为何适合自动裁判**:全局积分量,对局部奇异不敏感,对网格收敛快——线弹性悬臂梁里唯一适合自动对解析解的量。

### 裁判逻辑(知识库驱动,Agent 选依据、代码算误差)

```
问题配置(几何类型 + BC + 载荷)
   │
   ▼
Agent 检索知识库,语义匹配适用参考解
   │
   ├── 命中 → 取参考解 → 【确定性代码】算位移误差 → 比阈值 → PASS/FAIL
   │           (报告标注:用了哪条依据、其适用范围、来源)
   │
   └── 未命中 → 声明"无适用解析裁判" → 位移判 N/A,不伪造
```

> **铁律**:Agent 只从知识库**选**裁判依据,**绝不自己编公式**。误差计算与阈值判定永远由确定性代码执行。

### 检查内容

| # | 量 | 处理 | 阈值 |
|---|---|---|---|
| 1 | **Tip 位移** | 对 `δ=PL³/(3EI)` 算误差,自动裁判 | < 5%(`tol_disp`) |
| 2 | **应力** | **仅记录 max 值(von Mises),不裁判**;报告注明退出原因(上述三点)+ "请结合 PNG 人工判断" | 无(N/A) |
| 3 | **Mesh Convergence** | **MVP 不执行**;仅基于 tip 位移对理论解,不做网格收敛研究 | 范围外 |

### 知识库条目结构(`reference_solutions.py`)

每条:`id`/`name`/`applies_to`(几何类型、BC 模式、材料模型、变形假设)/`formula`(代码可调函数)/`scope_note`(适用范围限制,显式)/`source`/`expected_error`。

### MVP 必备条目

**标准悬臂梁**:`applies_to` = `cantilever_beam` + 一端全约束 + 自由端横向集中力 + 线弹性 + 小变形;`scope_note` = **小变形、线弹性、L/h>>1(梁理论)**;`formula`:`I=b·h³/12`,`δ=P·L³/(3·E·I)`(**仅位移**);`source` = 欧拉-伯努利梁。

### 验收(MVP 数值验证)

- 命中标准条目:位移 `|δ_fem−δ_ref|/δ_ref < 5%` → PASS;否则 FAIL(触发诊断)
- 应力:**始终 N/A**(仅记录),报告附退出原因 + PNG 人工判断提示
- BC 偏离标准 → 不命中 → 位移 N/A + FEM 结果交人,不伪造判定

---

## 第 8 层 — 诊断 (Diagnostics Agent)

| | |
|---|---|
| **脚本** | `diagnostics.py` |
| **负责** | LLM(语义推理);**纯输出建议,一律不自动执行** |
| **触发** | 几何/网格/求解/物理校验/数值验证 任一环节失败时 |
| **Input** | Solver 日志(stdout/stderr)+ 元数据(几何/网格尺寸/BC/载荷/材料)+ `PhysicsValidationReport` + `VerificationReport` |
| **Output** | `out/diagnostic_report.txt` = **Human Review Report** |

### Human Review Report 内容

Failure Summary / Root Cause / Confidence / Suggested Fix / Recommended Next Action。
输出后**暂停执行,等 Human 决策**:Approve(采纳建议继续)/ Reject(停止)/ Modify(改参数重跑指定步骤)。

### 失败分类(初始知识结构)

| 类 | 示例 | 典型根因 | 建议修复 |
|---|---|---|---|
| A Mesh | Negative Jacobian | 单元畸变 | 减小单元尺寸 |
| B Solver | 达到最大迭代 | 发散 | 细化网格 / 减小载荷增量 |
| C Physics | tip 位移方向错 | 载荷符号错 | 检查 load sign |
| D Verification | 位移误差超阈 | 网格/建模问题 | (MVP:人工复核;未来:网格收敛研究) |

### 边界(v3)

- **不修改结果、不自动执行**;`auto_fix_possible` 字段**仅作信息标注**(标记"理论上可自动化",留未来闭环钩子),不触发任何动作
- **网格收敛 Agent** 为未来扩展:将来提供"是否需收敛分析/推荐网格尺寸/计算成本/收敛风险",其建议并入 Human Review Report

**验收(对已知错误案例,逐步逼近)**:Top-1 根因准确率目标 >80%;修复建议成功率目标 >70%。MVP 阶段先把机制跑通,指标边做边量。

---

## 编排 (`run_all.sh`)

依次:前处理(网格→BC→inp)→ 求解 → **物理校验** → 后处理 → 数值验证。物理校验是廉价的快速失败闸门:**FAIL 则跳过后处理**,直接调诊断层。任一环验收 FAIL → 停 + 调诊断层出 Human Review Report,暂停等决策。支持 `--skip-checkpoints` 跳过所有 human 检查点(agent 全自动模式)。

---

## 第二步预留钩子(本 MVP 不实现)

链路确认后,加非线性主要动 `build_inp.py`(`.inp` 生成)+ 知识库 + 验证策略,骨架复用:
- **几何非线性**:`*STEP, NLGEOM` + 增量步
- **材料非线性**:`*PLASTIC`(屈服+硬化);参数加 `yield_stress`;`analysis.nonlinear` 开关已预留
- **验证策略演进(关键)**:非线性阶段**不再对解析解**,改为**非线性结果 vs 线性基线结果**对比 max 位移和 max 应力,Agent 分析变化是否符合工程规律(如塑性后位移增大、应力增长被屈服削顶),反馈给人。这条把验证从"对公式"演进到"对基线",更贴近真实工程做法——**届时应力重新进入对比(对基线,非对解析解),不受 MVP 应力退出自动裁判的限制**
- **验证层框架不变**,知识库加条目即可扩展

---

## 依赖清单(M5 / arm64,已确认原生支持)

| 工具 | 安装 | 角色 |
|---|---|---|
| Gmsh | `pip install gmsh` + 官方 arm64 .dmg(GUI) | 几何+网格,双入口 |
| CalculiX ccx | `conda install -c conda-forge calculix` 或 `brew install costerwi/homebrew-calculix/calculix-ccx` | 求解器 |
| PyVista | `pip install pyvista` | 程序化出图 |
| **ccx2paraview** | `pip install ccx2paraview` ⚠️非显然依赖 | .frd→.vtu |
| Pydantic | `pip install pydantic` | 参数化层校验 |
| ParaView | 官方 arm64 .dmg | human 交互后处理 |

> GPU/MPS:本链路为 CPU 多核(ccx 的 SPOOLES 多线程吃满 M5 性能核),不使用 MPS。
