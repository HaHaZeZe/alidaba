#### 【跑通Baseline】赛题一：市场参与者交易行为识别与资金流向分析

#### 项目简介

### 一、赛题背景介绍

本赛题聚焦构建一套“市场参与者交易行为识别与资金流向分析”的体系化解决方案，要求结合逐笔成交数据、订单簿微观结构特征、个股基本面等信息，实现从海量高频数据中自动解析不同参与者的买卖行为和资金流向，识别背后真实意图的模型——例如识别不同订单的资金属性（游资/量化私募/散户等），分析资金背后意图（吸筹、试盘、对倒、拉升、出货等），并生成清晰易懂的资金流向分析结果（当前谁占主导，买入还是卖出，真实意图是什么）。解决该问题将帮助投资者穿透复杂盘面、理性跟随不同机构资金动向，优化买卖时机，降低被游资或虚假挂单误导的风险，实现从“凭感觉跟风”到“有数据支撑的辅助决策”。

在股票投资中，市场参与者（公募基金、量化私募、游资等）的大额买卖行为往往对股价产生显著影响，其资金流向被视为重要的行情风向标。但普通投资者面对 Level-2 数据时普遍存在两大痛点：

- ##### **信息"碎片化"**：单个指标难以还原机构真实的建仓、拉抬或出货意图，机构常通过拆单、对倒等方式隐藏真实动作。
- **解读"滞后性"**：多数资金流向指标基于日线统计，盘中动态变化无法捕捉，等信号明确时股价往往已大幅偏离理想买点。

**通俗理解**：假设

你是一个散户，看到某只股票突然放量大涨。你想知道——这是游资在拉涨停准备出货，还是量化基金在程序化做 T+0 套利？如果是前者，你该跟；如果是后者，你该躲。本赛题就是训练 AI 来自动回答这个问题。

参赛者基于A股个股的**逐笔委托、逐笔成交、逐笔撤单、十档盘口快照**四大类全量Level2数据，构建完整的方法论体系，精细化区分游资/量化两类机构资金，识别各类市场参与者的买卖方向与交易意图，最终输出可落地的交易模式识别与资金流向分析结果。

### 二、任务介绍

本赛题包含**两大核心任务**，所有输出结果需严格匹配赛事指定格式，最终用于平台T+5日实盘行情回溯评测：

#### Task1：交易模式识别（无监督聚类任务）

- **核心目标**：基于全量Level2特征，对单日个股的交易行为进行无监督聚类，划分出具有显著区分度的交易模式，输出可解释的模式类型与业务说明；
- **核心要求**：聚类结果需满足「类内高聚合、类间高区分」，模式解释需贴合A股市场真实交易逻辑，无业务常识错误；
- **输出文件**：`pattern_reco.csv`，固定4个字段，顺序不可修改：`stock_code`(股票代码)、`transaction_date`(交易日期)、`pattern_type`(交易模式类别)、`pattern_explanation`(交易模式详细说明)。

#### Task2：资金类型与交易意图识别（无真值规则判别任务）

- **核心目标**：基于Level2数据特征，识别单日个股的主导资金类型，同时判断资金的核心交易意图；
- **核心要求**：资金类型仅可输出「游资/量化/散户」三类，交易意图仅可输出「买入/卖出/T0交易」三类，严格匹配赛事指定分类；
- **输出文件**：`predict_result.csv`，固定4个字段，顺序不可修改：`stock_code`(股票代码)、`transaction_date`(交易日期)、`capital_type`(主导资金类型)、`capital_intention`(资金交易意图)。

### 三、任务数据与数据集构成

#### 数据整体规划

#### 原始数据字段构成

本次赛事提供的Level2原始数据包含**65个核心字段**，覆盖四大类核心数据维度：

[并没有, /data/raw里是我找到数据], 需要用这些数据转换成特征集.

#### 官方参考特征集

赛事官方提供了**7大类标准化参考特征集**，是本次赛题的核心特征依据，所有特征均基于当日盘中可获取数据计算，无未来函数风险：

1. **OSS大单分级特征**：超大/大/中/小单的金额、笔数占比，游资基准占比等；
2. **RS订单时序特征**：成交间隔变异系数、拆单相似度、订单爆发率、买卖间隔分化等；
3. **CB撤单系列特征**：快速撤单占比、买卖撤单分化、撤单间隔变异系数等；
4. **AP主动成交特征**：主动买卖占比、连续成交笔数、单边强度、净成交占比等；
5. **OBP盘口微观特征**：最优档位挂单、价差穿越、挂单偏移、盘口失衡度等；
6. **PD价格发现特征**：多维度价格冲击、盘口失衡、成交效率等；
7. **PI日内时段特征**：开盘30分钟/尾盘10分钟成交占比、赫芬达尔集中度、价格波动等。

### 四、任务规则及提交说明

#### 提交格式要求

- 所有结果文件需打包为 `submit.zip`压缩包，内部包含 `pattern_reco.csv`和 `predict_result.csv`两个文件，不得嵌套文件夹；
- 两个CSV文件的字段顺序、名称必须与赛事要求完全一致，不得修改、新增、删除字段；
- 资金类型仅可输出「游资/量化/散户」，交易意图仅可输出「买入/卖出/T0交易」；
- 所有结果需基于当日盘中可获取数据生成，无未来函数；
- 编码格式为UTF-8-sig，避免中文乱码，无空行、无缺失值。

#### 评分规则

A/B榜总得分 = 交易模式识别分 × 0.4 + 参与者识别分 × 0.6，其中：

1. Task1 交易模式识别（40%）：基于轮廓系数、CH指数、Wasserstein距离、DTW时序距离四大指标，评估聚类的类内聚合度与类间区分度；

- 轮廓系数：衡量类内聚合度与类间区分度，取值范围[-1,1]，越接近1效果越好；
- CH指数：衡量类间方差与类内方差的比值，值越高聚类效果越好；
- Wasserstein距离：衡量不同交易模式的分布差异，距离越大区分度越好；
- DTW时序距离：衡量不同交易行为的时间序列差异，距离越大区分度越好。

1. Task2 资金类型识别【参与者识别】（60%）：基于平台T+5日实盘回溯的真实资金标签，计算加权F1-Score，评估分类准确率。

- 加权F1-Score：基于平台T+5日实盘回溯的真实标签计算，是分类任务的核心评分指标，越接近1效果越好；
- 精确率：预测为某类资金的样本中，真实为该类的比例，衡量查准率；
- 召回率：真实为某类资金的样本中，被正确预测的比例，衡量查全率。

#### 合规红线要求

1. 严禁使用未来函数：所有特征、模型、规则仅可使用当日盘中可获取的Level2数据，不得使用当日收盘后、未来交易日的任何数据；
2. 严禁硬编码个股标签：不得针对特定个股、特定日期设置固定的标签规则，所有规则需具备普适性；
3. 严禁使用平台评测真值：平台T+5日回溯的真实资金标签仅用于线上评分，严禁导入模型训练、规则优化环节，违者取消比赛成绩；
4. 代码可复现性要求：B榜前15名队伍需提交完整项目代码，确保评审可完整复现结果，代码逻辑需与方案报告一致。

## 分析赛题、对问题进行建模

### 一、赛题核心分析

#### 赛事背景

本赛题是国内金融AI领域面向全量Level2高频数据的机构资金识别专项赛事，核心解决A股市场中机构资金行为隐蔽、普通投资者无法穿透的行业痛点。本赛题希望参赛者基于A股个股的逐笔委托、逐笔成交、逐笔撤单和十档盘口快照数据（提示：可通过淘宝、闲鱼、百度网盘等渠道自行获取相关数据），构建一套完整方法论体系，对游资/量化两类机构进行精细化的区分，识别各类参与者的买卖方向、买卖意图。整体方法的技术手段不做强制要求，鼓励参赛者结合大模型等AI技术和经典量化模型给出更好的解决方案。

#### 问题本质

本赛题的核心是**无监督场景下的金融时间序列数据挖掘与行为识别问题**，可拆解为两个相互耦合的子问题：

1. **Task1 无监督聚类问题**：无任何标注标签，基于高维的Level2时间序列特征，对单日个股的交易行为进行分簇，核心目标是最大化类内聚合度与类间区分度，同时为每个簇赋予可解释的业务含义；
2. **Task2 无真值规则判别问题**：无官方提供的训练标签，基于A股量化交易的行业常识与资金行为规律，构建多维度的量化打分规则，区分游资/量化两类资金，同时识别资金的交易意图，核心目标是最大化与实盘真实资金行为的匹配度。

问题的核心难点在于：

- 高频Level2数据维度高、噪声大，需要从海量数据中提取与机构交易行为强相关的有效特征；
- 机构交易行为具有极强的隐蔽性，需要穿透拆单、对倒、虚假挂撤单等操作，识别真实的资金属性与意图；
- 无任何标注标签，无法使用传统的有监督机器学习算法，所有方案需基于金融逻辑与无监督方法构建，同时保证可解释性。

#### Baseline技术建模框架

Baseline技术建模框架分为4个核心层级，从数据输入到结果输出形成完整闭环，全程无未来函数、无监督学习，完全贴合赛事要求：

```plain
原始Level2数据输入（逐笔成交/委托/撤单/十档盘口）
        ↓
数据预处理层：数据清洗、时间标准化、JSON盘口解析、异常值过滤、时序排序
        ↓
特征工程层：全量官方7大类参考特征提取 + 十档盘口衍生特征构建，按个股+交易日聚合为特征矩阵
        ↓
建模推理层
    ├ Task1：多维全特征KMeans聚类 → 8项关键特征统计 → 多条件联合匹配交易模式
    └ Task2：11维度归一化多因子打分 → 资金类型判定 + 盘口+主动成交联合规则识别交易意图
        ↓
结果输出层：格式校验、空值填充、编码标准化，生成赛事要求的两个CSV提交文件
```

### 二、任务分层拆解

Baseline完整开发流程可拆解为4个核心层级，每个层级有明确的目标、输入、输出与核心工作，形成完整的落地闭环：

#### ① 数据理解层

- **目标**：深入理解 Level-2 快照数据的结构和含义。
- **关键动作**：
- 熟悉 65 个字段的业务含义（价格、累计成交量/额、盘口 JSON、时间戳等）。
- 解析 `bids` 和 `asks` 的 JSON 数组结构，提取盘口深度、大单占比等信息。
- **关键发现 1**：volume/amount/transactions/bigordervolume 均为当日累计值（从开盘累加到当前快照），需 diff 得到逐笔量。
- **关键发现 2**：`date` 字段为 UTC 毫秒级时间戳，直接 `.dt.hour` 得到的是 UTC 小时（0-8），而 `hh` 字段才是北京时间小时（8-16）。必须使用 `hh` 做时段判定。
- **关键发现 3**：`bigordervolume`、`changepercent`、`rangepercent` 等字段直接有值可用，无需重新计算。

#### ② 特征工程层

- **目标**：从原始高频数据中提取对交易行为和资金流向有区分度的特征。
- **关键动作**：
- **累计值转逐笔**：`volume/amount/transactions/bigordervolume.diff()` 得到真正的逐笔成交量和成交额。
- **OSS 大单分级**：基于逐笔量按超大单/大单/中单/小单分级统计金额和笔数占比。
- **TRD 交易结构**：平均每笔交易量、交易量标准差、大单成交量占比、日内涨跌幅/振幅。
- **RS 时序特征**：成交间隔变异系数、拆单相似度、订单爆发率。
- **AP 主动成交**：基于价格变动判定主动买卖方向，统计主动买卖占比和连续笔数。
- **PI 日内时段**：开盘 30 分钟和尾盘 10 分钟的成交额占比（使用 `hh` 列的北京时间）。
- **PD 价格发现**：价格冲击指标、订单簿不平衡比率。
- **OBP 盘口特征（两套方案）**：
- 方案 A：从首条 bids/asks JSON 提取最优买卖价差（spread）、盘口失衡度（book_imbalance）、大单挂单占比。
- 方案 B：利用数据中已有的 `totalbidvolume`、`totalaskvolume`、`weightedbidprice`、`weightedaskprice` 字段计算全天的盘口失衡度统计量。

#### ③ 模型构建层

- **目标**：构建交易模式聚类和参与者识别的模型。
- **关键动作**：
- **Task 1（无监督聚类）**：KMeans 聚类（8 类），基于聚类中心的关键指标进行多条件联合匹配（≥3 个条件命中才生效），赋予 8 种语义解释。
- **Task 2（规则** **+** **打分）**：11 维度多因子加权打分，正向维度（大额/单边/冲击/时段集中）得分越高越像游资，反向维度得分越高越像量化。意图采用双源盘口失衡度（首条快照+全天均值）与主动成交占比联合规则。

#### ④ 结果输出层

- **目标**：按照赛题要求格式输出结果文件，并进行格式校验。
- **关键动作**：
- 生成 `pattern_reco.csv`：stock_code, transaction_date, pattern_type, pattern_explanation。
- 生成 `predict_result.csv`：stock_code, transaction_date, capital_type, capital_intention。
- 格式校验：字段顺序、合法值检查（capital_type 仅含游资/量化，intention 仅含买入/卖出/T0交易）。
- 打包为 `submit.zip` 提交。

### 三、开发建议

1. **从简单开始**：先跑通 Baseline，理解数据流和任务目标，再逐步优化特征和模型。
2. **重视特征工程**：高频数据的特征提取是核心，建议多尝试盘口微观结构特征（OFI、价差动态、大单冲击等）。
3. **注意累计值转逐笔**：volume/amount/transactions/bigordervolume 均为累计值，直接使用会导致 OSS 分类错误。
4. **注意时区问题**：`date` 字段是 UTC 时间戳，`hh` 字段是北京时间小时。时段判断必须使用 `hh`。
5. **伪标签/规则质量**：Task 2 没有真实标签，规则设计的金融逻辑合理性直接影响模型效果。
6. **代码可复现性**：固定随机种子，使用相对路径，代码注释充分，确保环境可复现。
7. **善用已有字段**：训练数据中很多字段（如 `changepercent`、`rangepercent`、`bigordervolume`）已经直接有值，直接使用比重新计算更准确。

### 四、解题思考过程

面对本赛题，完整的解题思考过程可分为以下6个核心步骤，从问题理解到方案落地形成完整闭环，适合快速上手，作为baseline方案的解题逻辑：

#### 步骤1：深度审题，明确核心要求与约束

首先，反复研读赛题文档，明确两个核心任务的边界、要求与合规约束：

1. 明确Task1是无监督聚类任务，无任何标注标签，核心是「类内聚合、类间区分」，输出需要有明确的业务解释；
2. 明确Task2是无真值的规则判别任务，无官方训练标签，核心是贴合A股资金行为规律，输出严格匹配赛事指定的分类；
3. 明确合规红线：严禁使用未来函数、严禁硬编码标签、严禁使用平台评测真值，所有结果需基于当日盘中数据生成；
4. 明确提交格式：两个CSV文件的字段顺序、名称、编码必须完全符合要求，不得修改。

#### 步骤2：数据探查，理解数据结构与业务含义

对赛事提供的样例训练数据进行全面的探查，理解数据的结构、字段含义、分布特征：

1. 查看原始数据的65个字段，区分基础信息、逐笔成交、盘口快照、行情统计四大类数据，理解每个字段的业务含义；
2. 重点探查bids/asks十档盘口的JSON格式，明确数据结构，设计JSON解析逻辑；
3. 查看数据的时间粒度、覆盖范围，明确「股票代码+交易日期」是最小的样本单元；
4. 探查核心指标的分布特征，比如成交量、成交额、大单占比的分布，识别异常值，为后续的数据清洗和特征提取提供依据。

#### 步骤3：核心问题拆解，方案选型

基于对赛题和数据的理解，将核心问题拆解为可落地的子问题，进行方案选型：

1. 数据预处理方案：针对高频Level2数据的特点，设计「读取→清洗→转换→JSON解析→排序」的标准化预处理流程，解决数据噪声、格式不统一的问题；
2. 特征工程方案：基于赛事官方提供的7大类参考特征集，设计「官方特征全量落地+盘口衍生特征补充」的特征体系，将高频时间序列数据转换为结构化的特征矩阵；
3. Task1聚类方案：针对无标签的场景，选择KMeans作为基础聚类算法，该算法成熟、稳定、可解释性强，适合作为Baseline方案；同时设计「多条件联合匹配」的模式命名逻辑，避免单一特征误判；
4. Task2分类方案：针对无真值的场景，放弃有监督机器学习算法，选择「多因子量化打分」的方案，基于A股行业常识构建规则，同时解决不同特征的量纲差异问题，确保方案的可解释性与合规性；
5. 意图识别方案：采用「主动成交+盘口失衡+资金类型」的联合规则，提升意图识别的准确率与金融逻辑合理性。

#### 步骤4：方案落地，全流程跑通

基于选型的方案，编写代码，先在样例训练集上跑通全流程，验证方案的可行性：

1. 先编写数据预处理代码，处理样例数据，验证JSON解析、格式转换的逻辑；
2. 编写特征提取代码，基于预处理后的数据，提取全量官方特征与盘口衍生特征，生成特征矩阵；
3. 编写Task1聚类代码，对特征矩阵进行聚类，统计簇画像，匹配交易模式，验证聚类的效果；
4. 编写Task2打分代码，计算多因子得分，判定资金类型与交易意图，验证规则的合理性；
5. 编写结果输出代码，生成符合赛事要求的CSV文件，验证格式的合规性。

#### 步骤5：效果评估，迭代优化

在全流程跑通后，基于离线评估指标，对方案进行迭代优化：

1. 聚类效果优化：调整聚类数、特征组合、距离度量方式，提升轮廓系数、CH指数，增强类间区分度和类内聚合度；
2. 分类规则优化：调整打分维度、权重配置、特征阈值，提升规则的金融逻辑合理性，避免错判；
3. 特征工程优化：基于特征的区分度，筛选有效特征，去除无效特征，同时新增更贴合业务逻辑的衍生特征；
4. 逻辑优化：优化交易模式的匹配规则、交易意图的判定规则，增强方案的可解释性与合理性。

#### 步骤6：方案整理，提交准备

完成方案优化后，整理完整的项目代码、方案报告，准备提交：

1. 代码整理：按照赛事要求，整理项目代码结构，设置清晰的模块划分，编写详细的代码注释，确保代码的可读性与可复现性；
2. 文档整理：编写完整的方案报告，说明方案设计、特征工程、模型逻辑、效果评估、优化方向等内容；
3. 提交文件校验：再次校验两个CSV文件的字段、格式、数据，确保符合赛事要求；
4. 打包提交：将所有文件打包为 `submit.zip`，完成线上提交。

## Baseline方案详解

### 一、完整代码

[main.py]

```plain
baseline方案架构：特征工程(8类56维) → KMeans聚类(Task1) + 多因子打分(Task2)

运行：python main.py | python main.py --input data.xlsx | python main.py --input "data/*.xlsx" -o out/

输出：pattern_reco.csv / predict_result.csv
```

### 二、Baseline概况

### 三、Baseline核心代码及解释

本Baseline方案的代码分为5个核心模块，分别对应数据处理、特征提取、模型训练、结果预测、离线评估，每个模块都有详细的代码和解释，完全适配赛事提供的训练数据。

#### 数据处理

核心代码位于 `load_and_preprocess()` 函数：

```plain
def load_and_preprocess():
    df = pd.read_excel(INPUT_PATH, engine='openpyxl')
    # 日期时间标准化
    # 注意：date 字段为 UTC 毫秒级时间戳，hh 字段为北京时间（UTC+8）的小时数
    df['transaction_date'] = df['dt'].astype(str)
    df['datetime'] = pd.to_datetime(df['date'], unit='ms')
    df['hour'] = df['hh']  # 北京时间小时，用于 PI 日内时段特征
    df['minute'] = df['datetime'].dt.minute  # 分钟，UTC 和北京时区一致
    df = df.rename(columns={'symbol': 'stock_code'})
    # 异常值过滤
    df = df[(df['price'] > 0) & (df['volume'] >= 0) & (df['amount'] >= 0)]
    # 时序排序
    df = df.sort_values(by=['stock_code', 'transaction_date', 'datetime'])
    return df
```

**重要说明**：

- `date` 字段为 UTC 毫秒级时间戳，通过 `pd.to_datetime(unit='ms')` 转换为 datetime（UTC 时间）。
- `hour`** 必须使用 `df['hh']`（北京时间小时），而非 `df['datetime'].dt.hour`（UTC 小时）**。这是训练数据最关键的时区陷阱——UTC 小时为 0-8，而北京交易时间为 9:30-15:00。如果直接用 UTC 小时做时段判断，开盘 30 分钟（9:30-10:00）和尾盘 10 分钟（14:50-15:00）的条件永远无法匹配，导致 PI 特征全为 0。
- `minute` 使用 `df['datetime'].dt.minute`，因为 UTC 和北京时间的分钟偏移量一致（都是整 8 小时）。
- 按股票代码、交易日、时间戳排序后，volume 单调递增，diff 可正确得到逐笔量。

#### 特征提取

**累计值转逐笔量（支持多个字段）**：

```plain
# volume/amount/transactions/bigordervolume 均为累计值，需 diff 得到逐笔量
group['tick_volume'] = group['volume'].diff().fillna(0).clip(lower=0)
group['tick_amount'] = group['amount'].diff().fillna(0).clip(lower=0)
group['tick_transactions'] = group['transactions'].diff().fillna(0).clip(lower=0)
if 'bigordervolume' in group.columns:
    group['tick_big_order_volume'] = group['bigordervolume'].diff().fillna(0).clip(lower=0)
```

经验证：按 `datetime` 排序后所有累计字段单调递增（0 个负 diff），diff 后可正确得到逐笔量。

**OSS 大单分级特征**（基于逐笔量的正确分类）：

```plain
# 阈值：超大单 ≥50000 股，大单 ≥10000 股，中单 ≥1000 股，小单 <1000 股
mega_mask = group['tick_volume'] >= 50000
large_mask = (group['tick_volume'] >= 10000) & (group['tick_volume'] < 50000)
mid_mask = (group['tick_volume'] >= 1000) & (group['tick_volume'] < 10000)
small_mask = group['tick_volume'] < 1000

feature['oss_mega_amount_pct'] = group.loc[mega_mask, 'tick_amount'].sum() / total_tick_amount
feature['oss_large_amount_pct'] = group.loc[large_mask, 'tick_amount'].sum() / total_tick_amount
# ...
```

**TRD 逐笔交易结构特征**：

```plain
# 平均每笔成交量/额
feature['trd_avg_trade_size'] = group['tick_volume'].sum() / (group['tick_transactions'].sum() + 1)
feature['trd_avg_trade_amount'] = group['tick_amount'].sum() / (group['tick_transactions'].sum() + 1)
# 每笔交易量标准差（反映交易规模离散度，标准差大 → 有大单混在小单中 → 游资特征）
valid_trades = group[group['tick_transactions'] > 0]
if len(valid_trades) > 0:
    trade_sizes = valid_trades['tick_volume'] / (valid_trades['tick_transactions'] + 1)
    feature['trd_trade_size_std'] = trade_sizes.std()

# 大单成交量占比（bigordervolume 是交易所定义的大单统计）
if 'tick_big_order_volume' in group.columns:
    feature['trd_big_order_ratio'] = group['tick_big_order_volume'].sum() / (total_tick_volume + 1e-8)

# 日内涨跌幅和振幅（直接使用已有字段，比重新计算更准确）
feature['trd_change_percent'] = group['changepercent'].iloc[-1]
feature['trd_range_percent'] = group['rangepercent'].max()
```

**AP 主动成交特征**（基于价格变动判定主动买卖）：

```plain
group['price_change'] = group['price'].diff()
active_buy_amt = group.loc[group['price_change'] > 0, 'tick_amount'].sum()
active_sell_amt = group.loc[group['price_change'] < 0, 'tick_amount'].sum()
feature['ap_active_buy_pct'] = active_buy_amt / (active_buy_amt + active_sell_amt)
feature['ap_active_sell_pct'] = active_sell_amt / (active_buy_amt + active_sell_amt)
```

**PI 日内时段特征**（使用北京时间 `hh` 列）：

```plain
# 开盘 30 分钟（9:30-10:00）
open_30min = group[
    ((group['hour'] == 9) & (group['minute'] >= 30)) |
    ((group['hour'] == 10) & (group['minute'] == 0))
]
# 尾盘 10 分钟（14:50-15:00）
close_10min = group[(group['hour'] == 14) & (group['minute'] >= 50)]
feature['pi_open_30min_amount_pct'] = open_30min['tick_amount'].sum() / total_tick_amount
feature['pi_close_10min_amount_pct'] = close_10min['tick_amount'].sum() / total_tick_amount
feature['pi_time_concentration'] = feature['pi_open_30min_amount_pct'] + feature['pi_close_10min_amount_pct']
```

**OBP 盘口特征（AB两套方案互补）**：

```plain
# 方案 A：从首条 bids/asks JSON 提取精确档位盘口特征
book_feature = get_book_feat(group['bids'].iloc[0], group['asks'].iloc[0])
feature.update(book_feature)

# 方案 B：利用 totalbidvolume/totalaskvolume 计算全天盘口统计量
total_bid = group['totalbidvolume'].values
total_ask = group['totalaskvolume'].values
total_all = total_bid + total_ask + 1e-8
imbalance_series = (total_bid - total_ask) / total_all

feature['obp_imbalance_mean'] = np.nanmean(imbalance_series)
feature['obp_imbalance_std'] = np.nanstd(imbalance_series)
feature['obp_imbalance_max'] = np.nanmax(imbalance_series)
feature['obp_imbalance_min'] = np.nanmin(imbalance_series)

# 加权买卖价差（从 weightedbidprice / weightedaskprice 计算）
feature['obp_weighted_spread_mean'] = np.nanmean(w_ask - w_bid)
feature['obp_weighted_spread_std'] = np.nanstd(w_ask - w_bid)

# 委买/委卖总量均值及比值
feature['obp_total_bid_mean'] = np.nanmean(total_bid)
feature['obp_total_ask_mean'] = np.nanmean(total_ask)
feature['obp_bid_ask_ratio'] = np.nanmean(total_bid) / (np.nanmean(total_ask) + 1e-8)
```

#### 模型训练

**Task 1 — 交易模式聚类（KMeans** **+** **多条件联合匹配）**

```plain
# 动态调整聚类数：n_samples 不足时降级
n_clusters_actual = min(N_CLUSTERS, n_samples)

kmeans = KMeans(n_clusters=n_clusters_actual, random_state=RANDOM_SEED, n_init=10)
df_feature['cluster_id'] = kmeans.fit_predict(X_scaled)

# 多条件联合匹配（≥3 个条件命中才生效）
for cluster_id in range(n_clusters_actual):
    profile = cluster_profile.loc[cluster_id]
    scores = {name: 0 for name in pattern_names}

    # 游资强势连板拉升：超大单>12% + 买盘失衡>0.2 + 主动买入>55% + 时段集中>30%
    if profile.get('oss_mega_amount_pct', 0) > 0.12: scores['游资强势连板拉升'] += 1
    if profile.get('book_imbalance', 0) > 0.2: scores['游资强势连板拉升'] += 1
    if profile.get('ap_active_buy_pct', 0) > 0.55: scores['游资强势连板拉升'] += 1
    if profile.get('pi_time_concentration', 0) > 0.3: scores['游资强势连板拉升'] += 1
    # ... 8 种模式共 30+ 条件

    if max(scores.values()) >= 3:
        pattern_name = max(scores, key=scores.get)
    else:
        pattern_name = '机构长线配置'  # 默认兜底
```

**Task 2 — 参与者识别（11 维多因子打分）**

```plain
# 11 个打分维度，每个维度包含对应特征
dim_list = [
    ['oss_mega_amount_pct', 'oss_large_amount_pct'],  # 1. OSS大额成交
    ['rs_split_similarity', 'rs_burst_ratio'],         # 2. RS拆单时序
    ['cb_fast_cancel_ratio', 'cb_buy_cancel_ratio'],   # 3. CB撤单分化
    ['ap_active_buy_pct', 'ap_active_net_pct'],        # 4. AP主动单边
    ['spread', 'book_imbalance'],                      # 5. OBP盘口
    ['pd_impact', 'pd_Q1_ratio'],                      # 6. PD价格冲击
    ['pi_time_concentration', 'pi_price_std_pct'],     # 7. PI时段波动
    ['ap_active_buy_run_max'],                         # 8. 连续买入
    ['big_bid_ratio', 'big_ask_ratio'],                # 9. 盘口大单
    ['cb_sell_cancel_ratio'],                          # 10. 卖出撤单
    ['ap_unilateral_intensity'],                       # 11. 单边强度
]

# 游资倾向维度索引：值越大越像游资（大额、单边、冲击、时段集中）
yz_like_dims = {0, 3, 5, 6}

# 跨样本全局 MinMax 归一化后加权打分
# 游资倾向维度：dim_score 越大 → 游资分越高，量化分 = 1-dim_score
# 量化倾向维度：dim_score 越大 → 游资分越低，量化分 = dim_score
for dim_idx_local, dim_cols in enumerate(valid_dim_list):
    dim_score = np.mean([row[c] for c in dim_cols])
    if orig_dim_idx in yz_like_dims:
        score_yz += dim_score * weight_yz[dim_idx_local]
        score_qt += (1 - dim_score) * weight_qt[dim_idx_local]
    else:
        score_yz += (1 - dim_score) * weight_yz[dim_idx_local]
        score_qt += dim_score * weight_qt[dim_idx_local]

# 二分类：得分高者胜出
return '游资' if score_yz >= score_qt else '量化'
```

#### 结果预测

**意图判定（双源信号联合：首条快照** **+** **全天均值）**：

```plain
def get_intention(row):
    buy_pct = row.get('ap_active_buy_pct', 0.5)
    sell_pct = row.get('ap_active_sell_pct', 0.5)
    # 双源盘口失衡：首条快照 + 全天均值
    imbalance_snap = row.get('book_imbalance', 0)
    imbalance_mean = row.get('obp_imbalance_mean', 0)
    # 综合失衡度：首条快照（即时信号）加权 0.4，全天均值（稳定信号）加权 0.6
    imbalance = 0.4 * imbalance_snap + 0.6 * imbalance_mean

    # 买入：主动买入占比 > 60% 且 盘口综合买盘失衡 > 0.08
    if buy_pct > 0.6 and imbalance > 0.08:
        return '买入'
    # 卖出：主动卖出占比 > 60% 且 盘口综合卖盘失衡 < -0.08
    elif sell_pct > 0.6 and imbalance < -0.08:
        return '卖出'
    # T0交易：其余情况（含多空均衡、信号不明确）
    else:
        return 'T0交易'
```

**设计说明**：本方案中，意图判定采用首条快照（权重 0.4）+ 全天均值（权重 0.6）的加权综合分母，比单一快照信号更稳定地反映全天多空力量对比。

#### 离线评估

```plain
# Task 1: 轮廓系数 + CH指数 + DB指数（三项指标综合评估）
if n_clusters_actual > 1:
    sil_score = silhouette_score(X_scaled, df_feature['cluster_id'])
    ch_score = calinski_harabasz_score(X_scaled, df_feature['cluster_id'])
    db_score = davies_bouldin_score(X_scaled, df_feature['cluster_id'])
    print(f"轮廓系数: {sil_score:.4f}（越接近 1 越好）")
    print(f"CH指数: {ch_score:.4f}（越高越好）")
    print(f"DB指数: {db_score:.4f}（越低越好）")

# Task 2: 分类分布统计
print(f"游资占比: {(df_result['capital_type'] == '游资').sum() / len(df_result):.2%}")
print(f"量化占比: {(df_result['capital_type'] == '量化').sum() / len(df_result):.2%}")

# 格式合法性校验
assert df_result['capital_type'].isin(['游资', '量化']).all()
assert df_result['capital_intention'].isin(['买入', '卖出', 'T0交易']).all()
```

### 四、Baseline 运行说明

#### 环境要求

```plain
Python >= 3.8
pandas, numpy, scikit-learn, openpyxl
```

#### 安装依赖

```plain
python3 -m pip install pandas numpy scikit-learn openpyxl
```

#### 运行方式

```plain
python main.py
```

#### 预期输出

```plain
【1/5】数据预处理开始
预处理完成 | 有效数据行数: 4937 | 覆盖股票数: 1
【2/5】全量特征提取开始
特征提取完成 | 总特征数: 52 维 | 总样本数: 1
【3/5】Task1 交易模式聚类开始
>>> 样本数 (1) < 预设聚类数 (8)，动态调整为 1
===== 各簇关键特征画像 =====
            oss_mega_amount_pct  ...  pi_time_concentration  pd_impact
cluster_id                        ...
0                          0.51  ...                  0.318     13.096
Task1 聚类完成 | 样本数不足，仅生成 1 个聚类
交易模式分布:
pattern_type
游资强势连板拉升    1
【4/5】Task2 资金与意图识别开始
Task2 识别完成
资金类型分布:
capital_type
游资    1
交易意图分布:
capital_intention
T0交易    1
【5/5】结果保存与离线评估开始
所有流程完成！
```

**说明**：训练数据仅 1 只股票 × 1 天，样本数为 1，聚类数自动降级为 1。A/B 榜有 100 只股票 × 多天，聚类数将为 8，各项评估指标可正常计算。

#### 输出文件

```plain
pattern_reco.csv    → stock_code, transaction_date, pattern_type, pattern_explanation
predict_result.csv  → stock_code, transaction_date, capital_type, capital_intention
```

### 五、baseline方案思路

本 Baseline 的设计思路是 **"特征工程先行，规则后模型，先简单后复杂"**：

1. **累计值转逐笔**：volume/amount/transactions/bigordervolume 均为累计值，必须 diff 后才能正确进行 OSS 大单分类和成交金额统计。
2. **时区处理**：`date` 字段为 UTC 时间戳，必须使用 `hh` 列（北京时间）进行时段判定，否则 PI 特征全为 0。
3. **特征工程覆盖 8 大类**：参照赛题官方参考特征集，从快照数据中尽可能还原 OSS/TRD/RS/CB/AP/PI/PD/OBP 八类特征。同时利用数据中已有的 `totalbidvolume`、`totalaskvolume`、`weightedbidprice`、`weightedaskprice`、`changepercent`、`rangepercent`、`bigordervolume` 等字段补充特征。
4. **规则启发**：利用金融领域先验知识（游资激进、量化稳健）设计打分规则，保证金融逻辑合理性。
5. **多条件联合匹配**：Task 1 的模式匹配采用 ≥3 个条件命中才生效的策略，避免单一特征误判。
6. **双源信号联合**：Task 2 的意图判定采用首条快照 + 全天均值的加权综合分母，比单一快照信号更稳定。
7. **边界处理鲁棒**：支持少样本（聚类数动态降级）、缺失值填充、无穷大替换、格式合法性校验。

### 六、baseline核心逻辑

1. **数据安全逻辑**：所有特征、规则、模型均基于当日盘中可获取数据，无未来函数、无跨日数据、无硬编码标签，完全符合赛事合规要求；
2. **Task1核心逻辑**：数学聚类→全维度簇画像→多条件联合匹配8类交易模式，确保聚类结果的区分度与业务可解释性；
3. **Task2核心逻辑**：11维度归一化多因子打分，游资侧重大额成交、单边买入、盘口买盘失衡，量化侧重小单高频、拆单均匀、窄价差、多空均衡；
4. **双任务联动逻辑**：Task1的聚类结果可反向校验Task2的资金类型判定，同簇内资金类型差异过大时，可微调打分权重，提升整体效果；
5. ###### **可解释性逻辑**：所有特征、规则、模式都有对应的金融逻辑解释，无黑箱模型，符合金融场景的可解释性要求，提升方案的评审得分。
