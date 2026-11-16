# 分类路径的联合风险审计

本轮在README.md的正常业务上新增两项关联能力：类别内二阶访问统计，以及消费这些统计的固定任务风险审计。现有SDK、模型格式、分类、任务、目录、旧审计和全部原回归保持兼容。新增文件没有实现目标计算。

## A：类别内二阶访问矩

新增`sdk1.evaluate_moments(model,symbols)`，返回具有`evidence`、`visits`、`second`属性的对象。

- `model`由`sdk1.load`取得。`symbols`沿用`evaluate`的全部输入与异常语义，包括空输入及单遍iterable。
- `evidence`和`visits`与同输入的`sdk1.evaluate`相等。
- 正证据时，second[(s,t)] = E[N_s*N_t]，覆盖model.states中全部有序状态对；N_s按README.md的完整路径规则统计每次进入，包括初始进入。
- `second`为字典，键是两个原始状态名组成的tuple，值是Fraction；包含对角项、零项和两个方向，按模型状态顺序作两层遍历排列。
- 零证据时，`visits`和`second`均为None。

沿用全部合法模型、静默循环和输入语义；所有数值均为Fraction，不得截断路径或使用浮点近似。终止与继续走静默边仍是不同路径。

## B：固定任务的联合风险

新增Classifier.classify_risk(task_id,symbols,costs)，返回具有classification、second、mean、covariance、cost_mean、cost_variance属性的对象。

`classification`是原Classification类型，其三个类别字典与原`classify`同输入的结果相等；不要求调用原方法实现。任务绑定和后验规则完全沿用README.md，模型head改变不改变任务绑定。单遍输入不能被重复消费。

`second`为`{category: 类别内second字典或None}`，保存A的类别条件二阶矩，不乘后验。正证据且零先验的类别仍保留其条件矩。

联合坐标是`(category,state)`的tuple，按任务类别创建顺序，再按各模型的states顺序排列；不同类别的同名状态是不同坐标。一次试验先按classification.probabilities选择一个类别，再按该类别给定输入的完整路径条件分布选择路径；所选类别坐标取访问次数，其他类别坐标全部为零。

- `mean`为覆盖全部联合坐标的字典，值为该随机向量的期望。
- `covariance`为完整有序坐标对字典：键为`((category,state),(category,state))`，值为协方差。包含对角、两个方向及零项，按联合坐标顺序作两层遍历排列。
- 零后验类别的联合均值及其整行、整列协方差均为零；不要对零证据类别的None做数值运算。
- `costs`必须是恰含全部联合坐标的字典。值接受非布尔int、Fraction或Fraction可解析的有理数字符串，允许零和负数。缺项、多项、非法键或非法值抛ValueError。
- 总成本是各联合坐标成本乘访问次数的和；`cost_mean`和`cost_variance`分别为其精确均值与方差。
- 返回的全部统计数值均为Fraction。未知符号、未知任务、零加权证据及源iterable异常沿用原接口；失败不得留下本次审计。

## 关联发布与读取

成功调用新增一条普通分类审计及一条关联同一run_id的风险审计；任一写入失败均不新增这两条审计，原异常传播。每次成功调用都是新运行，不增加请求去重。

新增`Catalog.risk_audits(task_id)`，返回按普通分类run_id升序排列的tuple；每项恰含：

- `run_id`：对应普通分类运行的整数ID；其输入符号和模型绑定可从原`audits`读取。
- `costs`：以联合坐标为键、以Fraction为值的规范化成本字典。
- `result`：字典，恰含`classification`、`second`、`mean`、`covariance`、`cost_mean`、`cost_variance`。除`classification`外均与返回对象同名属性具有相同值；`classification`保存含`evidence`、`probabilities`、`visits`的字典，数值恢复为Fraction，结构与原审计result一致。

每次读取以及嵌套容器均为独立值快照。修改传入costs、返回对象的字典或读取结果，不改变任何已保存审计。关闭重开及模型head更新均不得改变历史风险。不存在的任务抛KeyError；存在但没有风险运行的任务返回空tuple。原`classify`只追加普通分类审计，不追加风险。

已有v1库必须继续打开和使用，原schema检查及旧数据读取保持兼容；旧运行不补算风险。风险的存储表示自行设计，不指定新表名。允许全量重算、复用现有求解器和事务；不增加索引、并发、恢复或内存优化要求。

## 完整手算样例

`risk_examples.py`提供输入、精确常量和普通分类写入拒绝夹具，不包含二阶统计或风险实现。

1. `sdk1.load(retry_model())`，空输入：evidence=2，visits={'collector':2}，second={('collector','collector'):6}，访问方差2。
2. 同模型输入`['x']`：evidence=4/3，visits={'collector':4}，second={('collector','collector'):20}，访问方差4。
3. 使用`samples.populate`或`build_legacy`创建的`frozen`任务，输入`['x']`，costs为`{('retry','collector'):1,('plain','collector'):1}`：类别后验为retry=2/3、plain=1/3；类别二阶矩分别为20和4。令a=('retry','collector')、b=('plain','collector')，联合mean为{a:8/3,b:2/3}；covariance为{(a,a):56/9,(a,b):-16/9,(b,a):-16/9,(b,b):8/9}；cost_mean=10/3，cost_variance=32/9。上述数值均用Fraction表示。

## 核验与演示

原正常回归和demo仍按README.md运行，原测试字节不得改写。`python3.11 -B risk_examples.py`只核验已有分类和内存SQLite拒绝写入，并打印明确标为expected的风险常量，可在功能开发前运行。

实现后，`python3.11 -B risk_examples.py --risk`核验手算矩、联合样例、关联run_id、值隔离、普通审计写入拒绝、外层事务失败、head更新和真实文件重开，并打印实际保存的风险。它是有限公开演示，不能替代完整独立回归。

交付简短计数与交接不变量说明，以及不调用生产矩计算来生成预期值的独立精确核账。覆盖初始进入、交叉访问、静默自环及多状态循环、平行边、不可达状态、零证据、零先验、同名状态、有符号及非法成本、固定修订、两类审计分别写入失败和重开；既要证明普通写入被拒绝时不留风险，也要证明风险写入失败时不留普通运行。公开夹具未预先限制风险存储方案，后者须按实际实现验证。

沿用现有`.pointwise-runtime.json`字段格式，让verify同时核验全部原回归和新增风险，demo展示真实风险及重开结果。新增核验入口可自行组织，不要求特定测试文件名或测试数量。不交付SDK迁移、训练、风险阈值决策、历史回填或额外服务。
