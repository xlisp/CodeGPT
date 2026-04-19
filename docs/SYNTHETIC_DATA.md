# 合成数据：怎么把一堆垃圾代码变成高质量训练数据

> 用户问题：像 Claude 这样的大模型是怎么把**垃圾的代码数据变废为宝**的？从低质量的原始语料，最终转化成 SFT / RL 都能吃的高质量训练数据——具体走哪几步？哪里该用专家系统（linter、类型检查、测试）？哪里该蒸馏优秀软件的输入输出？哪里合成才是正解？以及——**什么时候必须上人工标注**，才能把总成本压到最低？
>
> 这篇文档回答四件事：
>
> 1. **为什么原始代码语料本质上就是"垃圾"**——把"垃圾"拆成四种具体故障，才能对症下药。
> 2. **变废为宝的谱系**：过滤（丢）→ 修复（改）→ 合成（造）→ 验证（验收）。每一步用什么工具，代价多少。
> 3. **SFT 要什么数据、RL 要什么数据**——两个阶段对"高质量"的定义根本不一样。
> 4. **人工标注的"最划算点"**：不是多就好，是**打在刀刃上**——种子、分歧、边界、最终验收这四个点，其他位置花钱都是浪费。

---

## 目录

1. [先定义"垃圾"：原始代码语料的四种故障](#1-先定义垃圾原始代码语料的四种故障)
2. [总图：从脏语料到训练批次的六个环节](#2-总图从脏语料到训练批次的六个环节)
3. [环节 1——过滤：专家系统是免费的标注员](#3-环节-1过滤专家系统是免费的标注员)
4. [环节 2——修复：linter / formatter / typer 的"可微分版本"](#4-环节-2修复linter--formatter--typer-的可微分版本)
5. [环节 3——执行反馈：interpreter 是最诚实的 reward](#5-环节-3执行反馈interpreter-是最诚实的-reward)
6. [环节 4——从强模型蒸馏：input/output 到思维链](#6-环节-4从强模型蒸馏inputoutput-到思维链)
7. [环节 5——Self-Instruct / Evol-Instruct：用 LLM 造 LLM 的指令数据](#7-环节-5self-instruct--evol-instruct用-llm-造-llm-的指令数据)
8. [环节 6——Self-Refine / Critique：模型给自己打分](#8-环节-6self-refine--critique模型给自己打分)
9. [SFT 要什么样的合成数据](#9-sft-要什么样的合成数据)
10. [RL 要什么样的合成数据（和 SFT 完全不同）](#10-rl-要什么样的合成数据和-sft-完全不同)
11. [什么时候必须上人工标注：四个"最划算点"](#11-什么时候必须上人工标注四个最划算点)
12. [回到 CodeGPT：一条可落地的合成数据管线](#12-回到-codegpt一条可落地的合成数据管线)
13. [小结：一张数据-阶段-成本表](#13-小结一张数据-阶段-成本表)

---

## 1. 先定义"垃圾"：原始代码语料的四种故障

先看 CodeGPT 的数据入口 `data/python_code/prepare.py` 做了什么——它的 `collect_python_files` 只做了最朴素的筛选：跳过 `__pycache__`、限制文件大小、验证 UTF-8。这只是把**完全不能用**的东西挡掉，里面留下来的代码**仍然绝大多数是"垃圾"**。"垃圾"不是一个模糊的情绪词，具体是四种故障：

| 故障类型 | 典型表现 | 对训练的破坏 |
|---|---|---|
| **F1 语法级错误** | `SyntaxError`、不完整片段、半截文件 | 模型学会生成不能解析的代码 |
| **F2 语义级错误** | 类型不匹配、未定义变量、死循环、空指针 | 模型学会生成"看起来对"但跑不通的代码 |
| **F3 风格/规范噪声** | 无意义变量名（`a1`, `tmp2`）、没注释、缩进混乱、`print` 调试残留 | 模型学会"丑陋但能跑"的代码 |
| **F4 任务无关** | 自动生成的 stub、vendored 第三方代码、minified 前端 bundle、重复 boilerplate | 模型浪费参数记忆这些，挤压真正有价值知识的容量 |

**关键认知**：这四种故障需要的工具完全不同。F1 用 AST 解析器 5 分钟搞定；F2 需要执行环境或类型检查器；F3 需要风格模型或强模型蒸馏；F4 需要去重 + 聚类。**一刀切没有用**，必须按故障类型走不同管线。

预训练阶段可以容忍一些 F3 和 F4（规模本身就是正则），但 SFT / RL 阶段必须把四种全部清掉——因为 SFT 的每个样本都会被**完整拟合**，一个垃圾样本就是一颗定时炸弹。

---

## 2. 总图：从脏语料到训练批次的六个环节

```
原始语料 (GitHub, pip packages, 客服对话, 论坛帖子……)
   │
   │  [环节 1] 过滤：规则 + 专家系统
   ▼
干净语料 (AST 能解析、文件大小合理、去重、licence 合规)
   │
   │  [环节 2] 修复：auto-format / auto-fix / type-inference
   ▼
标准化语料 (black 化、isort 化、类型标注补全)
   │
   │  [环节 3] 执行验证：interpreter / test runner
   ▼
带执行标签的语料 (哪些能跑、输入输出是什么、覆盖哪些分支)
   │
   │  [环节 4/5] 蒸馏 + Self-Instruct：用强模型造 (prompt, response)
   ▼
(prompt, response) 对 (SFT 能直接吃的格式)
   │
   │  [环节 6] Critique + 人工抽检
   ▼
(prompt, chosen, rejected) 三元组 (RL / DPO 能吃的格式)
```

这张图的核心洞察是：**"合成"不是一步操作，是一条流水线**。每一步都在做一件非常具体的事，而且有明确的**工具**和**失败模式**。下面逐环节拆。

---

## 3. 环节 1——过滤：专家系统是免费的标注员

第一件事不是造新数据，是**扔掉脏数据**。这里的"专家系统"不是 AI 术语意义上的 expert system，而是**几十年编译器/工具链社区积累下来的确定性检查器**。对一个 Python 文件，你可以几乎**免费**拿到以下标签：

```python
import ast
import tokenize
from io import StringIO

def quality_score(src: str) -> dict:
    labels = {"parseable": False, "has_docstring": False, "complexity": 0}
    try:
        tree = ast.parse(src)                 # F1 过滤器
        labels["parseable"] = True
    except SyntaxError:
        return labels                         # 解析失败直接扔

    # 有没有文档字符串？（F3 信号）
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if ast.get_docstring(node):
                labels["has_docstring"] = True
                break

    # 圈复杂度（F2 信号：过高说明可能是混淆/自动生成的）
    labels["complexity"] = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try))
    )
    return labels
```

这段代码在一台机器上每秒能处理几万个文件，而它提供的标签和**花几百万美金请标注员打出来的标签一样可靠**——因为它们是**确定性的**。同类工具谱系：

| 工具 | 捕捉的故障 | 代价 |
|---|---|---|
| `ast.parse` / `tree-sitter` | F1 语法 | 几乎为零 |
| `pyflakes` / `ruff` | 未使用 import、未定义名字 | 每文件 <10ms |
| `mypy` / `pyright` | F2 类型错误 | 每文件 50–500ms |
| `bandit` | 安全漏洞（`eval`、shell injection） | 每文件 <50ms |
| `radon` | 圈复杂度、可维护性指数 | 每文件 <10ms |
| `simhash` / MinHash | F4 近似去重 | 百万文件级聚类 |

**实践建议**：过滤阶段**不要用 LLM**。LLM 贵、慢、还不如一个 `ast.parse` 可靠。LLM 在这里唯一有用的位置是**最后一步**——用小模型给留下来的文件打一个 0-5 的"教育价值分"，和 phi-1 论文里用的 "textbook quality" 分类器是同一个套路。但主力过滤必须是规则。

> **过滤阶段的铁律：能用 AST 解决的，就别用 Transformer。**

---

## 4. 环节 2——修复：linter / formatter / typer 的"可微分版本"

过滤完之后，剩下的代码**仍然不是最优**——变量命名差、格式乱、缺类型标注。这些不严重到扔掉，但会污染 SFT。修复环节的核心思想是：

**如果有一个确定性的工具能把 A 改成更好的 A'，就用工具，不要用模型。**

```python
import subprocess

def standardize(src: str) -> str:
    # black 是格式上的确定函数
    src = subprocess.run(["black", "-"], input=src, capture_output=True, text=True).stdout
    # isort 对 imports 也是确定函数
    src = subprocess.run(["isort", "-"], input=src, capture_output=True, text=True).stdout
    # ruff --fix 能修一批琐碎问题（移除未用变量等）
    src = subprocess.run(["ruff", "check", "--fix", "-"], input=src, capture_output=True, text=True).stdout
    return src
```

经过这一层，你等于免费获得了**标准化的训练目标**——所有样本都服从同一套格式约定，模型学习的噪声就小了一个数量级。

**需要模型介入的修复**：只有当规则工具做不到时才上 LLM。典型场景：

- **补充 docstring**：用强模型为每个函数生成文档字符串，作为同一函数的"改进版本"。
- **重命名无意义变量**：把 `a1, tmp2, x` 改成有语义的名字。这需要理解上下文，规则做不到。
- **把 `print` 调试改成 `logging`**：语义级重构。

这一层的输出一般是**成对样本**：`(脏版本, 干净版本)`。这对后面做 RL（脏版本是 rejected，干净版本是 chosen）非常有价值，属于"顺手就能拿到的偏好数据"。

---

## 5. 环节 3——执行反馈：interpreter 是最诚实的 reward

**代码数据相比文本数据最大的红利**：它能跑。一个 Python 解释器给出的 "`pass` / `fail` / 输出是什么" 是**比任何人类标注员都精确的信号**——没有标注分歧、没有主观偏差、没有疲劳。

最基本的用法是：

```python
import subprocess, tempfile, json

def execute_and_label(src: str, test_cases: list) -> dict:
    """对每个 test case 记录函数的实际输出"""
    results = []
    for tc in test_cases:
        script = f"{src}\n\nimport json\nprint(json.dumps({tc['call']}))"
        try:
            out = subprocess.run(
                ["python", "-c", script],
                capture_output=True, text=True, timeout=5,
            )
            results.append({
                "input": tc,
                "stdout": out.stdout.strip(),
                "stderr": out.stderr.strip(),
                "passed": out.returncode == 0,
            })
        except subprocess.TimeoutExpired:
            results.append({"input": tc, "passed": False, "error": "timeout"})
    return {"src": src, "exec_results": results}
```

这段代码提供了**四种高价值的训练信号**：

1. **绝对的对错**：`passed` 字段就是 reward。RL 里的 verifiable reward 几乎全部来自这里。
2. **输入输出示例**：`(input, stdout)` 对本身可以反向变成 SFT 数据——"给我这个输入，写一个函数输出这个"。
3. **异常种类**：`TypeError`、`IndexError` 各占多少比例——这是模型弱点的直方图，指导下一轮数据配比。
4. **行为等价聚类**：同一个 test case 下 stdout 相同的多个函数可视为"行为等价"，能自动产出语义等价的改写对。

### 5.1 从执行结果"倒推"训练数据

这是代码数据比文本数据本质上更强的地方。给定一个能跑的函数 `f(x)`，可以**机器生成**一整套变体：

```python
# 原始函数
def add(a, b):
    return a + b

# 基于执行结果自动合成的训练样本：
# 样本1：docstring 补齐 SFT
#   prompt:  "def add(a, b):\n    "
#   target:  '"""Add two numbers.\n\n    >>> add(1, 2)\n    3\n    """\n    return a + b'
#
# 样本2：测试用例生成 SFT
#   prompt:  "# Write tests for:\ndef add(a, b):\n    return a + b"
#   target:  "def test_add():\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0"
#
# 样本3：FIM 训练（项目已有）
#   prompt:  "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n    return result<|fim_middle|>"
#   target:  "result = a + b"
```

一个函数能派生出 5–10 条训练样本，**而这些样本里每一条都经过了解释器验证**。这就是 phi 系列和 Code Llama 的关键数据来源。

> **执行反馈的铁律：解释器能验证的，就不要花钱标注。**

### 5.2 与 CodeGPT 的连接

CodeGPT 现有的 FIM 训练（`tokenizer.py:148` 的 `apply_fim_transform`）已经是合成数据的一个实例——它把普通代码切成前缀/中间/后缀三段，人造出一个"给你前后文，补中间"的任务。这是**在数据层合成任务**的经典案例：原始语料里没有"补全"这个标签，但通过切分 + 特殊 token 拼接，就凭空变出了一个新的训练目标，且完全不需要人工标注。

---

## 6. 环节 4——从强模型蒸馏：input/output 到思维链

执行反馈能解决"能跑"的问题，但解决不了"写得好"的问题——一个能跑的函数可能风格很烂、命名很差、没有文档。这时就需要**从一个已经"写得好"的强模型（比如 Claude / GPT-4 级别）蒸馏出训练数据**。蒸馏的几种层次：

### 6.1 Output 蒸馏（最便宜）

给强模型一堆 prompt，收集它的 response，当作 SFT 数据。Alpaca / Vicuna 就是这么做出来的。代价：一次调用强模型 API 的钱。

```python
# 伪代码
for prompt in prompt_pool:
    response = call_strong_model(prompt)
    sft_dataset.append({"prompt": prompt, "response": response})
```

**陷阱**：强模型也会出错。如果不加验证，会把强模型的系统性 bias 原样继承给小模型。**修复**：叠加环节 3 的执行验证——只保留**能跑通**的 response。

### 6.2 Input/Output 对齐蒸馏

已有一段好代码 `f(x)`（来自优秀开源项目），用强模型为它**反推 prompt**："如果我要让模型写出这段代码，应该问什么？"

```python
def reverse_engineer_prompt(good_code: str) -> str:
    return call_strong_model(
        f"下面是一段优秀的代码。请写出一个用户可能提出的、能恰好得到这段代码作为答案的需求描述：\n\n{good_code}"
    )
```

这等于**把优秀开源项目的 codebase 反向变成了一个 SFT 数据集**。CodeGPT 项目里 `data/python_code/prepare.py` 已经在收集本地 Python 项目，加上这一层就能从"纯代码预训练"升级到"指令对齐"。

### 6.3 思维链蒸馏（最贵，最有用）

不只要求强模型给答案，还要求它**把推理过程写出来**。得到的是形如 `<prompt, reasoning, answer>` 的三元组，小模型在 SFT 时被迫学会模仿推理链。这是 o1、DeepSeek-R1、Claude Extended Thinking 的数据路径。

关键点：**推理链也要验证**——不能让强模型的错误推理链被原样蒸馏过来。验证方式：让推理链的最终 answer 过一遍 test case（环节 3），只保留 pass 的样本。这叫 **rejection sampling**，是把强模型 + 解释器串成一个"过滤式生成器"的标准做法。

---

## 7. 环节 5——Self-Instruct / Evol-Instruct：用 LLM 造 LLM 的指令数据

蒸馏依赖 prompt 池。**如果一开始连 prompt 都没有呢？**比如你要训一个代码助手，但手里只有代码、没有"用户问题"。这时候用 **Self-Instruct**：

```
种子：100 条人工写的指令 (人工标注，但只要 100 条)
    │
    ▼
让强模型"扩写"：给你看 3 条种子，再写 10 条风格类似但主题不同的
    │
    ▼
过滤：去重、去有害、去歧义
    │
    ▼
扩充后的指令池 (10,000 条)
    │
    ▼
为每条指令调用强模型生成 response
    │
    ▼
SFT 数据集
```

**Evol-Instruct** 是进一步迭代：对每条已有的指令，让 LLM 把它"进化"成更难、更复杂、更细节的版本。经过几轮进化，数据的难度分布会显著变宽——而训练集的难度分布宽度是模型能力上限的直接决定因素。

### 7.1 对代码领域的特化

对 CodeGPT 这类代码模型，Self-Instruct 的循环可以绑定到执行反馈：

```python
for iter in range(N):
    # 1. 用 LLM 生成一个代码任务描述
    task = call_llm("请生成一个中等难度的 Python 编程问题，附带 3 个测试用例")
    # 2. 解析出函数签名和 test cases
    signature, tests = parse(task)
    # 3. 让 LLM 写解法
    solution = call_llm(f"请实现：{task}")
    # 4. 用解释器验证
    if all(run_test(solution, t) for t in tests):
        dataset.append({"task": task, "solution": solution, "tests": tests})
    # 不过的就丢掉（可以作为负样本留给 RL 环节）
```

这就是 MagicCoder、WizardCoder 的数据管线核心——**LLM 造题、LLM 解题、解释器打分**，三者闭环不需要人工干预。

---

## 8. 环节 6——Self-Refine / Critique：模型给自己打分

有了 (prompt, response) 数据，SFT 能跑了。但 RL 需要的是**偏好对** `(prompt, chosen, rejected)`。这一对从哪来？三条路：

### 8.1 从同一个 prompt 多次采样，用判别器排序

```python
responses = [model.generate(prompt) for _ in range(K)]
scores = [critic(prompt, r) for r in responses]  # critic 可以是:
                                                  # - 解释器（代码）
                                                  # - 强模型评分（通用）
                                                  # - 另一个训好的 reward model
chosen   = responses[argmax(scores)]
rejected = responses[argmin(scores)]
```

### 8.2 Self-Refine：让模型自己批评自己

```python
response   = model.generate(prompt)
critique   = model.generate(f"请指出下面回答的问题：{response}")
refined    = model.generate(f"根据批评改进：{critique}")
# 自动获得一对 (rejected=response, chosen=refined)
```

这个套路最早在 Self-Refine、Constitutional AI 里用，Claude 训练里也是主力——**让模型自我迭代的同时，副产品就是偏好数据**。

### 8.3 规则注入"劣化"

一种便宜得奇怪的做法：对一段优质代码**人为劣化**——删掉 docstring、把变量名换成 `a, b, c`、把 `list comprehension` 改成显式 loop——然后组成 `(劣化版=rejected, 原版=chosen)`。因为是规则注入，**标签绝对可靠**，成本几乎为零。

---

## 9. SFT 要什么样的合成数据

先回顾 SFT 的 loss（`model.py:192`）：

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

SFT 的本质是**在 prompt 段把 target 设为 -1，只对 response 段算 loss**。这个性质决定了 SFT 数据的三条核心要求：

### 9.1 Format 必须绝对一致

SFT 把整个 (prompt, response) 序列完整拟合。如果一半样本用 `### Instruction: ... ### Response: ...` 格式，另一半用 `<|user|> ... <|assistant|> ...` 格式，模型就学会**两个都凑合但哪个都不精通**——推理时用哪个格式都性能打折。合成数据时，**一条流水线只吐出一种 format**。

CodeGPT 里这就是 `SPECIAL_TOKENS`（`tokenizer.py:18`）的意义——如果未来做 SFT，要选定一套 chat template 并且训练/推理完全一致。

### 9.2 Response 必须是"想让模型学会的样子"

这一点非常反直觉：**模型会严格拟合你喂给它的东西**。如果你的 response 里有"作为一个 AI 模型我不能……"，模型就学会说这句话；如果你的 response 里 `print` 调试残留没清掉，模型就学会留 `print`。

合成数据的验收标准是：**把 response 原样 copy-paste 到生产环境，是否可以直接用？** 不能的话就继续过 refine/critique 环节。

### 9.3 多样性 >> 数量

phi-1、LIMA、Alpaca 这几篇工作共同证明了一件事：**1 万条高质量覆盖广的数据，比 100 万条同质化数据效果好得多**。合成数据的**多样性采样**比追求数量重要：

```python
# 不好的做法：每次调用强模型都用相似的 prompt
for i in range(10000):
    task = call_llm("生成一个 Python 编程问题")  # 会高度同质化

# 好的做法：显式注入多样性维度
dimensions = {
    "difficulty": ["easy", "medium", "hard", "expert"],
    "domain":     ["数据处理", "算法", "网络", "GUI", "并发", "文件 IO"],
    "style":      ["函数式", "面向对象", "过程式"],
    "language_feature": ["装饰器", "生成器", "async", "type hints", "context manager"],
}
for combo in product_of_dimensions(dimensions):
    task = call_llm(f"生成一个 {combo} 的 Python 编程问题")
```

**数据是一个空间，你要覆盖它；不是一条河，你要灌满它。**

---

## 10. RL 要什么样的合成数据（和 SFT 完全不同）

RL 阶段（PPO / DPO / GRPO）要的数据是**偏好对**或**带 reward 的轨迹**，和 SFT 的 (prompt, response) 完全不同。关键差异：

| 维度 | SFT 数据 | RL 数据 |
|---|---|---|
| 格式 | `(prompt, response)` | `(prompt, chosen, rejected)` 或 `(prompt, response, reward)` |
| 目标 | 模仿 | 排序 / 评分 |
| 需要"对的"答案吗？ | **必须** | **不必须**——甚至 chosen 本身都可以不完美，只要严格优于 rejected |
| 对 reward 精度要求 | 不涉及 | 非常高，噪声直接变成策略退化 |
| 数据量 | 10K–1M | 1K–100K（通常少一个数量级） |
| 适合的合成手段 | 蒸馏 + Self-Instruct | 执行反馈（代码）/ self-refine pair / 规则劣化 |

### 10.1 RL 对"verifiable reward" 的偏爱

RL 训练对 reward 的噪声极其敏感——reward model 差 5%，最终策略可能完全崩掉（奖励黑客）。所以 RL 阶段应当**优先使用可验证的 reward**：

- **代码**：测试通过率（环节 3）
- **数学**：答案字符串匹配
- **格式遵循**：正则匹配 / JSON 解析是否成功
- **工具调用**：工具是否返回了预期类型

DeepSeek-R1 / Qwen-Math 系列靠的就是"只在可验证任务上做 RL"——reward 是硬的，策略收敛是硬的。**不可验证的 reward（审美、风格、有用性）留给 DPO + 人类偏好**，不在 RL 里硬刚。

### 10.2 On-Policy 采样 vs Off-Policy 数据

RL 的另一条不同：数据最好是**当前策略自己采出来的**（on-policy）。这意味着 RL 的"合成数据"很大一部分不是**预先合成好的**，而是**训练过程中动态生成的**：

```python
# 每个 RL step:
response = current_policy.generate(prompt)       # on-policy 采样
reward   = verify(prompt, response)               # 环节 3 打分
# 用这对 (prompt, response, reward) 做 policy gradient
```

这意味着 RL 需要的基础设施是：

1. **Prompt 池**（可以是合成的）
2. **Verifier**（解释器、reward model、规则）
3. **高吞吐采样器**（vLLM、SGLang）

这三者比"存一堆 RL 数据到磁盘"更关键。

---

## 11. 什么时候必须上人工标注：四个"最划算点"

这是用户问题里最实用的一部分——**人工标注很贵，但不是不能用，而是要打在刀刃上**。四个最划算的位置：

### 11.1 种子（Seed）：100–1000 条

Self-Instruct 需要种子才能启动。种子的质量直接决定整个合成管线的上限——种子有偏，合成数据继承偏见并**放大**。

**人工做什么**：写 100–1000 条覆盖广、质量顶的"金标"样本。花一两个人一周时间，值。

**人工不做什么**：不要写 10 万条。超过 1000 条后，让 LLM 扩写效率更高。

### 11.2 分歧样本（Disagreement）：主动学习

当两个 reward model、或者 LLM critic 和 解释器对同一个样本**判断不一致**，这些样本才有高信息量。

```python
disputed = []
for sample in synthetic_pool:
    score_a = critic_a(sample)
    score_b = critic_b(sample)
    if abs(score_a - score_b) > threshold:
        disputed.append(sample)
# 只把 disputed 送去人工
```

这是 active learning 的标准做法：**人工预算只花在模型们拿不准的地方**，单位样本的价值比随机标注高 10–100 倍。

### 11.3 边界（Boundary）：安全、合规、价值观

这一类是合成数据**结构上做不到**的：

- 什么算"有害代码"——不是语法问题，是价值观问题。
- 什么算"正确的拒绝"——"不能教你写病毒" vs "可以讨论病毒原理"的边界。
- 什么算"合规"——某些 licence 下的代码能不能进训练集。

这些不可能让 LLM 自己决定，因为 LLM 就是来学这个边界的。Claude 的 Constitutional AI 里这一层靠**人工写 constitution + 少量人工对齐**，合成的部分只是"按 constitution 批评自己的输出"。

### 11.4 最终验收（Eval Set）：500–5000 条

**训练数据可以完全合成，但测试数据必须人工**。原因：如果你的 eval set 也是 LLM 合成的，那 eval 指标衡量的是"模型多像合成器"，不是"模型多好"。人工精心写的 eval set 是**外部锚点**，是唯一能告诉你"合成管线有没有坏掉"的东西。

**人工做什么**：每个能力维度 50–200 条，覆盖主流用法和长尾边界。保密、不进训练集、每个版本迭代都重用。

### 11.5 总账

用一张表说明"合成 vs 人工"的分工：

| 阶段 | 人工 | 合成 | 人工占比 |
|---|---|---|---|
| 种子指令 | ✅ | | 100% |
| 过滤/修复 | | ✅ 规则 + 解释器 | 0% |
| 指令扩写 | | ✅ Self-Instruct | 0% |
| Response 生成 | | ✅ 强模型蒸馏 | 0% |
| 分歧样本仲裁 | ✅ | | 100% |
| 偏好对生成 | | ✅ self-refine / 规则劣化 | <5% |
| 安全/合规边界 | ✅ | | 100% |
| Eval set | ✅ | | 100% |

**总体**人工标注占训练数据总量的比例，应当控制在 **1%–5%** 之间。低于 1% 通常意味着种子和边界都没覆盖好；高于 5% 意味着没有充分利用合成流水线。

---

## 12. 回到 CodeGPT：一条可落地的合成数据管线

CodeGPT 当前状态：

- **已有**：原始 Python 语料抓取（`data/python_code/prepare.py`）、FIM 数据合成（`tokenizer.py:148`）、预训练 loss（`model.py:192`）。
- **没有**：过滤/修复管线、执行反馈管线、SFT 数据、RL 数据。

如果要从现状走到"能跑 SFT + 有初步 RL"的路径，最小可行步骤：

### 步骤 A：加过滤层（1 天工作量）

在 `prepare.py` 的文件收集阶段，额外加 AST 解析和 ruff 检查。不过就扔。预期**数据量减少 30–50%，但质量提升一个档**。

### 步骤 B：加修复层（1 天）

对留下的代码跑 `black + isort`，标准化格式。产出**干净版预训练语料**。

### 步骤 C：加执行反馈数据生成（1 周）

选一部分"函数 + 其 docstring 里的 doctest 示例"——Python 生态里这类数据天然存在。跑 doctest 做自动验证。产出带执行标签的 `(function, docstring, tests, exec_result)` 四元组。

### 步骤 D：合成 SFT 数据（2 周）

用强模型 API 为步骤 C 的每个函数生成 5 种派生任务（描述 → 代码、代码 → 测试、不完整代码 → 补全等）。过解释器验证，只保留 pass 的。用 CodeGPT 已有的 `SPECIAL_TOKENS`（`tokenizer.py:18`）包装成 chat 格式，准备好 SFT 数据。

这时已有的训练脚本几乎**不需要修改**——SFT 的 loss 和预训练一样（`F.cross_entropy(..., ignore_index=-1)`），只要在 `get_batch` 里把 prompt 段的 target 设成 -1 即可。这和 `apply_fim_transform` 里把 `<|fim_pad|>` 设成 -1 是同一个机制。

### 步骤 E：合成 RL 数据（2 周）

对步骤 D 每个 prompt 让**当前 SFT 模型**采样 4–8 条 response，过解释器排序，取最好和最差组成偏好对。这就是 DPO 需要的数据。

**人工介入点**（整条管线的 1–5%）：

- 步骤 D 之前：写 100–200 条种子 prompt，覆盖 CodeGPT 的核心场景（函数补全、bug 修复、重构、测试生成）。
- 步骤 E 之后：抽 500 条 eval set，作为 release 前的最终验收标尺。

---

## 13. 小结：一张数据-阶段-成本表

| 合成技术 | 捕捉的故障 | SFT 有用? | RL 有用? | 代价 | 人工需求 |
|---|---|---|---|---|---|
| AST / linter 过滤 | F1, F4 | ✅ | ✅ | 极低 | 0 |
| black / isort 标准化 | F3 | ✅ | | 低 | 0 |
| 执行反馈标注 | F2 | ✅ (rejection) | ✅ (verifiable reward) | 中（沙箱成本） | 0 |
| 强模型 output 蒸馏 | F3 | ✅ | | 高（API 成本） | 种子 |
| 思维链蒸馏 + rejection | F2+F3 | ✅ | ✅ | 很高 | 种子 + 抽检 |
| Self-Instruct / Evol-Instruct | 数据多样性不足 | ✅ | 次要 | 中 | 100 条种子 |
| Self-Refine pair | 偏好数据缺失 | | ✅ | 中 | 抽检 |
| 规则劣化 pair | 偏好数据缺失 | | ✅ | 极低 | 0 |
| 人工写种子 | 冷启动 | 必需 | 必需 | 高 | 100% |
| 人工仲裁分歧样本 | 合成的盲区 | 关键 | 关键 | 中 | 100% |
| 人工写 eval set | 验收锚点 | 必需 | 必需 | 中 | 100% |

**一句话总结**：

> **合成数据不是一项"技术"，是一条"流水线"。规则能做的就不要用 LLM，LLM 能做的就不要用人工，人工只花在种子、分歧、边界、验收这四个刀刃上。垃圾代码变废为宝的关键是——每一种垃圾对应一把合适的工具，而不是指望一个银弹。**

与本项目其他文档的衔接：

- [SFT / RL 推理机制](SFT_RL_INFERENCE_MECHANICS.md) 解释了训练数据最终如何变成权重里的能力——这篇是上游（数据怎么来），那篇是下游（数据进来之后怎么生效）。
- [多次 SFT 的灾难性遗忘](SFT_FORGETTING_AND_MOE.md) 解释了 SFT 数据不配比时会发生什么——合成管线输出的数据如果不做 rehearsal 配比，同样会触发遗忘。
- [RAG vs SFT](RAG_VS_SFT.md) 解释了什么数据该进参数、什么数据留在 context——合成数据默认是冲着进参数去的，但如果更新频率高，考虑留给 RAG。
