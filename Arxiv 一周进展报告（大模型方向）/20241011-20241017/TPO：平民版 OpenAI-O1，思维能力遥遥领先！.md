# Thinking LLMs: General Instruction Following with Thought Generation

*Tianhao Wu, Janice Lan 等*

*Meta FAIR, University of California, Berkeley, New York University 等*

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/10/18/1729222986175-46fd263f-d887-424a-b9e9-79f275f592e1.png)

本文提出了 **Thought Preference Optimization (TPO)** 方法。该方法通过无监督的方式生成和优化模型的思维过程，使得大语言模型能够在回答前进行思考，并显著提升其应答质量。

### 思维偏好优化（TPO）

在 TPO 方法中，思维生成部分初始是通过指令微调过的模型（**Llama-3-8B-Instruct** ）生成的，但这种生成并未经过进一步的优化，导致其并不能显著提升最终的应答质量。为了有效利用思维生成，TPO 采用了基于 AI 反馈的强化学习范式（Reinforcement Learning from AI Feedback, RLAIF），并结合 **直接偏好优化（Direct Preference Optimization, DPO）** 进行优化。DPO 方法因其简洁高效而被选用，尤其适用于处理多轮迭代训练。

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/10/18/1729222403230-3cd2eff6-34f8-4474-ac17-fd34d03790d6.png)

**1. 思维过程生成**

首先，模型用以下提示生成一个包含两个部分的输出：**思维部分**和**应答部分**。思维部分是模型的内部推理，描述了它如何处理和分析问题，而应答部分则是直接给出的答案。

![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/18/1729222414658-9a5745f2-aaaa-4ba5-b9d9-52bfb976785e.png)

例如，模型可能生成如下的**内部思维过程**：

```text
这个问题要求我比较两种算法的效率。我可以从它们的时间复杂度开始入手，考虑输入规模的变化对其性能的影响。假设输入规模较大，算法A的复杂度为 O(n^2)，而算法B为O(nlogn)，因此算法B应该表现更好。
```

与传统的链式思维（CoT）不同，**TPO 将思维过程隐藏起来，不展示给用户**。这一点和 OpenAI-o1 十分类似。

**2. 偏好优化过程**

在每轮迭代中，模型 $ M_t $ 会基于输入指令生成多个**包含思维和应答**的候选输出。具体步骤如下：

- **思维与应答生成**：模型 $ M_t $ 接收输入 $ x_i $ 以及指令 $ p $ 后，生成 $ k $ 个输出，每个输出包含思维部分 $ z_i^k $ 和应答部分 $ y_i^k $，记为 $ M_t(p + x_i) \rightarrow \{z_i^k, y_i^k\} $。

- **评分与构建偏好对**：接着，仅将生成的应答部分 $ y_i^k $ 传递给评分模型 $ J $，对每个应答 $ y_i^k $ 进行评分，得到评分结果 $ s_i^k \in \mathbb{R} $。当使用成对比较的评分模型时，会比较候选应答对中的所有组合，最后将比较结果转化为个体评分。然后，选择得分最高和最低的应答作为“**优选应答**”和“**劣选应答**”，构造出偏好对：

$$
	\text{Pair} = \{p + x_i \rightarrow z_i^c + y_i^c; p + x_i \rightarrow z_i^r + y_i^r\}
$$

其中，$ c = \arg \max s_i^k $，$ r = \arg \min s_i^k $。
    
- **迭代训练**：有了偏好对之后，利用 DPO 损失函数对当前模型 $ M_t $ 进行训练，使其更新为 $ M_{t+1} $。每次迭代仅使用当前轮次生成的偏好对进行训练，避免了低质量的历史数据干扰模型的更新。这样，模型可以学习哪些思维过程有助于生成更好的应答。

- **长度控制**：为了防止应答在训练过程中变得过于冗长，TPO 还引入了**长度控制（Length-Control）**机制。具体做法是对应答长度进行**标准化**，并在评分中对较长的应答进行惩罚，避免模型过度生成冗长的应答。

在实验部分，作者使用了AlpacaEval和Arena-Hard基准测试来验证 TPO 方法的有效性。实验主要集中在**通用任务的指令跟随能力**，以展示 TPO 在广泛任务中的适应性。

该工作采用 **Llama-3-8B-Instruct** 模型作为种子模型开始。该模型经过初步的指令调优，但不具备思维生成的能力。实验中使用了两个评分模型：Self-Taught Evaluator (STE) 和 ArmoRM。STE 模型基于 Llama-3-70B-Instruct，并通过链式思维（CoT）生成自然语言的偏好评价。ArmoRM 则直接为每个应答输出一个分数。

![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/10/18/1729222453370-71854e4c-930d-45ae-865c-a480100bf22a.png)

在两个基准数据集上分别取得了52.5%和37.3%性能，表现优于所有基准方法以及更大的模型。

综上，TPO 方法通过无监督的思维生成与偏好优化，不仅克服了链式思维（CoT）在通用任务上的局限，使模型适应更广泛的任务类型，如通识、市场营销等，还有效解决了思维过程监督数据不足的问题。

---

- 原文链接：https://arxiv.org/abs/2410.10630v1
- 更多文章请详见以下 Github 仓库：
https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 撰稿：葛宇航

