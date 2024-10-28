# WEB AGENTS WITH WORLD MODELS: LEARNING AND LEVERAGING ENVIRONMENT DYNAMICS IN WEB NAVIGATION

**作者**：*Hyungjoo Chae, Namyoung Kim, Kai Tzu-iunn Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sunghwan Kim, Dongha Lee, Jinyoung Yeo*

**单位**：*Yonsei University*

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/10/24/1729773018477-a7693480-3a5b-405c-a6a3-339995903e08.png)

## 方法详解

本文研究的是**如何提升 Agent 在网络导航任务中的决策能力**。在网络导航中，LLMs需要生成一系列动作（如点击）来完成用户在网站上的目标。然而，现有的LLMs在长期任务中表现不佳，经常犯下不可逆的错误，例如重复购买不可退的机票。为了解决这一问题，本文提出了一种新的Web Agent框架——**World-Model-Augmented (WMA)  Web Agent**。

WMA Web Agent的核心思想是利用**世界模型**来模拟Agent行动的可能结果，从而进行更好的决策。**世界模型**是指系统内部对环境的表示，能够预测Agent行动对环境的影响。本文中，世界模型通过生成自由形式的自然语言描述来突出时间步之间的重要状态差异，从而帮助Agent在不实际执行动作的情况下预见行动的结果。

### 初步实验分析

本文首先进行了初步实验分析：

1. **分析LLM预测网络上动作造成的结果的能力**：给定LLM一个动作和两个候选的结果，两个结果的词义比较相似，令LLM通过二元分类任务判断该动作会造成什么结果。


![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2024/10/24/1729773111639-e2a66a6f-8204-4f73-bc26-c30f25b9b29c.png)


2. **分析LLM已知结果之后对动作的选择能力**：将每个候选动作的结果提供给LLM，分析其是否可以选择与用户目标相一致的正确操作。

![](https://fastly.jsdelivr.net/gh/bucketio/img2@main/2024/10/24/1729773122259-34d556c9-e2f3-41ff-ae99-9dfc28f9de5c.png)

结论：**用SOTA LLMs构建的Web Agent很难预测它们的行为的结果；然而，当了解到一个行动可能产生的潜在结果时，LLMs可以做出更好的决定。**


WMA Web Agent 框架图如下图所示，包括**训练世界模型**和**执行策略优化**两个主要步骤。

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/10/24/1729773156194-06139659-4513-47dc-9953-a77f4165fc27.png)

### 训练世界模型

1. **数据收集**：首先，通过LLM作为Web Agent与环境交互，收集训练数据集 $\mathcal{D}=\sum_{t=1}^n\{I,o_t,a_t,o_{t+1}\}$，包括用户指令 $I$、当前状态 $o_t$、Agent动作 $a_t$ 和下一状态 $o_{t+1}$。


2. **信息提取**：

   简单地使用文本观察来表示环境状态并将它们作为训练目标可能会带来以下缺点：
   （1）**训练中的信息增益低**：

   网页的状态转换往往只涉及对之前观察结果的一小部分进行修改（例如，点击下拉菜单）。因此，在时间步 $t+1$ 的观察结果 $o_{t+1}$ 中，大部分信息与时间步 $t$ 的观察结果 $o_t$ 相同。从头开始预测整个文本观察结果可能导致在训练过程中信息增益较低。

（2）**序列长度过长**：

   处理基于**文本**的完整观察结果可能导致序列长度异常长，从而导致计算成本高昂。虽然可以通过使用相对简单的**可访问性树**来替代**原始HTML**来部分缓解这个问题，但作为LLMs的训练目标，它仍然引入了较长的序列长度（平均4K个标记）。

   为了解决这些问题，本文提出了一种**专注于状态转变的观察抽象法(Transition-focused Observation Abstraction)**，该方法通过如图所示步骤来改进训练过程：

   - 使用**匈牙利算法**来计算 $o_t$ 和 $o_{t+1}$ 之间的成本矩阵，以匹配两个状态之间的元素。
   - 将匹配结果转换为状态转换的列表，指出网站上**新增**、**删除**和**更新**的元素。
   - 利用LLM将提取的 $\Delta(o_t,o_{t+1})$ 转换为自由形式的**自然语言描述** $\tilde{o}_{t+1}$，突出新旧观察结果之间的差异。


![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/10/24/1729773089638-ed4f9db1-aeaf-49c7-b07a-1de511fa8527.png)


3.**训练世界模型**：使用上述生成的描述 $\tilde{o}_{t+1}$ 作为训练目标，通过以下损失函数训练一个LLM作为世界模型，使其能够预测给定当前状态和动作后的下一状态描述。

$$
\mathcal{L}_\phi=-\log\sum_{(\tilde{o},o,a,I)\in\tilde{\mathcal{D}}}p(\tilde{o}_{t+1}|o_t,a_t,I)
$$

### 执行策略优化

Web Agent 由三个主要部分组成：**策略模型 $\theta$**、**世界模型** $\phi$ 和一个**值函数 $V$**。在推理时，策略模型 $\theta$ 是固定的，不会更新其参数。在时间 $t$ ，Agent会使用当前的观察结果 $o_t$ 和世界模型 $\phi$ 来预测下一个观察结果 $\tilde{o}_{t+1}$，并据此从策略模型 $\theta$ 中找到最优的动作/策略 $a_t$，以实现在 $I$ 中定义的目标。

首先，Agent 通过 top-p 解码从策略模型 $\theta$ 中采样 $k$ 个动作候选 $\{a^1_t, a^2_t, ..., a^k_t\}$。然后，使用世界模型 $\phi$ 来“模拟”每个动作候选 $a_t$ 可能引起的下一个观察结果 $\tilde{o}_{t+1}$：

$$
\{\tilde{o}^i_{t+1}\}^k_{i=1} = \{\phi(o_t, a^i_t, I)\}^k_{i=1}
$$

最后，Agent 使用现成的 LLM 作为价值函数 $V(\cdot)$ ，用来估计每个动作候选产生的奖励，并选择奖励最高的动作 $\hat{a}_t$：

$$
\hat{a}_t = \arg\max_{a_t \in \{a^1_t,...,a^k_t\}} V(I, o_t, a_t, \tilde{o}^i_{t+1})
$$

通过这个过程，可以在推理时优化 Web Agent的策略选择，而**无需训练**策略模型。这种无需训练的世界模型增强方法能够轻松地将世界模型 $\phi$ 适应于现有的Web Agent，包括基于提示的（prompt-based）和微调过的LLMs。

### 实验结果

**1. WebArena 性能**：

表1首先将 WMA Web Agent（16.6%）与 Vanilla CoT（13.1%）进行比较，发现在 WebArena 的几乎所有领域中都有显著提升，具体如表 2 所示。并且，当使用 GPT-4o-mini 作为策略模型时，在 Gitlab 和 Map 领域，该代理分别比 CoT 实现了 181% 和 92% 的性能提升。在 Shopping 领域的提升相对较小，可能是由于该领域的大规模状态空间，比如不同用户查询得到的搜索物品列表的多样性，这使得世界模型更难正确地学习环境动态。尽管如此，整体的提升表明了在推理时利用学习到的环境动态的有效性。

接下来，将 WMA Web Agent 与 Tree search agent 进行比较。当使用 GPT-4o 作为策略模型时，WMA Web Agent 的绝对成功率（16.6%）略低于 Tree search agent（19.2%）。然而，通过世界模型进行的策略优化为普通 CoT 带来的性能提升比树搜索更大（+29.7% 对比 + 28.0%）。


![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/10/24/1729773275507-696a2eed-84a0-47e7-9f19-909f78c70e5e.png)

**2. Mind2Web性能**：

将 WMA Web Agent 与 MindAct和 AWM 进行比较，它们分别是 Mind2Web 上先前和当前的最佳方法。表 3 显示， WMA Web Agent 代理显著优于 AWM，实现了新的最佳性能。此外，结果表明，在 Mind2Web 数据上训练的  WMA Web Agent 具有很强的泛化能力。


![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/10/24/1729773289181-e5b3ae07-7af7-4094-975a-943d6dae7f42.png)


本文首次在基于 LLM 的网络代理中引入**世界模型**，解决了当前 SOTA LLM 在理解环境动态方面的局限性。通过在 WebArena 和 Mind2Web 中的广泛实验，**表明 WMA Web Agent 有效**，且在成本和时间上优于强基线，并在 Mind2Web 中达到新的最佳性能，为网络导航的未来研究奠定了基础。

---

   - 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2410.13232]
   - **更多**文章请详见 Github 仓库: 
	  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
   - 本文编辑：宓禹