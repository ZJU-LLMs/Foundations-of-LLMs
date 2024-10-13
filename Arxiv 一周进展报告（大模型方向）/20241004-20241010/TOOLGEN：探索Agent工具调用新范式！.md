# TOOLGEN: UNIFIED TOOL RETRIEVAL AND CALLING VIA GENERATION

**作者**：*Renxi Wang, Xudong Han* 等

**单位**：*LibrAI, Mohamed bin Zayed University of Artificial Intelligence* 等

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/10/1728566483233-1931bd55-fb6d-46ce-b9ed-3eb6374f4a93.png)

## 方法详解

本文研究的主要内容是**Agent中的工具使用**。在Agent中，大语言模型在处理任务时，常常需要与外部工具进行交互以获取信息或执行操作。然而，在工具数量显著增多的场景下，将工具描述作为上下文输入的方法因受限于上下文长度而不再可行。先检索后调用的方法可以检索出候选的工具列表，然而需要额外的检索步骤，与模型生成过程相独立，效率不高。为了解决现有方法存在的问题，本文提出了一种名为**ToolGen**的框架。

ToolGen的核心思想是将每个工具表示为模型词汇表中的一个独特的**Token**，扩展大模型的词表，从而将工具的检索和调用直接集成到模型的参数中，令大模型利用其已有知识来检索和调用工具。

如下图所示，先前基于检索的方法使用检索器基于相似度匹配检索相关工具，并将检索结果放入Prompt，交给大模型来执行调用。而ToolGen可以通过直接生成工具Tokens来完成工具的检索和调用，不依赖于任何外部检索器。


![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/10/10/1728565866342-4980f659-79c2-46d4-b88d-c3ead6fb0461.png)


ToolGen框架如下图所示：


![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/10/1728565890204-2f54718d-fa8f-4bc2-aa56-1417004a48cf.png)


图中，ToolGen主要包括以下几个部分：

1. **工具虚拟化（Tool Virtualization）**

在ToolGen中，每个工具被映射为一个唯一的Token，所有工具的Tokens被添加到LLM的词表中。这种映射可以通过多种索引方式来实现，包括数值（Numeric）索引、分层（Hierarchical）索引、语义（Semantic）索引和原子（Atomic）索引。其中，原子索引确保了每个工具都是一个单一的Token，而不是多个Token的组合。

2. **三阶段训练过程**

ToolGen的训练过程包括三个阶段，每个阶段都旨在提升模型在不同方面的性能，从而实现更准确和高效的工具检索和调用。

（1）**工具记忆（Tool Memorization）**

在工具记忆阶段，模型通过将工具描述作为输入，工具Token作为输出，来学习关联每个虚拟工具Token与其文档。这个过程类似于教模型“记住”每个工具的功能和用途。通过这种方式，模型能够理解每个工具Token背后的含义，从而在后续的检索和调用中更加准确。

（2）**检索训练（Retrieval Training）**

检索训练阶段的目标是让模型学会如何根据用户的查询生成相关的工具Token。在这个阶段，模型接收用户查询作为输入，并被训练以生成相应的工具Token作为输出。这个过程使得模型能够根据用户的需要，从其“记忆”中检索出正确的工具Token，从而实现对工具的精确调用。

（3）**端到端Agent训练（End-to-End Agent-Tuning）**

在端到端Agent训练阶段，模型被训练以作为一个自主Agent，生成计划和工具，并确定完成任务的适当参数。这个阶段的训练使用了Agent任务完成轨迹，即一系列的用户查询和相应的工具调用序列。通过这种方式，模型学会了如何在实际任务中有效地使用工具，包括何时调用工具、调用哪个工具以及如何配置工具的参数。

实验使用了包含**47,000**个真实世界工具的数据集进行验证，包括**工具检索任务**和**端到端生成任务**。

在工具检索任务中，ToolGen不仅取得了与当前最佳工具检索方法相当的性能，而且成本更低，效率更高。


![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/10/10/1728566040995-d42f18ab-d70d-4c8a-90b8-8dda5da5849e.png)


在端到端生成任务中，ToolGen在大多数设置下保持领先。

![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/10/10/1728566007900-0aac5339-1216-4fa3-910e-a3d6aefad17a.png)

综上所述，ToolGen框架通过将工具检索转化为生成过程，使得LLM能够更自然、更高效地在语言生成过程中调用工具，从而提高了Agent的自主性和效率。

---

- 原文链接: https://arxiv.org/abs/2410.03439
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/XXX
