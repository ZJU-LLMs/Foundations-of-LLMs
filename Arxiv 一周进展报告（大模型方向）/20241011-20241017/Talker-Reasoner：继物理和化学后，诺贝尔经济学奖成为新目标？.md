# Agents Thinking Fast and Slow: A Talker-Reasoner Architecture

**作者**：*Konstantina Christakopoulou, Shibl Mourad* 等

**单位**：*Google DeepMind* 

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/10/20/1729426958171-89bc67e7-4540-45e8-95bd-774b243f6330.png)

## 方法详解

本文研究的是**如何通过双系统架构优化Agent在快速对话和复杂推理/规划中的性能**。Agent在与用户进行自然对话时，需要同时处理快速对话响应和复杂的多步推理/规划任务。然而，这两个任务在认知要求上存在差异，难以由单一的大型语言模型（LLM）同时高效完成。为了解决这一问题，本文提出了一种新的双系统架构——**Talker-Reasoner架构**。

#### Talker-Reasoner架构

Talker-Reasoner架构通过模仿人类的“快速思考”（System 1）和“慢速思考”（System 2）系统，将Agent分为两个部分：（1）**Talker**：负责与用户进行快速、直观的交流，生成对话响应。（2）**Reasoner**：负责进行复杂的多步推理和规划，调用工具，执行动作，并更新Agent状态。

下图为 Talker-Reasoner 架构图。" **Belief （信念）**" 是该架构的一个关键概念。这里的 belief 指的是一个关于世界状态或用户状态的记忆，被表示为 XML 或 JSON 中的结构化语言对象。（1）世界状态：Reasoner 在进行多步骤推理时，会生成多个中间结果，belief 可以从这些中间结果中提取生成有关世界的信息；（2）用户状态： Shared Memory 中保存着过去交互历史，可以从中提取关于的用户模型的信息，并存储在 Memory 中。belief 的形成是架构中的 Reasoner 与经典推理Agent的区别，这是因为，**提取 belief 的过程是该架构有意尝试建模世界/人类的过程**。

![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/10/20/1729427022444-cc42b804-9504-43cf-975d-f72eadd8f125.png)

在 Talker-Reasoner 架构中，Talker 和Reasoner之间的主要交互方式通过 Shared Memory 来实现。

1. **Reasoner**

Reasoner 的职责包括：（1）产生新的 belief 状态；（2）推导复杂的多步推理和规划；（3）将生成的belief 状态和推理结果存储在记忆中。公式表示如下：
$$
\hat{a}\sim\text{Reasoner}(b,\hat{a}|c_{\text{Reasoner}};Z)
$$

$\hat{a}$ 为预测的action，$b$ 为belief，$Z$ 为一个参数化的上下文学习型语言模型（为了更好地学习策略），$c_{\text{Reasoner}}$为 Reasoner 的上下文，由过去的推理轨迹 $\tau$ 、动作$a$、世界观察/用户话语 $o$ 以及belief 状态 $b$ 组成：
$$
c_{\mathrm{Reasoner}}=\mathrm{Concat}(\tau_1,a_1,o_1,b_1,\ldots,\tau_n,a_n,o_n,b_n;\hat{o}_t)
$$

2. **Talker** 

Talker 可以选择等待或不等待 Reasoner 的推理。Talker 需要理解语言和对话历史，并且能够生成自然的对话回应 utterance，公式表示如下：

$$
u_{(t+1)}\sim\mathrm{Talker}(u|c_{t+1},\mathcal{I}(\cdot|b_{\mathrm{mem}});\Phi)
$$

$u$ 是对话 utterance，$\mathcal{I}$ 为指令，指令可以根据记忆中的belief $b_{mem}$ 而变化，$\Phi$ 为模型参数， $c_{t+1}$ 为上下文，由最新用户话语 $\hat{o}_{t+1}$ 、记忆中的belief $b_{mem}$ 以及交互历史 $\mathcal{H}_{\mathrm{mem}}$ 组成：
$$
c_{t+1}=\mathrm{Concat}(\hat{o}_{t+1},b_{\mathrm{mem}},\mathcal{H}_{\mathrm{mem}})
$$


这样，Talker 就可以通过 Shared Memory 与 Reasoner **交互**。每当 Talker 需要belief信息时，它会从记忆中检索最新的状态。这种分工和协作机制使得Talker能够维持流畅的对话，而Reasoner则可以在后台进行深入的思考和规划，确保代理在需要时能够提供深思熟虑的响应。

#### 实例验证

本文通过在**睡眠辅导Agent**的实例中验证了Talker-Reasoner架构的有效性。实例使用Gemini 1.5 Flash模型实现Talker，并通过临床专家提供的输入实现Reasoner。

![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/10/20/1729427081917-e636b1d2-57c9-4178-8861-423f8f6380b3.png)


![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/20/1729427112777-27303ea4-7733-4eb6-9d57-19ae844fe4f0.png)

综上所述，Talker-Reasoner架构通过**分离对话响应和复杂推理，减少了系统延迟，实现了Agent的任务解耦**，展示了在现实世界应用中的高效性和实用性。

---

- 原文链接: https://arxiv.org/abs/2410.08328
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/XXX
