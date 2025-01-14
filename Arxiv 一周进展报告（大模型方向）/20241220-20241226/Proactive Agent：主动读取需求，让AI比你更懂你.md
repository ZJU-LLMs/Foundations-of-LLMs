# PROACTIVE AGENT: SHIFTING LLM AGENTS FROM REACTIVE RESPONSES TO ACTIVE ASSISTANCE

*Yaxi Lu, Shenzhi Yang, Cheng Qian* 等

*Department of Computer Science and Technology, Tsinghua University 等*

本文提出了一种数据驱动的方法，通过构建ProactiveBench数据集和奖励模型，训练大型语言模型（LLM）Agent主动预测并提出任务，而无需明确的人类指令。

## 研究内容

本文研究的是如何开发能够预测并主动提出任务的 LLM Agent。

下图为两种人机交互的Agent系统的比较。主动Agent被动地接收用户的查询，然后生成响应。主动行动主体根据环境观察结果推断任务，并相应地提出可能的援助请求。

<img src="https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/12/27/1735288290186-e1b6cbb3-09ff-41ef-87be-58eca547e2ab.png" style="zoom:50%;" />

## 研究动机

现有大多数 LLM Agent 处于被动响应模式，需要明确的用户指令才能执行任务，限制了其在需要预见性和自主决策的场景中的有效性。为了提高LLMAgent在需要预见性和自主决策的场景中的有效性，需要从传统的反应式响应转变为主动协助。

## 技术动机

人类在观察到他人可能需要帮助时主动提供协助，而无需对方明确请求。本文收集了人类行为数据，以训练一个能够根据环境变化和用户活动预测潜在任务的 Agent，使其能够像人类一样主动提供服务。

## 解决方案

#### 1. 任务定义（Task Definition）

本文的目标是开发一个能够基于用户活动、环境事件和状态预测用户可能分配的任务的Agent，希望通过构建自动数据生成流程来增强 LLM 驱动的Agent的主动能力。

#### 2. 流程概述（Pipeline Overview）

下图展示了数据生成的全流程，核心在于通过模拟用户活动和环境变化来生成训练数据，这些数据将被用来训练和微调LLMAgent，使其能够更好地预测和提出任务。一旦预测被接受，模拟Agent在模拟环境中执行任务并生成新事件。

<img src="https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/12/27/1735289801960-5b00aa75-711a-40ac-b233-048a6228111d.png" style="zoom: 50%;" />

1. **环境健身房（Environment Gym）**：模拟特定背景设置中的事件，并更新环境状态。
2. **主动代理（Proactive Agent）**：基于事件历史预测用户可能分配的任务，目标是最大化用户接受其提出任务的比率。Agent的预测可以是预测到的任务，或者在Agent认为不需要任务时预测为空。
3. **用户代理（User Agent）**：是使用人类标注的数据训练的**奖励模型**，用于模拟用户判断，根据Agent的预测和自身活动决定是否接受任务。

##### 2.1 Environment Gym

收集现实世界的事件作为参考，并将其转换为自然文本描述。基于收集的事件生成真实的交互场景，并提供足够的背景信息。根据用户活动生成详细的事件，并根据历史事件和当前环境状态更新实体状态。生成新事件时，需要更新环境实体的状态和属性。

##### 2.2 Proactive Agent

Proactive Agent接收新事件后，更新其记忆，并结合历史事件和用户特征提出潜在任务。一旦用户接受预测任务，Proactive Agent在Environment Gym中执行任务，生成关于Agent与环境交互的多个事件。

##### 2.3 User Agent

User Agent根据预定义的用户特征生成活动和动作，并决定是否接受Proactive Agent提出的任务。User Agent是使用人类标注的数据训练的**奖励模型**，用于模拟用户判断。

#### 3. Agent框架

下图展示了 Proactive Agent 如何通过监控事件、更新记忆、检测需求、预测任务和执行任务来实现主动性。

<img src="11.assets/image-20241227170724942.png" alt="image-20241227170724942" style="zoom:33%;" />

Proactive Agent 持续监控来自 Environment Gym 的新事件，根据新事件更新其内部记忆，这个记忆包含了用户的历史活动和环境状态信息。通过分析更新后的记忆，可以检测用户可能需要的任务。基于检测到的需求，Proactive Agent 提出一个初步的任务预测，将草稿预测发送给 User Agent，获取用户的反馈。根据用户的反馈，Proactive Agent 优化其任务预测，以提高预测的准确性和用户接受的可能性。一旦用户接受了预测任务，Agent就会在 Environment Gym 中执行这个任务，从而可能会引发更多的事件和用户活动。

该框架使得 Proactive Agent 能够在没有明确指令的情况下，根据用户活动和环境状态主动提出帮助。

## 实验结果

1. **数据集**：下表显示了ProactiveBench数据集的统计信息，包括不同设置中的事件条目数。

<img src="https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/12/27/1735291961886-cfb8df70-8f67-4c3b-ab5e-bed593e78b18.png" style="zoom:33%;" />

2. **奖励模型评估**：下表展示了不同模型作为奖励模型时与人类标注结果的一致性，本文提出的模型达到了91.80%的F1-Score。

<img src="https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/12/27/1735291994221-742ca49b-3d7c-4c59-aa6b-3014c50334b9.png" style="zoom:50%;" />

3. **Proactive Agent 评估**：比较了不同模型在ProactiveBench上的性能，本文的微调模型Qwen2-7B-Proactive达到了66.47%的F1-Score，优于所有开源和闭源模型。

<img src="https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/12/27/1735292034095-34bff6ae-03ca-4e71-abcc-1a0485a624b3.png" style="zoom:50%;" />

综上，本研究通过构建ProactiveBench数据集和奖励模型，有效提升了LLM代理的主动性，为未来人机协作的进一步发展铺平了道路。

------

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2410.12361]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：宓禹，毛玉仁



