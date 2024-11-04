# AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant

**作者**：*Miao Yu, Shilong Wang* 等    

**单位**：*Xi’an Jiaotong University, Shanghai AI Lab* 等

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/11/03/1730629912408-eb1c94ae-600a-4c52-9a5c-223e064fd121.png)

## 方法详解

本文研究的是**如何通过动态集成异构 Agent 和高效管理策略实现系统任务的自动化**。现有Agent方法在处理开放式任务时，特别是在真实世界环境中，表现出泛化和专业化能力的不足，缺乏有效的集成和管理机制。受App Store丰富功能的启发，本文提出了**AgentStore**平台，通过**动态集成异构Agent**、**引入MetaAgent 和 AgentToken 策略**以及**自动化训练过程**，提升Agent系统在处理开放式任务时的性能。

具体地，**AgentStore**平台的主要框架如下图所示，以下是详细介绍。

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/11/01/1730444796401-4106a8e0-dfe0-4284-8bd4-a030daeb3ebd.png)

#### 框架组成

1. **AgentPool**：存储具有不同功能的Agent，涵盖单模态到多模态、开源到闭源模型、命令行界面到图形用户界面等多种类型，以满足不同任务需求。

2. **AgentEnroll**：定义添加新Agent的集成协议，包括Agent的能力、限制、交互的应用程序和功能演示，确保所有Agent的注册信息以标准化格式存储，规范地融入平台，便于管理和查询。

3. **MetaAgent**：作为平台核心，根据任务描述和系统状态，从 AgentPool 中选择合适的Agent（单个或多个）独立或协作完成任务。

#### MetaAgent 与 AgentToken 策略

MetaAgent是AgentStore平台的核心组件，负责管理和调度，从 AgentPool 中选择合适的Agent（单个或多个）来完成任务。对于不同的任务，MetaAgent采用思维链的方式进行分析，并从**单Agent路由**与**多Agent协作**两种模式中进行选择。

在这一过程中，AgentToken策略是MetaAgent的关键技术，具体包括：

1. **Agenttoken嵌入**：

   - 每个Agent被表示为一个可学习的token嵌入，这些嵌入被添加到MetaAgent的词汇表中。

2. **单Agent路由**：
   - 在推理时，MetaAgent通过最大化条件概率来预测最可能的下一个token。
   - 如果预测的token是Agent token，则激活相应的Agent执行任务。

3. **多Agent协作**：
   - 对于需要多个Agent协作的任务，MetaAgent通过多token预测来选择多个Agent。
   - 使用TopK函数选择概率最高的K个Agent token。
   - MetaAgent切换到Manager模式，使用构建好的提示模板，将任务分解为子任务并分配给选定的Agent。

#### AgentToken训练

1. **数据生成**
   - 采用Self-Instruct方式，从少量原始演示集和Agent描述开始，让MetaAgent依据这些信息生成新的演示集。
   - 生成新演示集后，用BERTScore筛选。计算新演示与现有演示的相似度，设定阈值范围，如果不在该范围内，就认为该演示要么与现有数据过于相似（可能是冗余的），要么过于不相似（可能是错误或不相关的），从而得到精炼后的集合，不断重复此过程，直至生成足够的演示用于训练。
2. **训练过程**
   - 训练时把任务描述和初始状态作前缀，附上Agent token作为下一个token预测的正确答案。训练目标是让模型通过更新与Agent对应的Embedding矩阵参数，使预测正确Agent token的概率尽量高，无需更新模型其他参数，具体通过计算负对数似然损失来衡量预测误差。

### 实验结果

![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/11/01/1730444831381-06bf89af-2779-4d3a-9fa3-806a79356300.png)

测试基准采用 **OSWorld**，一个包含 369 个任务的可扩展真实环境，涉及真实的网络和桌面应用程序，用于评估计算机Agent处理开放域任务的能力，是实验的主要平台。

实验设定了AgentStore的不同管理模式如下：

- GT：代表了一种理想的任务分配方式，即将每个任务分配给最适合的Agent，可视为当前 AgentStore 实现的性能上限；

- ICL：是一种基于上下文学习的方法，通过在模型输入中提供任务描述和少量示例来让模型学习如何选择Agent；

- FT：对模型进行全面的微调，通过在大量任务数据上训练模型来调整模型的参数，以学习不同的任务和Agent的关系；

- AT：即采用本文创新性提出的 AgentToken 策略。

实验结果如表1所示，与之前的通用Agent方法对比，AgentStore 通过集成 20 多个专门Agent，克服了先前方法的局限性。这些专门Agent在各自擅长的领域表现出色，在几乎所有任务领域都能稳定发挥，而通用Agent在某些特定任务类别中表现较弱。不同任务管理方法下，**AgentStore 均优于单Agent系统**，其中 **AgentToken（AT）管理能力最佳**，显著超过其他方法。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/pdf/2410.18603]

- **更多**大模型学习资料，请详见浙大 Daily 实验室 Github 仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：董雪梅，毛玉仁
