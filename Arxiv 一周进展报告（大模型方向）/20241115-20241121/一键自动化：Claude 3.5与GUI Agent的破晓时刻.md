# **The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use**

**作者**：*Siyuan Hu, Mingyu Ouyang, Difei Gao*等

**单位**：*Show Lab, National University of Singapore*

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/11/21/1732195913585-796aceae-0ee6-4170-be85-40f5324a7cd5.png)

## 方法详解

本文**提出了一种API基础的GUI自动化模型部署框架，用以探索Claude 3.5 Computer Use模型在GUI自动化任务中的性能。**在自动化桌面任务的过程中，LLMs需要理解GUI状态并生成相应的动作，但现有模型在实际复杂环境中的性能尚不明确。为了解决这一问题，本文对首个基于API的GUI自动化模型——**Claude 3.5 Computer Use** 进行了案例分析，并提出了一个易于部署的框架：**Computer Use Out-of-the-Box**。

####Claude 3.5 Computer Use 模型

Claude 3.5 Computer Use是Anthropic公司发布的第一个GUI Agent，它通过API调用提供端到端的解决方案，直接从用户指令和观察到的纯视觉GUI状态生成动作，无需额外的外部知识或调用软件接口。模型主要包括以下部分：

1. **系统提示（System Prompt）**：定义了Claude 3.5 Computer Use与计算环境交互的规则，包括可调用的函数和参数。
2. **状态观察（State Observation）**：模型仅通过实时屏幕截图的视觉信息观察环境，不依赖于元数据或HTML。
3. **推理范式（Reasoning Paradigm）**：模型采用观察-行动范式，在决定行动前先观察环境，确保行动适合当前GUI状态。
4. **工具使用（Tool Use）**：模型使用三种Anthropic定义的工具：计算机工具、文本编辑工具和Bash工具，以执行鼠标和键盘操作、文件编辑和bash命令。
5. **GUI动作空间（GUI Action Space）**：包括所有原始的鼠标和键盘动作，如鼠标移动、点击、拖拽、打字和快捷键组合等。
6. **历史视觉上下文维护（History Visual Context Maintenance）**：模型保留历史屏幕截图的广泛上下文，以协助动作生成过程。

#### 案例分析

本文进行了全面的案例分析，以研究Claude 3.5 Computer Use模型在桌面任务自动化上的使用，涵盖了网络搜索、专业软件和游戏等领域，旨在反映各种用户群体的需求。

例如，下面是Claude 3.5 Computer Use依照用户指令完成《崩坏·星穷铁道》每日任务的案例。在该案例中，Claude根据用户需求和实时状态进行规划与反思，通过移动和点击鼠标，完成游戏中的任务选择和执行。

![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2024/11/21/1732195579025-aff3d40f-f711-4f56-9643-b81a6dd4bbcb.png)

Claude同样可以熟悉Office办公软件的使用。下面是一个在Excel中替换内容的任务，Claude模型通过键盘操作和字符键入，确定并执行内容的替换。

![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/11/21/1732195988376-eb883fc8-af70-48dc-be96-b5b57f642746.png)

评估主要从以下三个维度进行：

1. **规划（Planning）**：用于评估模型从用户查询中生成可执行计划的能力。计划应具有正确的流程，确保软件操作的整体成功，每一步都清晰且可执行。
2. **行动（Action）**：评估模型是否能够准确识别可交互的GUI元素。根据推导出的计划，评估模型是否能够逐步执行动作。
3. **批评（Critic）**：衡量模型对变化环境的意识，包括其对行动结果的适应能力。评估模型在任务不成功时是否能够重试，或在任务完成时是否能够终止执行。

通过对于多样化案例的分析，本文进行了错误分析，包括规划错误（PE）、行动错误（AE）和批评错误（CE）。规划错误发生在模型误解任务指令或计算机状态时，导致生成错误的计划。行动错误是指模型在有正确计划的情况下未能准确执行动作，通常与界面理解或精确控制能力不足有关。批评错误则是模型错误评估自己的动作或计算机状态，提供错误的任务完成反馈。

本文提出，未来GUI Agent的发展需要更动态和互动的基准测试环境，以反映现实世界的复杂性，包括考虑软件版本差异和屏幕分辨率多样性。模型的自我评估机制需要改进，以减少对任务完成情况的错误判断，可能通过引入严格的内置批评模块来实现。此外，当前模型在模仿人类使用计算机的细微差别方面仍有不足，这主要是由于训练数据的限制。

本研究通过提供即插即用的框架Computer Use Out-of-the-Box，旨在提高模型在现实世界场景中的部署和测试的可访问性，为GUI Agent研究的进步提供基础，推动向更复杂和可靠的自动化计算机使用模型发展。

---

   - 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2411.10323]
   - **更多**文章请详见 Github 仓库: 
	  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
   - 本文编辑：宓禹，毛玉仁