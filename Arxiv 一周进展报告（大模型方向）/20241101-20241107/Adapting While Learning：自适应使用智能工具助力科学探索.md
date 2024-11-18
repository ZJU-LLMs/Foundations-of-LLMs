# **Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation**

**作者**：*Bohan Lyu, Yadi Cao*等

**单位**：*Tsinghua University, University of California, San Diego*

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/07/1730970776990-ddf1a3a6-8941-4e5b-9fd9-2948e35d89fd.png)

## 方法详解

本文研究的是**如何使 LLMs 在解决科学问题时智能地适应工具使用**。在科学问题解决中，LLMs 需要在依赖工具获取准确答案和通过基础推理独立解决问题之间做出选择。然而，现有的LLMs要么过度依赖工具，要么完全依赖自身的推理能力，这限制了它们在不同复杂性问题上的表现。为了解决这一问题，本文提出了一种新的两组件微调方法——**世界知识蒸馏（World Knowledge Distillation, WKD）**和**工具使用适应（Tool Usage Adaptation, TUA）**。

WKD 的核心思想是让 LLMs 直接从使用工具生成的解决方案中学习，以内化领域知识。TUA 的核心思想是根据问题的复杂性智能地决定是否使用工具。以下是该方法的总体框架图。

图(a)表示 WKD 过程。WKD 通过监督式微调，使 LLMs 能够模仿工具生成的准确答案，从而学习到解决简单科学问题所需的关键知识。

图(b)表示 TUA 过程。通过评估 LLMs 在基准测试问题上的直接回答能力，将问题划分为简单（$D_{easy}$）和困难（$D_{hard}$）两个子集。对于简单问题，LLMs 继续使用 WKD 中的对齐目标；而对于困难问题，则训练 LLMs 遵循工具使用轨迹，从而实现基于问题复杂性的智能切换。

图(c)展示了模型改进的可视化过程。蓝色和红色分别代表简单和困难的问题。垂直虚线向左移动表示内部可以解决更多的问题；简单/困难问题的水平线的移动分别显示出更智能的工具使用决策。

![](https://fastly.jsdelivr.net/gh/bucketio/img18@main/2024/11/07/1730970217825-8b4bd132-fb77-43bf-a640-62af596871b4.png)

该方法的四个关键步骤如下：

1. **使用工具生成解决方案（Solution Generation with Tools）**

  将专业工具（如物理模拟器）与 LLMs 集成，以生成高精度答案。使用问题模板和相应的工具轨迹模板来生成解决方案。在每个工具轨迹步骤中，通过系统Prompt$P_f$指导LLM强制使用工具。LLM根据工具使用轨迹返回的信息$\{I_e\}_t$和问题$x$的上下文，生成解决方案$y$。整个过程表示为：
  $$
  y \sim \pi(\cdot | x, \{I_e\}_t, P_f)
  $$
  下图为解决方案的生成过程。对于选择题，LLM 在工具辅助下得到正确答案，使用答案进行微调；对于开放性问题，除了第一步之外，LLM 生成一个建议集合，使用预先定义的度量进行排序来构建偏好对，然后使用这些数据进行偏好优化。

  ![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/11/07/1730970997657-5ec3d808-9f76-4b68-8381-e30c7fb5f540.png)

2. **世界知识蒸馏（World Knowledge Distillation, WKD）**

  在生成解决方案后，直接对目标LLM进行微调。对齐损失定义为生成答案和直接答案之间的损失：
  $$
  J_{Direct}(\theta, D, P) = -\mathbb{E}_{x \sim D, y \sim \pi(\cdot | x, \{I_e\}_t, P_f)} [\log \pi_\theta(y | x, P)]
  $$
  其中$D$代表训练数据集。WKD的损失为：
  $$
  J_{WKD}(\theta, D) = J_{Direct}(\theta, D, P_n) 
  $$
  这里$P_n$是不允许使用工具的Prompt。**WKD的目标是在不依赖工具的情况下直接生成解决方案。**

3. **工具使用适应（Tool Usage Adaptation, TUA）**

  TUA首先评估WKD微调后的LLMs在基准问题上的表现。对于每个问题，采样一组直接生成的答案以计算准确率，并根据**预定义的准确率阈值**将问题划分为简单（$D_{easy}$）和困难（$D_{hard}$）两个子集。对于$D_{easy}$，保持WKD中的对齐目标；而对于$D_{hard}$，将对齐目标切换为包含工具使用轨迹的增强解决方案，并训练LLM准确跟随这些轨迹。这种情况下，正确轨迹的对齐损失为：
  $$
  J_{Trace}(\theta, D, P) = -\mathbb{E}_{x \sim D, t \sim \pi(\cdot | x, E, P_f)} \log \pi_\theta(t | x, E, P) 
  $$
  综合考虑简单和困难问题的培训损失为：
  $$
  J_{TUA}(\theta, D_{easy}, D_{hard}) = \lambda J_{Direct}(\theta, D_{easy}, P_i) + (1 - \lambda) J_{Trace}(\theta, D_{hard}, P_i) 
  $$
  其中$P_i$是允许LLMs智能选择是否使用外部工具的Prompt。$λ$调整两个子集之间的权重，以防止极端比例分布。

4. **跨Prompt策略的知识一致性（Knowledge Consistency Across Prompt Strategies）**

  在设置中，某些直接回答问题所需的知识应在WKD期间的$P_n$和TUA及部署期间的$P_i$下学习。**为了缓解一个Prompt策略下获得的知识可能不会顺利转移到另一个Prompt策略的问题**，提出了一个**混合损失**，同时考虑WKD和TUA目标，从而在不同的Prompt策略下保持一致的知识。混合损失函数定义为：
  $$
  J_{Mix}(\theta, D, D_{easy}, D_{hard}) = \alpha J_{WKD}(\theta, D) + (1 - \alpha) J_{TUA}(\theta, D_{easy}, D_{hard})
  $$
  
## 实验

与仅使用监督式微调（SFT）的方法相比，该方法不仅提高了LLMs的答案准确率，还增强了它们在何时使用工具方面的决策能力。

1. **答案准确率**：在自定义的 Mujoco，PDE，Climate和 Epidemiology 数据集上表现明显领先，这些数据集通常没有在预训练中覆盖。在公开数据集 MATH 和 SciBench 上没有超越目前的先进模型，但比基础模型有了改进。

![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/07/1730971981816-eda9e94d-1f60-4449-be2b-61b16c4e357b.png)

2. **工具使用准确性**：自定义了工具使用评估标准。模型在除 SciBench 之外的数据集上都表现出了明显领先的工具使用准确性，意味着模型能够更智能地决定在面对复杂问题时是否需要使用外部工具，以及在能够通过基础推理解决的问题上减少对工具的依赖，从而在保持准确性的同时提高了效率。

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/11/07/1730971999419-a21f270e-973f-4198-8450-f3d1254820e7.png)

综上所述，本文提出的两组件微调方法通过**智能适应工具使用**，减少了LLMs在解决科学问题时对预训练知识的遗忘，并提高了模型在不同复杂性问题上的表现。


---

   - 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2411.00412]
   - **更多**文章请详见 Github 仓库: 
	  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
   - 本文编辑：宓禹 毛玉仁