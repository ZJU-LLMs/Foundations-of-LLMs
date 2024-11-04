# Flow-DPO: Improving LLM Mathematical Reasoning through Online Multi-Agent Learning

**作者**：*Yihe Deng ; Paul Mineiro*

**单位**：*University of California, Los Angeles，Microsoft Research*

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/11/01/1730392356383-92fdd8bc-4958-436d-9946-5e76b8f1fe9e.png)


## 方法详解

本文研究的是如何提升 **LLM在数学推理任务中的性能**。在微调这些模型以适应特定数学问题时，一个关键挑战是生成详细且准确的数学推理路径。然而，现有的方法要么依赖于人工注释，要么通过单一模型推理来生成这些路径，这些方法往往效率低下或成本过高。为了解决这一问题，本文提出了 Flow-DPO。

Flow-DPO方法的核心思想是利用多个LLM组件通过迭代通信共同构建解决方案。这种方法不依赖于单一模型的推理，而是通过在线学习Flows来生成推理路径。

Flow-DPO包括两个部分，分别是**增量输出生产流程** （Incremental Output Production Flow）和 **在线Flow学习与Rollouts**（Online Flow Learning with Rollouts）。

1. **增量输出生产流程**（Incremental Output Production Flow）：该流程通过分步生成答案片段来构建完整的数学推理路径。这一流程主要涉及两个独立的LLM：Answer LLM和Stop LLM。它们使用相同的架构，但承担不同的任务，通过不同的LoRA适配器进行微调，以专门化它们各自的任务。

（1）**Answer LLM** 负责生成答案的一部分，即一个答案片段；

（2）**Stop LLM** 负责评估当前的部分答案是否已经构成了**完整的回答**。

   这个过程是迭代的，直到 Stop LLM 判断出最终答案已经完成。

   通过这种方式，Flow 逐步构建起完整的回答，其中较小的片段（Chunk）大小可以提供更细致的控制，而较大的片段大小则近似于单次模型生成的过程。这种设计允许模型在生成推理路径时更加灵活和精确。增量输出生产流程示意图如下：


![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/31/1730383587539-f1a04678-6525-4e63-b1b4-525a43087270.png)


2.**在线Flow学习与rollouts（Online Flow Learning with Rollouts）**：该方法旨在进一步提升Flow的性能。与优化预定义推理步骤的方法不同，该方法**在细粒度的答案片段上执行在线DPO学习**。

   对于每个输入问题，首先由 Answer LLM 生成答案片段，一直到产生**一个完整的回答**。在得到这个输出链之后，在每个输出节点进行**随机 rollout**。例如，在初始答案片段生成和 Stop LLM 判断为“No”之后，基于之前的部分答案生成一个替代的答案片段。这个过程会一直持续到得到**第二个完整的答案**。如果这两个答案在正确性上有所不同，就将它们视为Answer LLM的DPO对，其中能够生成正确答案的推理步骤被选为优选步骤。

   Answer LLM 和 Stop LLM都参与到这些 rollout 和随后的微调中。对于每个包含问题和答案的训练实例，生成一批DPO对来训练两个LLM。这种方法使得模型能够以在线的方式进行训练，即随着新数据的处理，模型会逐步更新。


![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/31/1730383653932-4cb41cfa-c3cb-4497-895f-e3f903545491.png)


   与传统的单一模型推理相比，Flow-DPO方法提供了更大的灵活性，它不局限于预定义的“推理步骤”，而是允许可调整的片段大小，从而适应不同粒度的推理需求。

### 实验

实验中使用了两个不同规模的模型，分别为Llama-3-8B-Instruct和Phi-3-medium-128k-instruct（14B）。实验在MetaMath数据集上进行了评估。

1. **泛化性评估**：该实验的目的是评估在线DPO训练和rollouts在**提升Flow模型泛化能力**方面的有效性。实验通过计算模型在训练前对即将到来的训练数据的累积准确率，即逐步验证准确率，来衡量Flow模型的泛化性能。

   实验比较了Flow模型在有无训练情况下的推理准确率，并与单一LLM一步生成推理和答案的零样本性能进行对比。结果发现，未经训练的Flow模型初始推理准确率略低，但在线DPO训练能显著提升模型性能，如Llama-3-8B-Instruct模型在2,000个训练实例后性能提升了20%，Phi-3-medium-128k-instruct模型的准确率也提高了4个百分点，达到近83%。


![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/10/31/1730383681905-7f706bbc-c7bd-4f10-b23a-3b21137d5eb8.png)


2.**数据质量评估**：将 **Flow 生成的推理轨迹**与在**单个LLM上收集的 SFT 推理轨迹**进行了比较。使用模型的 zero-shot 精度和基于数据集的 ground-truth 轨迹的 SFT 模型的性能来建立基线。


![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/10/31/1730383901156-69027eb3-04f4-4e41-963b-ebb37be35300.png)


结果表明，Flow-DPO方法能够有效提升模型在数学推理任务中的性能；并且在线DPO学习过程能够显著增强模型的泛化能力。

综上所述，Flow-DPO通过在线多智能体学习和增量输出生产流程，显著提升了LLM在数学推理任务中的性能。此外，这种方法与数据增强和DPO等进一步的增强措施兼容，有助于提升模型性能。


---

   - 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2410.22304]
   - **更多**文章请详见 Github 仓库: 
	  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
   - 本文编辑：宓禹