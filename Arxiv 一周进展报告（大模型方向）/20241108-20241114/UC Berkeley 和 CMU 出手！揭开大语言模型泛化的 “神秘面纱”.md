# What Do Learning Dynamics Reveal About Generalization in LLM Reasoning?

**作者**：*Katie Kang, Amrith Setlur* 等    

**单位**： *UC Berkeley, CMU* 等

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img2@main/2024/11/14/1731587290267-3abf0fe3-66b2-4b9a-8c69-8d59991a224c.png)

## 方法详解

本文研究的是**如何有效预测大语言模型在推理任务中的泛化能力并优化数据筛选**。现有方法在理解大语言模型微调过程中的学习动态与泛化关系上存在局限，所提出的各种泛化度量指标在LLM推理任务中与测试准确率相关性不强，在数据筛选方面也缺乏有效的度量标准。受对模型训练过程中逐步学习行为观察的启发，本文通过**定义预记忆训练准确率**来评估模型学习动态、并基于此**指导数据筛选策略**，有效预测模型泛化能力并提升数据筛选效率。

如下图所示，相同预训练模型在相同推理数据集上微调，仅因学习率不同，测试性能就有很大差异，传统基于记忆的解释无法完全说明 LLM 的泛化行为，而预记忆训练准确率则能够表现出与测试准确率较强的相关性，这说明了本方法的可行性，接下来我们具体介绍方法流程。

![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/11/14/1731587332612-553e1088-251f-493a-866b-d6c682b296be.png)

#### 分析模型学习动态

在分析模型的学习动态时，论文主要关注两个关键指标：（1）模型对训练查询的回答准确率，即衡量衡量模型生成的最终答案是否正确；（2）模型预测与目标推理步骤之间的距离（困惑度），衡量模型生成的推理步骤与目标推理步骤之间的相似性。

通过跟踪这两个指标在训练过程中的变化，可以全面评估模型的学习过程。具体来说，模型在训练初期可能会生成多样化的推理步骤，这些步骤可能与目标推理步骤不同，但最终答案是正确的。随着训练的进行，模型可能会逐渐“记忆”目标推理步骤，即生成的推理步骤与目标推理步骤高度一致。

##### 1. 分析学习动态与泛化能力的关联

首先，论文先定义了衡量记忆的指标，即困惑度。如果模型生成的推理步骤与目标推理步骤的困惑度低于某个阈值 \( p \)，则认为模型已经记忆了该目标推理步骤。

$$
\text{Perp}(f_{\theta}(y|x_i), y_i) > p
$$

，其中$$ \text{Perp}(f_{\theta}(y|x_i), y_i) $$是模型预测 $$ f_{\theta}(y|x_i) $$与目标推理步骤 $$ y_i $$ 之间的困惑度。

有了记忆的定义，接下来我们分析训练过程中的不同学习动态。如下图所示，在训练过程中主要存在三类情况：

- **高准确率+高困惑度**（粉色A点）：模型生成的推理步骤与目标推理步骤不同，但最终答案是正确的，表明模型在生成多样化的推理步骤时仍然能够正确解决问题。
- **低准确率+高困惑度**（黑色B点）：模型生成的推理步骤与目标推理步骤不同，且最终答案是错误的，表明模型在生成多样化的推理步骤时未能正确解决问题。
- **高准确率+低困惑度**（黄色C点）：模型生成的推理步骤与目标推理步骤高度一致，且最终答案是正确的，表明模型已经“记忆”了目标推理步骤。

![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/14/1731587487986-03bd7c12-bd0f-42cc-8ea8-f883690ea7fc.png)

下图进一步通过颜色编码展示了三个不同模型在训练过程中对训练查询的预测行为，在不同训练阶段主要有以下变化：

- **训练准确率的提高**：随着训练的进行，模型的训练准确率逐渐提高，表明模型在训练过程中逐步学会了如何正确解决问题。
- **困惑度的降低**：随着训练的进行，模型生成的推理步骤与目标推理步骤之间的困惑度逐渐降低，表明模型在训练过程中逐步“记忆”了目标推理步骤。
- **不同学习率设置的影响**：不同学习率设置的模型在训练过程中表现出不同的行为，表明学习率的选择对模型的学习动态有显著影响。
- **记忆与泛化的区别**：模型在训练过程中既有可能通过泛化（生成多样化的推理步骤）来解决问题，也有可能通过记忆（复制目标推理步骤）来解决问题。

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/11/14/1731587509744-0fa743de-c3be-4839-9568-090c93d232cc.png)



##### 2. 定义预记忆准确率

为了量化模型在学习过程中是否真正掌握了问题的解决方法，而不是简单地“记忆”训练数据，论文引入了“预记忆准确率”（pre-memorization accuracy）这一指标。预记忆准确率定义为模型在完全记忆目标推理步骤之前，对训练查询的最高准确率。具体计算方法如下：

- **掩码准确率**：对于每个训练查询，如果模型在某个训练阶段已经记忆了目标推理步骤，则该阶段的准确率被掩码为0。

$$
\text{MaskedAcc}(f_{\theta}(y|x_i), y_i, p) = \text{Acc}(f_{\theta}(y|x_i), y_i) \cdot \mathbb{1}[\text{Perp}(f_{\theta}(y|x_i), y_i) > p]
$$

 ，其中$$ \text{Acc}(f_{\theta}(y|x_i), y_i) $$ 是模型预测的准确率，$$ \mathbb{1}[\cdot] $$ 是指示函数，当条件为真时取值为1，否则为0。

- **预记忆准确率**：对于每个训练查询，预记忆准确率为模型在记忆目标推理步骤之前达到的最高准确率。

$$
\text{PreMemAcc}(f_{\theta_{1:m}}(y|x_i), y_i, p) = \min\left\{\max_{1 \leq m' \leq m} \text{MaskedAcc}(f_{\theta_{m'}}(y|x_i), y_i, p), \text{Acc}(f_{\theta_m}(y|x_i), y_i)\right\}
$$

，其中$$ f_{\theta_{1:m}}(y|x_i) $$ 表示模型在训练过程中的不同阶段，$$ m $$ 表示当前训练阶段。通俗来讲，预记忆准确率衡量了模型在训练进程中，尚未陷入对目标推理步骤死记硬背的情况下，所能展现出的最佳能力。通过这种方式，我们可以更精准地洞察模型在正常学习过程中的真实水平，以便更好地理解模型的泛化能力来源。



#### 基于预记忆准确率筛选数据

基于所提出的泛化指标预记忆准确率，论文提出了一种数据筛选策略，以提高数据效率和模型在推理任务中的性能。具体步骤如下：

- 计算现有训练数据集内每个示例的预记忆训练准确率。设定一个合适的阈值，这个阈值可以根据模型在验证集上的性能表现或者对模型泛化能力的期望来动态调整。

- 优先选择预记忆训练准确率低于阈值的示例组成新的数据分布，以此为基础收集新的数据。在多次迭代的训练过程中，不断评估模型在测试集上的准确率，同时根据新的数据分布更新阈值，逐步优化数据筛选过程。


通过这种迭代的数据收集方法，可以使新收集的数据更具针对性，能够有效提高样本效率，减少不必要的数据收集，在推理任务中表现出优于独立同分布采样和其他标准数据筛选方法的性能。

### 实验结果

#### 预记忆训练准确率与测试准确率的关系

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/11/14/1731587361754-43a79913-d1e2-4e85-a5a8-9ea650f59e87.png)

如上图所示，在不同的模型和不同的数据集中，预记忆训练准确率和测试准确率均呈现出很强的线性关系。

![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/11/14/1731587383842-3fd92a76-7673-4026-bc53-9520c7e441eb.png)

与其他用于预测泛化差距的指标对比实验中，论文对比了梯度方差、与初始化参数的距离、平均阈值置信度三个指标，结果显示预记忆训练准确率与测试准确率的相关性更强。

#### 预记忆训练准确率对模型预测稳健性的影响

![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/11/14/1731587537490-195da511-56fe-4f89-8bb3-83d044ed8bb9.png)

上图中说明了原始训练提示（紫色）、原始提示加上 “First”（粉色）以及原始提示加上 “We know that”（蓝绿色）。一个稳健的模型即使在提示中的推理步骤发生变化时也能得出正确的最终答案，而一个不稳健的模型预测在提示偏离训练数据时会产生错误的最终答案。基于此进行了以下实验：

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/11/14/1731587439179-2f5305f9-b08c-4a5f-80ba-5c39cfb8ccde.png)

结果表明，对于预记忆训练准确率低的示例，当对输入提示施加微小扰动时，模型预测准确性会显著下降。这意味着模型对这些示例的学习较为脆弱，可能只是表面地记住了部分特征，而没有真正理解问题的本质和解决方法，而对于预记忆训练准确率高的示例，模型在面对输入扰动时仍能保持较高性能。这为针对性地改进模型训练提供了方向。

#### 基于预记忆训练准确率的数据筛选效果

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/11/14/1731587462764-e1645473-5bf5-4b1b-960c-473b31887eb7.png)

相较于独立同分布采样和其他标准数据筛选方法，样本效率大幅提高。而随着数据集规模的增大，这种性能差距更加明显，进一步证明了基于预记忆训练准确率的数据筛选方法在提升模型训练效果方面的潜力。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/pdf/2411.07681]

- **更多**大模型学习资料，请详见浙大 Daily 实验室 Github 仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：董雪梅，毛玉仁
