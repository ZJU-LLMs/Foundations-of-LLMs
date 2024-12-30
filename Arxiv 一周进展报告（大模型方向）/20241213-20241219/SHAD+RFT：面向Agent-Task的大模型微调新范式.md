# Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning

*Ziang Ye, Zhenru Zhang* 等

*University of Science and Technology of China 等*

本文提出了一种分辨器 **SHAD**（Shuffle-Aware Discriminator）来区分大语言模型中的**推理 Tokens** 和**样板 Tokens**，并提出了**推理突出微调 RFT**（Reasoning-highlighted Fine-Tuning）方法，在大模型微调过程中自适应地强调推理 Tokens，比常见的监督微调（SFT）产生显著性能提升。

## 研究内容

研究如何通过区分大语言模型中的推理 Tokens 和样板 Tokens 来提升模型在 agent-task 中的表现。

下图为 agent-task 中一个有关推理 Tokens（绿色）和样板 Tokens（黄色和蓝色）的例子。样板 Tokens 可以进一步分为格式Tokens（黄色）和模板连接Tokens（蓝色）。

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/12/21/1734782216786-8a1ec03c-8f3c-47b6-a08b-7679b4516cf1.png)

<img src="SHAD+RFT：面向Agent-Task的大模型微调新范式.assets/image-20241221194943284.png" alt="image-20241221194943284" style="zoom:50%;" />

## 研究动机

现有的大型语言模型在多步推理和工具使用等代理（agent）能力方面存在不足，需要通过特定于 agent-task 的数据集来增强这些能力。

## 技术动机

本文进行了实验探索：下图为在常规SFT训练中，模型无法回答的不同类型的Token的损失变化。

![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/12/21/1734782203805-b436eb9a-a367-4209-b3d5-9cbd805b5a15.png)

<img src="SHAD+RFT：面向Agent-Task的大模型微调新范式.assets/image-20241221195348779.png" alt="image-20241221195348779" style="zoom:50%;" />

本文结合实验探索提出观点：由于样板 Tokens和推理 Tokens在学习难度和重要性上存在显著差异，因此模型往往容易在样本 Tokens 上过拟合，导致推理能力不足。因此，需要一种自动化和自适应的方法来区分它们，以避免过度拟合样板 Tokens。

## 解决方案

1. **SHAD（Shuffle-Aware Discriminator）**

![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2024/12/21/1734774676813-c3dfc152-9cef-4f25-9d4b-13b80fb5be91.png)

**（1）数据打乱（Data Shuffle）**：选择数据的一小部分，并将输入和输出的对应关系打乱。因为样板 Tokens由于在多个样本中重复出现，其可预测性不会因为打乱而改变，而推理 Tokens则与特定输入相关，打乱后其可预测性会降低。

**（2）模型微调（Model Tuning）**：使用打乱后的数据对大型语言模型进行微调。微调过程中，模型主要学习预测那些即使在打乱后数据中仍然保持可预测性的 Tokens ，也就是样板 Tokens 。

**（3） Tokens 分类（Classifying）**：通过比较微调模型和原始模型的 Tokens 级损失来分类 Tokens ，区分推理 Tokens 和样板 Tokens 。

- 如果损失差≤0，则该 Tokens 被分类为样板 Tokens ，因为它在微调模型中的损失没有增加，表明其可预测性没有因为输入和输出的打乱而受到影响。

- 如果损失差>0，则该 Tokens 被分类为推理 Tokens ，因为它在微调模型中的损失增加，表明其与特定输入相关，打乱后变得不可预测。

2. **RFT（Reasoning-highlighted Fine-Tuning）**

​	RFT对推理 Tokens 和样板 Tokens 的损失进行加权，使得模型在训练过程中更加关注于推理 Tokens ，从而提高模型的推理能力。

​	设 $L_b$ 为样板 Tokens 的总损失，$L_r$ 为推理 Tokens 的总损失。通过 softmax 函数动态计算权重：

$$
\omega_b = \frac{\exp(L_b / \tau)}{\exp(L_b / \tau) + \exp(L_r / \tau)}
$$

$$
\omega_r = \frac{\exp(L_r / \tau)}{\exp(L_b / \tau) + \exp(L_r / \tau)}
$$

其中，$\tau$ 是温度参数，控制权重分配的敏感度。

​	计算加权损失 $L_{RFT}$：

$$
L_{RFT} = \omega_b L_b + \omega_r L_r
$$

## 实验结果

使用 ToolBench 和 APIGen 数据集来训练模型，采用相同来源的 StableToolBench 和 BFCL 数据集进行held-in评估，使用 T-eval 数据集评估模型在多步推理任务中的表现，使用Nexus评估模型在复杂、嵌套的单步工具使用任务中的能力。

<img src="https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/12/21/1734774778020-2ba3da38-7233-4a8a-9a18-bfff2b4576d8.png" style="zoom:50%;" />

上表显示了SHAD+RFT方法在多个评估数据集上的性能比较，结果表明该方法在所有held-in和held-out评估数据集上均优于基线方法。

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/12/21/1734775187967-172dae48-aa58-4ba6-b5d1-ad90299ce8d9.png)

上图展示了SFT和RFT的训练损失。RFT可以降低了推理 Tokens的损失，同时保持了与SFT相当的样板 Tokens损失。

综上，SHAD和RFT方法有效地提高了大型语言模型在复杂真实世界问题解决中的推理能力，展现了在 agent-task 中提升模型性能的潜力。

------

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2412.14780]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：宓禹，毛玉仁



