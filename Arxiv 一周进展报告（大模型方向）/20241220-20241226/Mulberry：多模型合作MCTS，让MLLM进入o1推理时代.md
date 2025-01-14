# Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search

*Huanjin Yao, Jiaxing Huang 等*

Tsinghua University, Baidu Inc., Sun Yat-sen University

本文提出集体蒙特卡洛树搜索（Collective Monte Carlo Tree Search, CoMCTS）方法，利用多模型协作搜索和反思推理路径，并使用生成的推理数据对多模态大语言模型（MLLM）进行微调，从而提升其推理能力。

## 研究内容

改进多模态大语言模型（MLLM）的推理能力，使其能够通过逐步推理和反思来解决复杂问题。

## 研究动机

现有的MLLM在复杂任务上表现不佳，主要采用“直接预测”的模式，生成简短的最终答案，缺乏明确且定义良好的中间推理步骤和反思能力。

## 技术动机

通过多模型协作搜索与反思，使用多个模型共同搜索和识别有效的推理路径，并通过反思机制进行校准，从而提高 MLLM 在搜索推理路径时的效率和质量。

## 解决方案

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/12/27/1735293674989-4dd56311-79b2-4e55-b7b7-64aaa4dc84c2.png)

1. **集体蒙特卡洛树搜索（Collective Monte Carlo Tree Search, CoMCTS）**：

   CoMCTS通过引入集体学习的概念，利用多个模型的集体知识来协同推测、搜索和识别有效的推理路径。其核心步骤包括：

   - **扩展**：利用多个MLLM的集体知识，从当前节点扩展出多样且互补的候选推理节点。
   - **模拟与错误定位**：通过集体知识模拟推理结果，定位错误节点并剪枝其子节点。
   - **反向传播**：从叶节点向根节点反向传播，更新每个推理节点的得分和访问次数。
   - **选择**：根据上置信界（UCB）值选择下一个起始节点，平衡搜索的探索与利用。

2. **反思推理路径搜索**：

   基于CoMCTS构建的统一推理树，识别并整合负向推理节点，构建包含从负向节点到正向节点过渡的反思推理路径。通过学习反思推理路径，MLLM能够在长链推理中动态校准其推理轨迹。

3. **Mulberry-260k数据集**：

   使用CoMCTS搜索有效和反思推理路径，构建了Mulberry-260k数据集，该数据集为每个问题提供了丰富、明确且定义良好的推理节点树，用于训练具有逐步推理和反思能力的MLLM。

4. **集体监督微调（Collective Supervised Fine-Tuning, CoSFT）**：

   使用 Mulberry-260k 数据集进行集体监督微调，训练Mulberry模型，使其具备逐步推理和反思能力。具体包括：

   - **标准监督微调**：训练模型学习有效的推理路径。
   - **反思监督微调**：训练模型学习反思推理路径，校准负向推理节点。

![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/12/27/1735293690473-d77d50ab-2f64-4749-b4f6-31d11b4b4bd7.png)



## 实验结果

1. **CoMCTS的搜索效率与效果**：
   CoMCTS在搜索效率和成功率上显著优于其他树搜索方法，减少了迭代次数并提高搜索效果。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/12/27/1735293730390-c9e145c1-3bce-4973-95bd-af49bec64e01.png)

2. **Mulberry模型的性能**：
   在CoMCTS数据上训练的Mulberry模型，在多个基准测试中超越大多数开源MLLM，并与闭源模型竞争，展示了卓越的逐步推理和反思能力。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img2@main/2024/12/27/1735293741138-eebb8bf8-a00c-4ebc-a26c-20a060633dd6.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/12/27/1735293747513-68b5f0d2-1819-49a2-9d49-eb24835f52d3.png)

综上，**Mulberry**通过引入集体蒙特卡洛树搜索和反思推理路径搜索，显著提升了多模态大语言模型的推理能力。其逐步推理和动态反思机制使得Mulberry在处理复杂任务时表现更加出色，超越了现有的方法。

------

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2412.18319]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：樊怡江，毛玉仁
