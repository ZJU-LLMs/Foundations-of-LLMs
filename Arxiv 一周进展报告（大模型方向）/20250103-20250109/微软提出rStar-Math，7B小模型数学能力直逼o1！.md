# rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking

*Xinyu Guan, Li Lyna Zhang 等*

Microsoft Research Asia

本文提出**rStar-Math**框架，通过自我进化的深度思考训练，使小型语言模型（SLMs）能够在数学推理任务中达到或超越大型语言模型（如OpenAI o1）的表现。使用 MCTS 生成高质量 CoT 数据，并采用自我进化式的迭代过程来训练策略模型（SLM）和过程偏好模型（PPM），从而提升 SLM 的推理能力。

## 研究内容

在不依赖大模型蒸馏的情况下，生成高质量CoT训练数据，提升 SLM 解决复杂数学问题的能力。

## 研究动机

高质量数学推理数据稀缺，现有数据合成方法效果有限：

1. 高质量数学推理数据稀缺，标注成本高；

2. 基于蒸馏的数据合成方法存在收益递减问题，难以超越教师模型；

3. 语言模型生成的推理数据，其步骤对错难以判断；过程奖励模型可以评估步骤质量，但数据标注成本高，且自动标注方法效果不佳。

## 解决方案

![](https://fastly.jsdelivr.net/gh/bucketio/img18@main/2025/01/12/1736681334941-40c6805d-c179-440f-a568-f6ac5897965c.png)

1. **代码增强的CoT数据合成方法**：

   通过MCTS生成**逐步验证的推理轨迹**，每一步生成的自然语言推理（CoT）和对应的Python代码。通过对代码进行执行验证，过滤低质量的推理内容，确保中间步骤的正确性。MCTS的扩展过程通过选择、扩展、Rollout和反向传播四个步骤，生成高质量的推理轨迹。

   - **选择**：使用UCT算法选择最优节点进行扩展。
   - **扩展**：生成候选节点，并通过Python代码执行验证，保留有效的节点。
   - **Rollout**：执行 Rollout 更新节点的Q值（对节点的价值估计）。
   - **反向传播**：将最终答案的正确性反馈到每个中间步骤，更新Q值。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2025/01/12/1736681362517-098008cf-cdd6-4d5e-98c8-e1f3e5bb455c.png)

2. **过程偏好模型（Process Preference Model, PPM）**：

   PPM通过构建**偏好对**来训练过程奖励模型，避免了精确奖励分数的需求。具体来说，PPM选择Q值最高的步骤作为正例，Q值最低的步骤作为负例，并通过成对排序损失函数进行训练。

   - **偏好对构建**：选择Q值最高的步骤作为正例，Q值最低的步骤作为负例。
   - **成对排序损失**：通过Bradley-Terry模型优化PPM的评分预测。

3. **自演化深度思考（Self-Evolved Deep Thinking）**：

   rStar-Math通过四轮自演化过程，逐步提升模型的能力。每一轮使用最新的策略模型SLM（策略模型）和PPM生成更高质量的推理轨迹，并通过这些轨迹训练更强的SLM（策略模型）和PPM。

   - **第一轮**：使用DeepSeek-Coder-V2-Instruct（236B）作为初始策略模型进行MCTS生成初始数据。
   - **第二轮**：使用第一轮生成的策略模型（SLM-r1）进行更广泛的MCTS扩展，生成更高质量的推理轨迹，并训练第一个可靠的PPM（PPM-r2）。
   - **第三轮**：引入PPM增强的MCTS（PPM-r2），进一步提升数据质量。
   - **第四轮**：通过增加MCTS的扩展次数，解决更具挑战性的数学问题。
   
   ![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2025/01/12/1736681381916-3d52e5c5-f14a-4575-bc66-e468fb30e681.png)

## 实验结果

实验表明，rStar-Math在多个数学推理基准测试中表现优异，显著提升了小型语言模型的推理能力。在MATH基准测试中，rStar-Math将Qwen2.5-Math-7B的准确率从58.8%提升至90.0%，超过了OpenAI o1-preview的表现。在AIME 2024竞赛中，rStar-Math解决了53.3%的问题，表现优于大多数开源模型。

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2025/01/12/1736681403324-ecf8a89f-b7a2-471e-a43a-915b8c700096.png)

综上，rStar-Math通过自演化的深度思考机制，结合蒙特卡洛树搜索和过程偏好模型，显著提升了小型语言模型在数学推理任务中的表现。该方法不仅减少了对外部大型模型的依赖，还通过自我演化不断优化推理能力，为未来的数学推理研究提供了新的思路。

------

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2501.04519]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：樊怡江，毛玉仁
