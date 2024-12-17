# RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models

_Hieu Tran, Zonghai Yao 等_

_University of Massachusetts Amherst, University of Massachusetts Medical School, University of Massachusetts Lowell, VA Bedford Health Care 等_

本文研究的是**如何利用检索增强提升大语言模型在复杂知识密集型问答任务中的推理准确性与事实可靠性**。现有大语言模型在处理如医学问答和常识问答等复杂任务时面临诸多挑战，推理路径单一且缺乏有效的事实性评估机制，难以充分利用外部知识资源，导致在与顶尖模型竞争以及满足任务高要求方面存在不足。受蒙特卡洛树搜索（MCTS）在复杂决策优势以及检索增强技术有效性的启发，本文提出了 **RARE（Retrieval-Augmented Reasoning Enhancement）框架**，通过**设计检索增强行动和检索增强事实性评分器**，有效整合外部知识并精准评估推理结果，从而显著提升模型在相关任务中的性能表现。 

## 研究内容
提升大语言模型在复杂知识密集型中的推理准确性与事实可靠性。

## 研究动机
现有大语言模型在复杂知识密集型问答任务中存在局限，复杂问答任务（如医学和常识问答）需要多步推理、专业知识以及准确的事实依据，但当前大语言模型在处理这些任务时推理路径较为单一，对外部知识的利用不够充分，且缺乏有效的事实性评估机制。

## 技术动机
基于rStar自博弈互推理技术（自生成推理轨迹和推理轨迹选择），借助蒙特卡洛树搜索（MCTS）与检索增强提升模型性能，在 MCTS 框架内设计新的检索增强行动，以便在推理时能有效整合外部知识资源，同时利用检索增强事实性评分器对推理路径的事实性进行准确评估，提升模型整体性能。

## 解决方案

![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/12/12/1733994511185-c9880ee1-93ea-44e8-b7ce-9b71506ff0e7.png)

1. **检索增强生成器**：基于 MCTS 的 rStar 自生成器的五个动作，包括：
  - A1：提出一步思考：基于先前步骤生成下一步推理，使语言模型逐步构建解决方案。

  - A2：提出剩余思考步骤：对于较简单问题，语言模型一次性产生所有剩余推理步骤，类似于思维链。

  - A3：生成下一个子问题并回答：将主问题分解为一系列子问题，依次解决每个子问题。

  - A4：重新回答子问题：允许语言模型重新回答之前生成的子问题，通过少样本提示提高准确性。

  - A5：重新表述问题 / 子问题：重新表述问题以澄清条件，减少误解，增强模型对问题的理解。

引入两个新的检索增强动作，将其转变为检索增强生成器，包括：

- A6 ：搜索查询生成和信息检索：依据初始问题生成搜索查询并检索相关文档。

- A7：子问题检索与重新回答：针对子问题检索特定信息并重新作答以优化中间推理步骤。

最后借助 MCTS 选取最优行动路径生成候选推理轨迹。

<div style="display: flex;">
  <img src="https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/12/12/1734002307970-b0105663-e9e5-473b-b165-6e94d9b268d4.png" alt="Image 1" style="width: 53%;">
  <img src="https://fastly.jsdelivr.net/gh/bucketio/img2@main/2024/12/12/1734002345348-5b18f2af-d9b3-421c-b7cb-1a0cc0f029c0.png" alt="Image 2" style="width: 47%;">
</div>

2. **检索增强事实性评分器（RAFS）**：将推理轨迹拆分为陈述（statement），针对陈述生成检索查询并检索信息，依据检索到的信息对陈述进行事实性评估（支持或不支持），计算推理路径的事实性得分，选取最高分的路径作为最终答案。

![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/12/12/1734002425337-d5da92d5-9b30-4edc-9272-c509d6af69d4.png)

## 实验结果
在 MedQA、MedMCQA 和 MMLU-Medical 等医学推理基准测试以及 StrategyQA、CommonsenseQA 等常识推理基准测试中，RARE 在不同模型规模（如 LLaMA3.2 3B、LLaMA3.1 8B 和 LLaMA3.1 70B）上均显著超越基线方法，且随着模型规模增大性能提升更显著，LLaMA3.1 70B 在部分任务上超越 GPT-4。

<div style="display: flex;">
  <img src="https://fastly.jsdelivr.net/gh/bucketio/img14@main/2024/12/12/1733994209110-babcf460-4ef5-413f-b07b-db2a9b60efcb.png" alt="Image 1" style="width: 50%;">
  <img src="https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/12/12/1733994253450-48645b91-b4b6-42f7-ae83-9c722ab5de48.png" alt="Image 2" style="width: 50%;">
</div>

综上，该研究通过引入检索增强行动和事实性评分机制，显著提升了大语言模型在复杂推理任务中的性能与可靠性，为知识密集型推理任务提供了高效解决方案。

---

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/pdf/2412.02830]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
  https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：董雪梅，毛玉仁