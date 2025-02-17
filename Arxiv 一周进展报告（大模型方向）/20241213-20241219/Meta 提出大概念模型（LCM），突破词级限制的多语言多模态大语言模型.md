# Large Concept Models: Language Modeling in a Sentence Representation Space

*LCM Team，Loïc Barrault，Paul-Ambroise Duquenne 等*
*FAIR at Meta*

本文提出了一种名为“大概念模型（Large Concept Model, LCM）”的新型架构，通过在概念嵌入空间（而非传统的词级别）上进行推理和生成，从多语言和多模态的抽象层次改进当前大语言模型（LLMs）。LCM利用SONAR句子嵌入空间，在句子级别进行生成和推理。研究表明，该模型在生成任务（如总结和扩展任务）上展现了卓越的零样本泛化能力，并在多语言支持上超越了同等规模的现有LLMs。

## 研究内容

研究一种在概念嵌入空间中进行推理和生成的高效架构。

## 研究动机

现有大语言模型（LLMs）主要基于词级别操作，缺乏多层次抽象推理能力，无法实现人类般的高层次规划和推理。此外，这些模型多为英语中心化设计，对多语言支持不足，处理长上下文的效率也受限。

## 技术动机

通过在语言和模态无关的概念嵌入空间中进行操作，可以摆脱单词级别的限制，直接建模高层次语义推理过程，从而实现更好的长文本一致性和多语言零样本泛化性能。

## 解决方案

1. 总体架构设计

LCM 使用 SONAR 嵌入空间（支持 200 种语言和多模态数据）对输入进行编码，每个句子对应一个概念嵌入。整个模型包括以下三个主要步骤：

- **输入编码**：将输入文本或语音分割为句子，并使用固定的 SONAR 编码器将句子转化为概念嵌入。
- **嵌入推理**：通过 LCM 在嵌入空间中生成新的概念嵌入。
- **输出解码**：将生成的嵌入通过 SONAR 解码器转化为对应的文本或语音输出。

这种流程使得 LCM 的推理过程语言和模态无关，从而提升了跨语言和跨模态任务的泛化能力。

![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/12/22/1734874449337-7539c0f6-f126-49a2-b868-7af3dc364509.png)

2. SONAR 嵌入空间

SONAR 是一个高度语义化的嵌入空间，通过以下方式构建：

- **训练目标**：结合 200 种语言的翻译任务、去噪自动编码器任务，以及嵌入瓶颈层的 MSE 损失优化。
- **多模态扩展**：采用教师-学生方法，将文本嵌入扩展到语音模态。
- **语言和模态支持**：覆盖 200 种文本语言、76 种语音输入语言，并支持部分美式手语（ASL）。

SONAR 的多语言和多模态特性使 LCM 能够在统一的嵌入空间中进行推理。

![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2024/12/22/1734874477502-0205e8ba-1749-4e6c-9f90-f9caf8c23d14.png)

3. 模型变体

LCM 提出了三种不同的推理和生成方法：

a. **基准模型（Base-LCM）**

- 使用标准 Transformer 模型，优化均方误差（MSE）损失来预测下一个句子嵌入。
- 简单高效，但在多样性和准确性上存在局限。

![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/12/22/1734875328257-9c19708e-c904-4b9c-a149-b8a644c04187.png)

b. **基于扩散的生成模型（Diffusion-based LCM）**

- 联合上下文建模和逐步去噪推理。
- 包括两种架构：
  - **单塔架构（One-Tower）**：单个 Transformer 同时处理上下文和去噪任务。
  - **双塔架构（Two-Tower）**：上下文编码器（contextualizer）与去噪器（denoiser）分离，上下文通过交叉注意力提供条件信息。
- 使用不同的噪声调度方法（如余弦和 sigmoid）以优化扩散过程。

![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/12/22/1734875481485-17e64404-7f1f-4a34-ad6d-4bc5489ee17a.png)

c. **量化模型（Quantized LCM）**

- 使用残差向量量化（RVQ）方法将 SONAR 嵌入离散化为多级码本单元。
- 通过逐步生成量化嵌入（或预测残差），实现从离散表示到目标嵌入的迭代优化。
- 包括 **连续目标（Quant-LCM-c）** 和 **离散目标（Quant-LCM-d）** 两种优化方式。

### 实验结果

**1. 零样本泛化性能评估**：LCM在多语言任务中表现出显著的零样本泛化能力，超越同规模的现有LLMs。

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/12/22/1734875153782-85641ac2-9c7a-46b5-956c-fc1e5de50f8a.png)

**2. 生成任务实验**：在摘要生成和摘要扩展任务中，LCM表现出高质量的生成结果，扩散模型变体优于其他变体。

![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/12/22/1734875005652-a3470459-7964-469c-a934-34484769a9cb.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img18@main/2024/12/22/1734875178725-8ff101e1-5176-4b77-bbbd-9c66c303b9cb.png)

综上，LCM 的设计以抽象概念为核心，融合了多语言、多模态能力，且通过多种生成方法实现高效的推理与生成。其架构和方法为大语言模型提供了新的思路，特别是在高层次抽象推理和多样性生成方面展示了显著优势。

---

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2412.08821]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
  https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：葛宇航，毛玉仁