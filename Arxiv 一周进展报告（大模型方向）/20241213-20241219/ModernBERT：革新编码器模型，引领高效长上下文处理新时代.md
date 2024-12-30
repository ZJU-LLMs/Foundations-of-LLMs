# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

*Benjamin Warner1† Antoine Chaffin† Benjamin Clavié1† Orion Weller Oskar Hallström Said Taghadouini Alexis Gallagher Raja Biswas1 Faisal Ladhak\* Tom Aarsen Nathan Cooper Griffin Adams Jeremy Howard1 Iacopo Poli*

*Answer.AI, LightOn, Johns Hopkins University, NVIDIA, HuggingFace*

现有的仅编码器模型如 BERT，在分类和检索任务中被广泛应用，但其架构与训练技术自发布以来改进有限，难以满足长序列处理和效率优化的需求。此外，这些模型通常受限于短序列长度（512 tokens）和低效的推理性能，在大规模数据和复杂任务场景中表现受限。因此，本文提出了 **ModernBERT**，一种现代化的仅编码器模型。通过引入旋转位置编码、交替全局与局部注意力机制，以及无填充技术，结合大规模多样化数据训练，ModernBERT 显著提升了下游任务的性能和推理效率，同时大幅扩展了其在长序列任务和跨领域应用中的适用性。

## 研究内容

研究如何引入先进架构设计、效率优化技术及大规模多样化数据训练来提高编码器性能。

## 研究动机

**模型架构与技术更新滞后**：现有仅编码器模型缺乏对近年来先进技术（如长序列支持和优化架构）的整合，无法满足新兴任务需求。

**效率与资源利用不足**：现有模型在推理速度和内存效率上表现不佳，难以适应实际场景中大规模数据处理的需求。

**数据规模与多样性受限**：传统模型的预训练数据规模较小且缺乏代码数据，限制了跨领域和专业任务的表现能力。

## 技术动机

观察到现代化模型（如 LLMs）在生成任务中进展显著，但改进较少的仅编码器模型仍是分类和检索等任务的关键工具。通过结合最新的架构优化与大规模训练，可以提升仅编码器模型的性能和效率。

## 解决方案

#### 最新的 Transformer 架构

- **偏差项**：在除最终线性层之外的所有线性层中禁用偏差项。

- **GeGLU 激活函数**：ModernBERT 引入了 GeGLU 激活函数，它是基于 GLU 的改进版本，相较于原始 BERT 的 GeLU 激活函数，在性能上表现更优。

$$
GLU(X)=(XW+b)⊙σ(XV+c),\text{GLU}(X) = (XW + b) \odot \sigma(XV + c)
$$
 
$$
GeGLU(X)=(XW1+b1)⊙GeLU(XW2+b2),\text{GeGLU}(X) = (XW_1 + b_1) \odot \text{GeLU}(XW_2 + b_2)
$$
				

- **旋转位置编码 (RoPE)**：采用 RoPE 位置编码替代绝对位置编码，使得模型在处理长文本时更加高效，并具备更好的上下文扩展能力。

- **局部-全局交替注意力机制**：ModernBERT 的注意力模块通过交替使用全局注意力和局部注意力来提升效率与性能。全局注意力支持每个 token 关注全序列，而局部注意力则专注于相邻 token，从而在提升长文本任务性能的同时减少计算开销。

  在 ModernBERT 中，每三层使用全局注意力，RoPE theta 为 160,000，其余层使用 128 个标记的局部滑动窗口注意力，RoPE theta 为 10,000。

#### 效率优化

- **Unpadding**：ModernBERT 在训练和推理时采用无填充策略，通过删除填充标记、将小批次中的所有序列连接成单个序列并将其作为大小为 1 的批次进行处理来避免这种低效性。
- **Flash Attention**：通过利用 Flash Attention 技术，ModernBERT 优化了注意力计算，显著降低了内存占用并加速了训练与推理过程。
- **硬件友好设计**：利用 PyTorch 的内置编译来通过编译所有兼容模块提高训练效率。

#### 训练

- ModernBERT 在 **2 万亿** token 的多样化数据集上进行训练，涵盖网页文档、代码和科学文献。这种大规模数据训练增强了模型的泛化能力，使其能够更好地适应不同的下游任务。
- **上下文扩展训练**
  - **阶段 1**：在 1024 tokens 上训练 1.7 万亿 token，完成模型基础能力训练。学习率为 8e-4（base）和 5e-4（large）。
  - **阶段 2**：扩展上下文长度至 8192 tokens，追加 3000 亿 token 的训练，进一步优化长文本任务性能。降低学习率至 3e-4。

#### 其他训练优化

- **StableAdamW 优化器**：引入 StableAdamW 优化器，在 AdamW 的基础上结合 Adafactor 风格的更新裁剪，进一步提升训练稳定性。
- **改进学习率策略**：采用 Warmup-Stable-Decay 梯形学习率策略，在训练初期稳定模型收敛，同时避免传统学习率衰减方式可能导致的冷启动问题。
- **序列打包**：通过序列打包技术优化训练效率，避免了因无填充机制引发的小批量大小波动，有效提升训练过程的稳定性和资源利用率。

## 实验结果

![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/12/21/1734771302963-7ded7214-7a1a-4f48-a788-bee5dfe56680.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/12/21/1734777737846-aef6fe64-117c-4963-9fca-dc225da61f88.png)

#### 自然语言理解

在 GLUE 基准测试中，**ModernBERT-base** 成为首个超越 DeBERTaV3-base 的 MLM 训练模型，展现了其在自然语言理解任务上的卓越能力。同时，**ModernBERT-large** 以比 DeBERTaV3-large 少 10% 参数、快一倍的处理速度取得了第二名的成绩，进一步验证了其在性能和效率之间的良好平衡。

#### 信息检索

在 BEIR 基准测试中，**ModernBERT** 在单向量（DPR）和多向量（ColBERT）检索设置中均优于其他编码器模型，展现了其在信息检索任务上的显著优势。在长文本检索任务中，尤其是在多向量设置下，ModernBERT 比其他长文本模型高出至少 9 个 NDCG@10 点，证明了其在处理长文本检索任务时的领先地位。

#### 代码理解

在代码相关任务中，**ModernBERT** 同样表现出色。在 CodeSearchNet 和 StackOverflow 问答（StackQA）任务中，ModernBERT 超越了所有对比模型，表明它在代码搜索和理解任务中具有卓越的能力，并且通过预训练代码数据提升了对编程内容的处理性能。

#### 效率

**ModernBERT** 在效率方面也展现了显著提升。在短文本处理上，其推理速度是 DeBERTaV3 的两倍；在长文本输入中，其速度比其他模型快两倍以上。此外，ModernBERT 的内存效率也处于领先地位，在相同的内存条件下能够处理更大的批量，为实际应用场景提供了强大的技术支持。



---

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/pdf/2412.13663v2]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
  https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：徐文溢，毛玉仁