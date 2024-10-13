# PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead

**作者**：*Tao Tan, Yining Qian, Ang Lv, Hongzhan Lin, Songhao Wu, Yongbo Wang, Feng Wang,  Jingtong Wu, Xin Lu, Rui Yan* 等    

**单位**： *Gaoling School of Artificial Intelligence, Renmin University of China, Southeast University, Ant Group* 等

下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/12/1728740376497-406f60a0-a708-4a5e-9c4c-9bb9806acc51.png)

本文研究的是**如何在RAG任务中提升模型对上下文的感知能力**。现有增强上下文感知的方法存在效率低下、推理时产生时间或内存开销，且很多方法针对特定位置嵌入等问题。研究发现部分注意力头会抑制上下文信息流动，影响 LLMs 的上下文感知能力，因此本文提出了**PEAR**方法，通过削弱这种抑制机制，提高 RAG 任务的性能。**该方法首先定位上下文感知抑制头，然后对这些抑制头的输出乘以可学习的系数来削弱其影响。**

具体地，PEAR方法分为两个阶段，定位抑制头和重加权系数学习，以下是详细介绍。

### 定位抑制头

1. **任务输入**

对于每个输入样本，创建一个长度为 $$n$$ 的序列 $$\{{x_1,...,x_n}\}$$，其中 $$x_i$$ 是从词汇表中随机采样的标记。然后将此序列重复，得到输入样本 $$\{x_1,...,x_{2n}\}$$，其中$$x_i = x_{i+n} (i \in [1, n])$$ 。若在位置 $$n + i + 1$$ 时，输出logits最高的标记是 $$x_i$$ ，则认为模型成功执行了代理任务。

*注：这是因为在语义无意义的上下文中，模型倾向于检查序列中的最后几个标记是否先前出现过，并复制它们最后一次出现的后一个Token作为输出。这种处理倾向使得模型在面对这种重复的输入结构时，能够尝试按照这种模式进行预测。*

2. **抑制头定位**

构建输入序列，沿着序列维度平均每个注意力头的输出得到一个**平均向量**作为**干预向量**，然后替换正常运行的 $$A_{n - 1}^{(l,h)}$$ ，这个过程视为削弱该头的影响，如图1所示。

![](https://fastly.jsdelivr.net/gh/bucketio/img14@main/2024/10/12/1728742604571-e33932f5-88f6-457b-8966-12b06b4db503.png)

接下来计算指标为**logits差异**，对于第 $$l$$ 层的第 $$h$$ 个注意力头，计算：
$$
\Delta\pi^{(l,h)}=\frac{\tilde{\pi}_{2n}^{(l,h)}[x_{n - 1}]}{\pi_{2n}[x_{n - 1}]}-1
$$
，其中 $$\pi_{2n}[x_{n - 1}]$$ 是正常运行时位置 $$2n$$ 选择 $$x_{n - 1}$$ 的logits，$$\tilde{\pi}_{2n}^{(l,h)}[x_{n - 1}]$$ 是干预 $$A^{(l,h)}$$ 后的logits。该指标值越高，表明 $$A^{(l,\Lambda)}$$ 的抑制效果越强。使用不同的 $$n$$ 值重复实验取平均值以减轻上下文长度的偏差，最后将前 $$K$$ 个最负面影响的头确定为**抑制头**。

### 重加权系数学习

在标准的多注意力头机制中，所有注意力头的输出以相等的权重聚合。本文提出将抑制头集合中的每个头的输出乘以一个可学习的标量，称为重新加权系数，以削弱抑制头的影响，如图2所示。

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/12/1728742692855-ec6a18ea-d44d-488f-84f6-e911747f3b87.png)

为了优化这些重新加权系数，冻结LLM的原始参数，**仅训练加权系数以最小化代理任务上的损失**。损失仅在序列的后半部分计算，即 $$\mathcal{L}=-\sum_{i=n}^{2n - 1}log(p(x_{i + 1}|x_{1:i}))$$，目的是增强基于上下文的检索能力而非预测下一个标记。
在下游RAG任务中，重新加权系数与任务无关且保持固定。对于每个LLM，只需通过代理任务对抑制头进行一次优化。因此，PEAR在下游RAG任务的推理过程中引入零额外开销。此外，重新加权系数的学习与LLM架构无关，使该方法与各种位置编码算法兼容。

### 实验结果

在不同RAG任务上的表现如图3所示，推理时间对比如图4所示，表明本方法在引入零额外开销的情况下提升了RAG任务的性能。

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/10/12/1728742246332-41d3d58c-3fc3-4414-912a-9fee666c16b1.png)

图5是PEAR方法在不同位置编码上的表现，表明PEAR独立于位置编码，适配于各种模型结构。

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/12/1728742333606-f276b6b4-42b7-442c-b69d-01c29972bd2a.png)

---

- 原文链接: https://arxiv.org/pdf/2409.19745
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/XXX
