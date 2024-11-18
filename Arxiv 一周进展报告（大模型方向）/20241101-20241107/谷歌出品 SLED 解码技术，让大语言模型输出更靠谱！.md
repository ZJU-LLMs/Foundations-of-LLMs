# SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models

**作者**：*Jianyi Zhang, Da-Cheng Juan* 等    

**单位**： *Duke University, Google Research* 等

## 研究框图

下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/11/07/1730983904538-8030588e-5dd5-4fb7-808b-f33a7cb5ac48.png)

## 方法详解

本文研究的是**如何提高大语言模型输出的真实性**。现有方法在提高大语言模型输出真实性方面存在不足，缺乏高效且广泛适用的方法。受模型内部不同层 logits 关系的启发，本文提出了**SLED 方法**，通过**分析模型不同层 logits 的差异挖掘潜在知识**、**估计真实知识分布**、**实现 logits 自我进化**以及**降低计算复杂度**，提高大语言模型输出的真实性和在实际应用中的可行性。

具体地，**SLED**方法分为三个主要部分，估计真实知识分布、实现 logits 自我进化和降低计算复杂度，主要工作流程如下图所示，以下是详细介绍。

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/11/07/1730984037270-8ff069f3-b088-4945-8213-b9c79d8b4a3f.png)

#### 估计真实知识分布

大语言模型在生成文本的过程中，不同层的 logits 蕴含着丰富的信息。通过对比早期层和最终层的 logits，可以发现它们之间的差异能够反映出模型在推理过程中的潜在知识变化，从而近似真实知识分布与模型输出分布之间的关系。

SLED利用早期层和最后一层logits的差别来估算梯度并估计真实知识分布，主要基于以下原理：

1. **训练阶段的启示**：在大语言模型的训练过程中，最终层的logits（$$logits_N$$）直接通过损失函数与真实知识分布（$$P_{real}$$）相关联，训练的目标是最小化真实分布与输出分布之间的KL散度，这意味着$$logits_N$$比早期层的logits（$$logits_n$$）更能反映真实知识分布，即 $$KL(P_{real}, P_{logits_N}) < KL(P_{real}, P_{logits_n})$$。

2. **方向近似性**：基于上述训练阶段的特性，如果对比 $$logits_n$$ 和 $$logits_N$$ ，它们的差（$$logits_n - logits_N$$）在方向上可以近似于 $$KL(P_{real}, P_{logits})$$ 在 $$logits = logits_n$$ 处的梯度$$\nabla_{logits_n} KL(P_{real}, P_{logits_n})$$。

3. **估计真实知识分布（$$P_{real}$$）**：由于难以直接获取真实知识分布，SLED利用这种近似关系来估计$$P_{real}$$。具体而言，对于每个早期层$$n$$，通过计算余弦相似度，即
   $$
   CosSim(logits_n - logits_N, \nabla_{logits_n} KL(P_{e_{i}}, P_{logits_n}))
   $$
   其中，
   $$
   \nabla_{logits_n} KL(P_{e_{i}}, P_{logits_n})=(P_{logits_n}-P_{e_{i}})/\tau
   $$
   $$P_{e_{i}}$$为标准基向量，表示真实知识分布要求生成词汇中的第$$i$$个词。

   然后选择相似度最高的 $$P_{e_{i}}$$ 作为$$P_{latent}^{(n)}$$（硬估计），进一步扩展为软估计
   $$
   P_{latent}^{(n)}=(m_{1}^{(n)},..., m_{i}^{(n)},..., m_{d}^{(n)})/m^{(n)}
   $$
   ，其中 $$m^{(n)}=\sum_{i=1}^{d} m_{i}^{(n)}$$ 为归一化因子。

   最后对所有早期层的 $$P_{latent}^{(n)}$$ 进行加权平均得到 $$P_{latent}$$，作为真实知识分布的最终估计，其中权重$$s^{(n)}$$根据各层梯度近似与词汇表中Token的对齐程度确定，即如果某层 $$n$$ 的 $$logits_n - logits_N$$与词汇表中各Token的梯度 $$\nabla_{logits_n} KL(P_{e_{i}}, P_{logits_n})$$ 更为接近，那么该层在最终估计中的权重 $$s^{(n)}$$ 就更大。


#### 实现 logits 自我进化

利用得到的最终估计 $$P_{latent}$$，计算 $$KL(P_{latent}, P_{logits_N})$$ 在 $$logits = logits_N$$ 处的梯度
$$
\nabla_{logits_N} KL(P_{latent}, P_{logits_N})=(P_{logits_N}-P_{latent})/\tau
$$
，进而得到更新后的 logits
$$
\tilde{logits}_N = logits_N - \alpha \cdot \nabla_{logits_N} KL(P_{latent}, P_{logits_N})
$$
，其中$$\alpha$$为进化率，控制对$$logits_N$$调整的幅度。

#### 降低计算复杂度

为了使 SLED 方法在实际应用中更具可行性，需要降低计算复杂度。选择最终层中 logits 值最高的前$$k$$个Token进行自我进化，其他 Token 的 logits 调整为较低数值（如$$-1000$$），从而将计算复杂度从 $$O(d^{2})$$ 降低到$$O(k^{2})$$，其中 $$k \ll d$$，$$k$$ 被称为进化规模，决定了参与自我进化的高概率Token数量。

### 实验结果

#### 广泛的 LLM 基准测试结果

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/11/07/1730984071584-6b8a9b74-fcf6-4e2a-b313-2c5329cc85d9.png)

实验基线主要使用 DoLa ，它是一种与 SLED 类似的基于层对比的解码方法。根据实验结果，SLED在**多选任务**、**开放生成任务**以及**思维链推理任务**中均表现良好，在几乎所有指标上超越了基线方法。

#### 多样化 LLM 配置评估结果

![](https://fastly.jsdelivr.net/gh/bucketio/img18@main/2024/11/07/1730984111880-4f148d83-d1a5-4cff-8dfa-275a9b5fede3.png)

SLED 在不同模型家族（LLaMA 2、LLaMA 3、Gemma 等）和规模（2B - 70B）以及 Mixture of Experts（MoE）架构上均表现出强大的泛化能力，进一步证明了其在不同模型配置下的有效性。

#### 解码时间开销

![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/11/07/1730984180289-3b595a3e-3e06-4d40-8f86-7b727cb4a979.png)

SLED 增加的解码时间开销较小，与 DoLa 相比，增加范围在 0.1% - 10% 之间，确保了方法的实用性。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/pdf/2411.02433]

- **更多**大模型学习资料，请详见浙大 Daily 实验室 Github 仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：董雪梅，毛玉仁
