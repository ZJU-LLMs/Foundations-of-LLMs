# GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-Time Alignment

**作者**：*Yuancheng Xu, Udari Madhushani Sehwag, Alec Koppel, Sicheng Zhu, Bang An, Furong Huang, Sumitra Ganesh*

**单位**：*1 University of Maryland, College Park, 2 JPMorgan AI Research*

## 研究框图

下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img2@main/2024/10/20/1729426780422-3bf4455f-fdc4-491a-a784-29c2c9295d78.png)


## 方法详解

本文研究的是如何在测试阶段高效地对齐 LLM 以符合人类偏好。现有的 Training-Time 对齐方法（如 RLHF 和 DPO），通过人类偏好数据集微调 LLM，但这些方法成本高昂，而且在面对多目标偏好时训练过程复杂。此外，现有的 Test-Time 对齐方法（如控制解码），虽然训练成本低，但通常依赖于轨迹级的奖励模型来估计 Token 级的奖励，这导致推理成本高且奖励值不够准确。为了克服这些限制，本文提出了一种新的 Test-Time 对齐方法—— GenARM。

GenARM 的核心思想是训练自回归奖励模型来预测 Next Token 的奖励，从而准确且高效地指导 LLM 的文本生成。GenARM 首先使用偏好数据集训练自回归奖励模型，用于预测 Next Token 的奖励；然后使用类似控制解码的方式，利用 Token Level 的奖励指导模型采样 Next Token。

#### 1. 自回归奖励模型

本文的奖励模型采用自回归 Transformer 架构，在给定前文 $(x,y_{<t})$ 的条件下，预测 Next Token $y_t$ 的奖励值 $\pi_r(y_t|x,y_{<t})$，并将句子中所有 Token 的 log 奖励值之和作为完整输出的奖励，即

$$
r(x,y)=\sum_t\log\pi_r(y_t|x,y_{<t}),
$$
文中证明了这种参数化方法尽管将奖励函数限制为自回归，但其表达能力足够强，能够在KL正则化的强化学习框架内，引导 Base LLM 达到传统 RM 所能实现的任何分布。

然后，在传统的偏好数据集上，使用 Trajectory-Level 奖励模型的训练损失进行训练：

$$
\min_{\pi_r}-\mathbb{E}_{x,y_w,y_l\sim\mathcal{D}}\Big[\log\sigma\Big(\beta_r\sum_t\log\pi_r(y_{w,t}|x,y_{w,<t})-\beta_r\sum_t\log\pi_r(y_{l,t}|x,y_{l,<t})\Big)\Big],
$$
实验表明，这种方式训练的自回归奖励模型在 Token-Level 上具有区分无害和有害内容的能力。


![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/10/20/1729426792448-7468a16e-6fe7-4a9f-8cac-c79226bb1c18.png)


#### 2. 引导自回归生成


![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/20/1729426801918-b19f3735-b143-485e-b94a-16c0e393caa0.png)


**控制解码**是一种 Test-Time 的偏好对齐方法，它冻结 Base LLM，使用奖励模型指导 LLM 的生成过程，其闭式解如下

$$
\log\pi_{\mathrm{decode}}(y|x)=-\log Z(x)+\log\pi_{\mathrm{base}}(y|x)+\frac{1}{\beta}r(x,y),
$$
其中，$y$ 是任意完整回答，$Z(x)$是一个配分函数。

当使用自回归的奖励模型时，该式变为

$$
\log\pi_\text{decode}(y|x)=-\log Z(x)+\sum_t\log\pi_\text{base}(y_t|x,y_{<t})+\frac1\beta\sum_t\log\pi_r(y_t|x,y_{<t}).
$$
上式类似于来自两个语言模型的解码，参考从多个语言模型解码的方法，GenARM 最终的采样公式为：

$$
\tilde{\pi}_{\text{decode}}(y_t|x,y_{<t})\propto\pi_{\text{base}}(y_t|x,y_{<t})\Big(\pi_r(y_t|x,y_{<t})\Big)^{\frac{1}{\beta}}.
$$

#### 3. 实验

- **对齐效果**：GenARM 的**对齐效果与 Training-Time 对齐方法 DPO 相当**，显著优于现有的 Test-Time 对齐方法，而且其推理过程最高效。


![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/20/1729426822341-ade39fb5-bae4-460a-a1ea-4f3ce4bd43ed.png)


- **从弱到强的指导**： GenARM 可以使一个更小的自回归奖励模型（例如，7B参数）来指导一个更大的 LLM （例如，70B参数）。


![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/10/20/1729426830368-88f0608e-843f-418a-9b9c-452562a11c8f.png)


**多目标对齐**：通过调整多个自回归奖励模型的权重，GenARM 可以实现多目标对齐。

![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/10/20/1729426849948-18cd463b-e5c8-47a8-aa1e-eed8c24f6e57.png)

综上所述，GenARM通过训练**自回归奖励模型**在**测试阶段**实现了有效的偏好对齐，其**性能与DPO相当，显著降低了成本**。此外，该方法支持**多目标对齐**，允许调整不同偏好的奖励权重，为未来对齐算法的研究提供了新思路。

------

- 原文链接: https://arxiv.org/abs/2410.08193

- 更多文章请详见以下 Github 仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 撰稿：樊怡江
