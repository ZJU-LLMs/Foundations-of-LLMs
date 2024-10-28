# Aligning Large Language Models via Self-Steering Optimization

**作者**：*Hao Xiang, Bowen Yu, Hongyu Lin, Keming Lu, Yaojie Lu, Xianpei Han, Le Sun, Jingren Zhou, Junyang Lin*

**单位**：*1 Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences, 2 Alibaba Group, 3 University of Chinese Academy of Sciences, Beijing, China*


![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/10/24/1729775574669-b7f2709a-2235-403c-b4e6-ea923275ff50.png)

本文研究的是**如何自动化地对齐 LLM 与人类偏好**，而**无需人工标注偏好数据**。自动化对齐的关键在于提供可学习的、准确的偏好信号，以便于在没有人工标注的情况下进行偏好学习。为此，本文提出了**自我引导优化（Self-Steering Optimization, SSO）** 算法。

SSO 的核心思想是在迭代训练期间，基于**预定义的原则**自动生成高质量的偏好数据，从不同角度计算偏好对损失来获取偏好信号。SSO 通过确保正面回答和负面回答之间的偏好差距，同时保持偏好信号接近 On-Policy，即符合当前策略模型的学习状态，从而维持信号的准确性。

#### 1. 构建对比提示并采样回答

给定一个问题 $x$，策略模型首先识别与该问题最相关的特性和原则，包含**正面原则** $p^+$ 和**负面原则** $p^-$。然后，基于这些原则构建对比 Prompt，并抽样相应的回答，分别为**正面回答** $y^+$ 和**负面回答** $y^-$。然后用这些回答组成三个偏好对进行对齐。

给定正面原则 $p^{+}$和负面原则 $p^{-}$ 以及原始问题 $x$，模型生成正面回答 $y^{+}$ 和 负面回答 $y^{-}$，定义损失函数为

![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/24/1729775608485-5bdf729c-ed6b-4014-a6f1-7c3b23757614.png)

其中  $\mathcal{G}$ 是**自我引导损失**，用于控制 $y^{+}$ 和 $y^{-}$ 之间的质量差距，$\theta$ 是一个控制 $\mathcal{G}$ 权重的参数。$L$ 是基础损失函数（本文使用的是IPO损失），用于朝着优势回答优化模型。受 WPO（Zhou等人，2024）的启发，本文通过权重函数 $W$ 来控制符合策略的行为。

#### 2. 自我引导损失


![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/10/24/1729775772182-89e595c0-c39a-4722-a9fa-a8c1a236dbe0.png)


自我引导损失 $\mathcal{G}$ 用来 $y^{+}$ 和 $y^{-}$ 之间的质量差距，一种自然的方法是使用正面原则的Prompt $x^+$ 和负面原则的Prompt $x^-$ 作为指令来构建损失，它们相应的回答作为优势回答：

$$
\mathcal{G}=L_{base}(\mathbf{x}^+,\mathbf{y}^+,\mathbf{y}^-)+L_{base}(\mathbf{x}^-,\mathbf{y}^-,\mathbf{y}^+)
$$
但这种设计存在后门问题：容易通过精心设计的提示，将 $p^{-}$ 用作后门，从而操纵 LLM 产生有害文本。

因此，$y^{-}$应该在仍然满足 $x^{-}$ 的情况下尝试近似于模型当前的原始回答 $y^{o}$。SSO 通过使用 $y^{o}$ 作为优势回答来调整$L_{base}(x^{-}, y^{-}, y^{+})$。 $\mathcal{G}$  的最终形式为：

$$
\mathcal{G}=\mathcal{L}_{base}(\mathbf{x}^+,\mathbf{y}^+,\mathbf{y}^-)+\mathcal{L}_{base}(\mathbf{x}^-,\mathbf{y}^o,\mathbf{y}^+)
$$

#### 3. 权重函数

为了在迭代过程中调整对偏好回答的学习程度，SSO设计了一个权重函数 $W$，用来决定损失函数权重。它使用 $y^{+}$ 和 $y^{-}$ 的平均对数概率简单地**评估回答的 On-Policy 程度**，即

$$
\tilde{\pi}_{\theta}(y|x)=\frac{\log\pi_{\theta}(y|x)}{|y|}
$$
$\tilde{\pi}$ 越大，表明该回答越符合当前策略模型的行为。

权重函数 $W$ 综合正面回答 $y^{+}$ 和负面回答 $y^{-}$ 的平均对数概率，来决定损失函数的权重：

$$
\mathcal{W}(\mathbf{x},\mathbf{y}^+,\mathbf{y}^-)=\mathrm{Sigmoid}\left(-\left(\alpha\cdot\tilde{\pi}_\theta(\mathbf{y}^+|\mathbf{x})+(1-\alpha)\tilde{\pi}_\theta(\mathbf{y}^-|\mathbf{x})\right)\right)
$$

#### 4. 实验结果

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/24/1729775659610-94cd1e7e-1a0b-4cf5-82af-8e751409837f.png)

+ SFT 模型对齐

与基于原则的对齐方法相比，SSO 表现更好，在客观基准尤其是数学推理任务上存在优势。

+ Instruct 模型再次对齐

将 SSO 应用于对齐后的模型，性能仍有改进。

+ 离线数据训练奖励模型

使用 SSO 产生的离线数据集训练奖励模型，能够提升 Skywork 数据集的性能，而使用另外一个偏好数据集 UltraFeedback 数没有带来提升。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2410.17131]
- **更多**文章请详见 Github 仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：樊怡江