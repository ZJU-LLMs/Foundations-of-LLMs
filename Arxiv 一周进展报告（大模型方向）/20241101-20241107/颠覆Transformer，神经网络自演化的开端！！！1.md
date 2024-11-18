## TOKENFORMER: RETHINKING TRANSFORMER SCALING WITH TOKENIZED MODEL PARAMETERS

作者：Haiyang Wang, Yue Fan 等

单位：Max Planck Institute for Informatics  Google  Peking University等

当我们想要让Transformer模型“长大”时，需要改变模型结构，然后对其进行重新训练。这种方式导致Transformer不能像人脑一样，随着时间和环境自动的进行演进。本文提出的Tokenformer新架构，从底层赋予了神经网络模型自演化的能力。下面对Tokenformer展开介绍。

## 研究框图

 下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

![](https://fastly.jsdelivr.net/gh/bucketio/img14@main/2024/11/08/1731059796149-0f0e133d-d8a1-4718-a8e8-a70e23f1b53f.png)

  本文研究内容是Transformer模型参数扩展。Transformer通常将处理单个token所需的计算划分为两个不同的部分：**与其他输入token的交互（token-token交互）**和**涉及模型参数的计算（token-parameter交互）**。注意力机制促进了token-token交互，使现代通用基础模型能够将多模态数据编码为统一的token序列，并有效捕捉它们之间的复杂依赖关系。相对而言，token-parameter的计算主要依赖于线性投影，即输入token与一组固定参数相乘。这种设计限制了可扩展性，因为增加模型规模需要改变核心架构组件，通常需从头开始**重新训练整个模型**。随着模型规模的增长，这导致了过度的资源消耗，使其变得越来越不实际。

  为了解决这一问题，本文从模型架构入手，提出了一种新的架构-Tokenformer，引入token-parameter注意力（Pattention）层，增强了token-parameter交互的灵活性，**允许模型参数的增量扩展并有效重用先前训练的模型**，从而显著降低训练负担。

  本文提出的Tokenformer架构与原始Transformer架构对比如下图所示：

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/11/08/1731059901278-5ea58bfc-c6ab-4769-bf9c-52b64430519f.png)



#### 原始transformer：

首先我们来回顾一下原始transformer，给定一组包含 $ T $ 个输入 token 的集合 $ X \in \mathbb{R}^{T \times d} $，其中 $ d $ 是通道维度。自注意力块首先通过三种不同的线性投影生成输入相关的查询 $ Q $、键 $ K $ 和值 $ V $：

$$
Q = X \cdot W^Q, \quad K = X \cdot W^K, \quad V = X \cdot W^V
$$

其中，$ W^Q, W^K \in \mathbb{R}^{d \times d_k} $ 和 $ W^V \in \mathbb{R}^{d \times d_v} $ 是可学习的权重矩阵。注意力分数通过查询和键向量的相似度计算，并使用 softmax 函数获得归一化的权重。这些分数用于计算缩放点积注意力的输出：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^\top}{\sqrt{d}}\right) \cdot V
$$

其中 $ \sqrt{d} $ 是缩放因子，用于缓解 softmax 引起的小梯度问题。最终输出为：

$$
O = X_{\text{att}} \cdot W^O
$$

其中 $ X_{\text{att}} $ 是注意力输出，$ W^O \in \mathbb{R}^{d_v \times d} $ 是输出投影矩阵。

#### Tokenformer

Tokenformer的核心创新是token-parameter注意力（Pattention）层，它引入了两组可训练的token充当模型参数，并使用交叉注意力管理输入token与这些参数token之间的交互。这样，Pattention层**引入了一个附加维度**，即**参数token的数量**，与输入和输出通道维度无关。这种解耦**允许网络参数沿着参数token轴无缝扩展**，为通过重用预训练模型实现增量模型扩展提供了所需的灵活性。

###### 对于Pattention 层，其架构如下图：

具体来说，设输入 tokens 和输出 tokens 分别表示为 $ I \in \mathbb{R}^{T \times d_1} $ 和 $ O \in \mathbb{R}^{T \times d_2} $，其中 $ T $ 是序列长度，$ d_1 $ 和 $ d_2 $ 分别为输入和输出维度。为了实现 Pattention 机制，我们引入了两组 $ n $ 个可学习的参数 tokens：$ K_P \in \mathbb{R}^{n \times d_1} $ 表示键，$ V_P \in \mathbb{R}^{n \times d_2} $ 表示值。Pattention 层的输出 $ O $ 通过缩放点积计算得到：

$$
\text{Pattention}(X, K_P, V_P) = \Theta \left( X \cdot K_P^\top \right) \cdot V_P
$$

其中 $ \Theta $ 是用于稳定 Pattention 层优化的改进型 softmax 操作。Pattention 分数 $ S \in \mathbb{R}^{n \times n} $ 的计算方式为：

$$
S_{ij} = f \left( \frac{A_{ij} \times \tau}{\sqrt{\sum_{k=1}^n |A_{ik}|^2}} \right), \quad \forall i, j \in 1...n,
$$

其中 $ A $ 是从 $ X \cdot K_P^\top $ 得到的分数，$ \tau $ 是缩放因子，默认为 $ \sqrt{n} $，$ f $ 是非线性函数，在模型中为 GeLU 函数。

![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/11/08/1731059957110-2af0a697-8c66-4778-a1bc-1b13eaefb44c.png)

###### Tokenformer的整体架构如上图所示:

具体来说：给定输入 tokens $ X_{\text{in}} \in \mathbb{R}^{T \times d} $，采用 pre-norm Transformer 的设计，其 Tokenformer 层的输出计算如下：

$$
X_{\text{inter}} = X_{\text{in}} + \text{MHA}(\text{LN}(X_{\text{in}})), \quad X_{\text{out}} = X_{\text{inter}} + \text{FFN}(\text{LN}(X_{\text{inter}}))
$$

其中，$ \text{LN} $ 表示层归一化，MHA 和 FFN 分别指我们修改后的多头自注意力和前馈层。

在多头自注意力模块中，为简化起见，采用单头变体并将 $ d_k $ 和 $ d_v $ 设为 $ d $。然后我们用 Pattention 层替代所有线性投影。记 $ \text{LN}(X_{\text{in}}) $ 为 $ X $，该模块的公式如下：

$$
Q = \text{Pattention}(X, K_P^Q, V_P^Q), \quad K = \text{Pattention}(X, K_P^K, V_P^K), \quad V = \text{Pattention}(X, K_P^V, V_P^V)
$$

$$
X_{\text{att}} = \text{softmax} \left( \frac{Q \cdot K^\top}{\sqrt{d}} \right) \cdot V
$$

$$
O_{\text{att}} = \text{Pattention}(X_{\text{att}}, K_P^O, V_P^O)
$$

其中公式 (7) 和 (9) 表示 token-parameter 注意力，而公式 (8) 表示 token-token 注意力。QKV 投影的键值参数 token 分别为 $ (K_P^Q, V_P^Q) \in \mathbb{R}^{n_q \times d} $，$ (K_P^K, V_P^K) \in \mathbb{R}^{n_k \times d} $，$ (K_P^V, V_P^V) \in \mathbb{R}^{n_v \times d} $，而 $ (K_P^O, V_P^O) \in \mathbb{R}^{n_o \times d} $ 则用于输出投影层。

为了保持一致性和简洁性，**Tokenformer 中的前馈块使用单一 Pattention 层**。记 $ \text{LN}(X_{\text{inter}}) $ 为 $ X_{\text{ffn}} $，前馈网络 (FFN) 的计算如下：

$$
O_{\text{ffn}} = \text{Pattention}(X_{\text{ffn}}, K_{\text{ffn}}^P, V_{\text{ffn}}^P)
$$

其中 $ (K_{\text{ffn}}^P, V_{\text{ffn}}^P) \in \mathbb{R}^{n_{\text{ffn}} \times d} $ 是 FFN 块的可学习键值对。

通过上述的架构设计，**将所有基本组件（包括输入数据和模型参数）表示为计算框架中的 token**。以 token 为中心的视角允许利用注意力机制的优势，**统一 Transformer 中的两个主要计算**，即 token-token 和 token-parameter 交互，从而构建出一个完全基于注意力的神经网络，以实现渐进式模型扩展。

###### 模型参数扩展

为了在不影响通用性的情况下便于理解，以下使用一个单独的 Pattention 层来展示模型扩展的细节。考虑一个现有的 Tokenformer 模型，带有一组预训练的键值参数 token，记作 $ K_{\text{old}}^P, V_{\text{old}}^P \in \mathbb{R}^{n \times d} $。如上图所示，为了扩展模型，将网络参数沿着参数token轴扩展，通过追加新的键值参数 token 来增强这组参数，记作 $ K_{\text{new}}^P, V_{\text{new}}^P \in \mathbb{R}^{m \times d} $，如下所示：

$$
K_{\text{P}}^{\text{scale}} = \begin{bmatrix} K_{\text{P}}^{\text{old}} , K_{\text{P}}^{\text{new}} \end{bmatrix}, \quad V_{\text{P}}^{\text{scale}} = \begin{bmatrix} V_{\text{P}}^{\text{old}} , V_{\text{P}}^{\text{new}} \end{bmatrix}
$$

其中 $ [\cdot_{\text{old}}, \cdot_{\text{new}}] $ 表示在 token 维度上的拼接操作，扩展后的参数集 $ K_{\text{scale}}^P, V_{\text{scale}}^P \in \mathbb{R}^{(m+n) \times d} $。扩展模型的前向传播过程定义如下：

$$
O = \text{Pattention}(X, K_{\text{scale}}^P, V_{\text{scale}}^P)
$$

这种扩展方案允许任意数量的参数集成，而无需更改输入或输出的维度。

#### 实验方面

实验方面，主要分为三个部分，分别是**渐进式模型扩展测试**，**扩展后的模型表达能力的基准测试**以及与**标准Transformer模型扩展方法的对比。**

###### 1.渐进式模型扩展测试

数据集使用了OpenWebText语料库，基线为从头训练的Transformer模型，对具有124M到1.4B参数的模型扩展性能进行了测试。Tokenformer每次扩展迭代都利用了预训练的小型Tokenformer来部分初始化较大模型的权重，例如，为了训练一个具有354M参数的模型，使用124M模型作为部分初始化器。


![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/11/08/1731060023847-2000888b-9292-4c4b-9c4a-e7a709239e4e.png)


实验结果如上图中的Figure 3所示，使用Tokenformer的渐进式扩展方法在显著减少训练预算的同时，取得了与从零训练的Transformer模型相当的性能。具体而言，从一个使用300B token训练的124M参数模型开始，逐步扩展至354M、757M和1.4B参数，仅需额外的30B token——相比于从零训练的Transformer仅需十分之一的计算预算。该扩展过程在1.4B参数规模上实现了11.77的测试困惑度。相比之下，同等规模的从零训练的Transformer模型达到了类似的困惑度11.63，但耗费了3倍的训练成本。

###### 2. 扩展后的模型表达能力测试

Tokenformer在语言建模和视觉建模上，在多个广泛认可的下游任务中测试，与同等规模的主流开源 Transformer/Vision Transformer  模型进行对比，如下图所示，Tokenformer的表现与标准 Transformer 架构的模型能力相当。

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/11/08/1731060121885-7a32f13d-640d-4446-842c-57b67636015e.png)


###### 3.与标准Transformer模型扩展方法的对比

Transformer在一定程度上也可以实现模型的重用。Net2Net 作为一种经典的模型增长方法，提出了通过复制神经元来扩展神经网络宽度的技术。在这种方法中，较小模型中某层Transformer的预训练权重矩阵 $$ W_{\text{s}}^{\text{old}} \in \mathbb{R}^{d_s \times d_s} $$ 被用于生成一个较大的权重矩阵 $$ W_{\text{l}}^{\text{new}} \in \mathbb{R}^{d_l \times d_l} \ (d_l > d_s) $$ 来填充更大的模型。此扩展过程的公式如下：

$$
W_{\text{l}}^{\text{new}} = 
\begin{bmatrix} 
W_{\text{s}}^{\text{old}} & W_{\text{l(12)}}^{\text{new}} \\ 
W_{\text{l(21)}}^{\text{new}} & W_{\text{l(22)}}^{\text{new}} 
\end{bmatrix}
$$

其中 $$ W_{\text{l(12)}}^{\text{new}} \in \mathbb{R}^{(d_l - d_s) \times d_s} $$，$$ W_{\text{l(21)}}^{\text{new}} \in \mathbb{R}^{d_s \times (d_l - d_s)} $$，以及 $$ W_{\text{l(22)}}^{\text{new}} \in \mathbb{R}^{(d_l - d_s) \times (d_l - d_s)} $$ 是用于扩展的新参数。

Tokenformer对比此方法在模型参数扩展的优势如下：

- **长上下文建模的可控token-token交互成本** ，如前所述，Transformer架构的训练成本主要分为两部分：涉及模型参数的交互和输入序列之间的交互。其中token-参数交互的成本是线性扩展的，而token-token交互的成本是二次扩展的（新token需要与已有的所有token计算注意力）。传统上，扩展Transformer模型通常通过增加通道维度来实现，比如此方法（Net2Net ）。对于更长的文本，这会导致更高的计算成本，主要是因为主要的token-token交互变得更加密集，从而影响模型在长文本处理上的性能。而Tokenformer将token-token交互的计算成本与模型扩展过程解耦。增加了参数大小而不改变token通道维度，从而保持token-token交互相关的计算成本不变。

- **扩展而不丢失已学分布** 当新增参数初始化为零时，Tokenformer可以保持现有的输出分布。这一特性在模型不断扩展以纳入更多数据时非常有利，因为它可以在增加模型容量的同时不破坏模型已有的知识，从而促进快速收敛。为评估Tokenformer的扩展效率，我们将Net2Net的Transformer扩展与Tokenformer扩展的损失曲线进行对比。如下图Figure 6所示，Tokenformer不仅收敛更快，而且达到了更低的最终损失，这归功于其在恢复训练过程中保持输出分布的能力。

- **增量扩展的性能更好** 使用Net2Net方法逐步扩展标准Transformer和Tokenformer扩展。如下图Figure 7所示，Tokenformer在扩展方面的性能优于标准Transformer。

![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/11/08/1731060197595-ea3d1a2e-0b57-4207-a027-341a2b21d955.png)

综上所述，本文提出了Tokenformer，这是一种天然可扩展的架构，通过将模型参数表示为token，用Pattention层替代了Transformer中的所有线性投影层，使得模型可以无缝且高效地进行增量扩展，而无需从零开始重新训练，同时，它原生具备参数高效调优的能力，可无缝适应新任务需求。这种比传统Transformer更具灵活性的架构将进一步推动基础模型的发展。

---

  - 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/pdf/2410.23168]
  - **更多**大模型学习资料，请详见浙大Daily实验室GitHub仓库: 
    **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
    - 本文编辑：胡中豪，毛玉仁
