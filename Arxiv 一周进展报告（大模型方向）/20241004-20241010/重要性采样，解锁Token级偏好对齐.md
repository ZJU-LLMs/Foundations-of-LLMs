# TIS-DPO: Token-Level Importance Sampling for Direct Preference Optimization with Estimated Weights






**作者**：*Aiwei Liu, Haoping Bai, Zhiyun Lu, Yanchao Sun, Xiang Kong, Simon Wang, Jiulong Shan, Albin Madappally Jose, Xiaojiang Liu, Lijie Wen, Philip S. Yu, Meng Cao*

**单位**：*Tsinghua University, Apple, University of Illinois at Chicago*

下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。

## 研究框图

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/10/13/1728790010076-b2d39a7c-5707-4081-97f0-f24c4571819b.png)


## 方法详解

本文研究的是**如何提高大型语言模型在偏好对齐任务中的优化效率和结果的最优性**。在现有的 DPO 方法中，所有 Token 同等重要，这忽略了 Token 间重要性的差异，可能导致优化效率低下，且难以达到最优结果。为了解决这一问题，本文提出了一种新的优化目标——**基于估计权重的 Token 级重要性采样 DPO（TIS-DPO）**。

TIS-DPO 的核心思想是**为每个 Token 分配基于其奖励的重要性权重**，从而在优化过程中对不同 Token 进行差异化处理。**重要性权重是通过正面与负面LLMs对相同Token的预测概率之差来计算的，这个差值被视作奖励的估计。**

### 1. DPO的局限性：忽略 Token 级的重要性差异

由于 Token 之间的重要性差异很大，甚至好回答也可能包含低奖励的 Token。**与 Token 级奖励值相比，文本的平均奖励存在噪声**。DPO 对所有 Token 一视同仁会降低优化效果。

本文将正负回答的**平均奖励值出错的概率**称作**数据噪声**，其上界为（证明见原文附录）:

$$
P(S_w\leq S_l)\leq\exp\left(-\frac{2n_wt^2}{(b_w-a_w)^2}\right)+\exp\left(-\frac{2n_lt^2}{(b_l-a_l)^2}\right)
$$
其中，$S_w$ 和 $S_l$ 分别为正负回答的平均奖励值，$n_w$ 和 $n_l$ 分别为正负回答的 Token 数量，$b_w$ 和 $a_w$ 分别是正回答的最大奖励值和最小奖励值，$b_l$ 和 $a_l$ 分别是正回答的最大奖励值和最小奖励值。

可以发现，**文本中不同 Token 的奖励波动越大，正负回答的平均奖励出错的概率越大**，越容易影响优化过程。

### 2. 最优的偏好数据分布

为了克服上述问题，理想的偏好数据的 Token 奖励应该尽可能稳定。**最优的偏好数据分布** $D^*$ 应当满足以下定义：

**定义1**：在最优数据集 $\mathcal{D}^*$ 中，对于所有的 $x$ 和 $y^{<t}$，下一个 Token $y_t$ **是从具有相同期望奖励 $R^*$ 的分布中采样的**。也就是说，$\mathcal{D}^*$ 具有以下性质：

$$
\forall(x,y^{<t}),\quad\mathbb{E}_{y^t\sim\mathcal{D}^*(\cdot|x,y^{<t})}[r(y^t\mid x,y^{<t})]=R^*
$$
其中， $\mathcal{D}^*(\cdot\mid x,y^{<t})$  表示给定上文 $(x,y^{<t})$ 时，从 $\mathcal{D}^*$ 中采样 $y_t$ 的概率。

**由于最优的偏好数据分布很难采样，因此可以应用重要性采样的方法，从实际数据分布中采样来逼近理想分布的性质。**

**重要性抽样**是一种利用来自不同分布的样本来估计目标分布特性的方法，即

$$
\mathbb{E}_{x\sim p}[f(x)]=\mathbb{E}_{x\sim q}[f(x)\frac{p(x)}{q(x)}],
$$
其中，$p$ 为目标分布，$q$ 为抽样分布，$\frac{p(x)}{q(x)}$ 为**重要性权重**。·

而**最优分布与实际分布的关系**为（证明见原文附录）：

$$
D^*(x,y^{<t},y^t)=\frac{D(x,y^{<t},y^t)}{w(y^t\mid x,y^{<t})}.
$$
所以可以通过重要性采样逼近最优分布，其重要性权重为 $w(y^t\mid x,y^{<t})=k*exp(\mu r(y^t\mid x,y^{<t}))$​，该权重与 Token 的奖励值正相关。（直觉上理解：在“好回答”中，将奖励值太大的Token采样概率要减小一些，将奖励值太小的Token 采样概率变大一些。“坏回答”类似。）

### 3. 正反模型对比估计奖励值

传统的奖励模型难以估计 Token Level 的奖励，所以本文采用 **训练正反模型** 的方式，**通过对比二者的输出概率来估计 Token 奖励值**，即将正面模型和负面模型对同一 Token 的输出概率之差作为奖励值估计，进一步计算出重要性权重。

采用三种方式训练对比模型： **Prompt-based**，**Sft-base**，**DPO-base**.


![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2024/10/13/1728789783352-dc0c19fb-cc3f-414b-86aa-420d123efbee.png)


为避免重要性采样的高方差问题，对奖励值进行了裁剪：

$$
w_t=k\cdot\exp(\mu\cdot\text{clamp}(\log\frac{\pi^+(y_t\mid x,y^{<t})}{\pi^-(y_t\mid x,y^{<t})},L,U)),
$$

### 4. TIS-DPO（Token 级重要性采样的 DPO）

**将 Bradley-Terry 奖励模型转换为 Token-level，并且应用重要性采样，得到 TIS-DPO 的目标函数**为： 


![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/10/13/1728789668624-ec3fe64a-7077-42ce-8732-8413e07e9d1e.png)


### 5. 实验

 TIS-DPO 能够有效提升 LLMs 在偏好对齐任务中的性能，在无害性和有帮助性对齐以及摘要任务上表现出色。


![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/13/1728789717282-bc8a1873-d15b-4076-88cf-221047ea5ef5.png)


综上所述，TIS-DPO通过**为每个 Token 分配基于奖励的权重，并在优化过程中进行重要性采样**，有效地**提高了LLMs在偏好对齐任务中的优化效率和结果的最优性**。这种方法不需要修改原始数据构建过程，适用于实际应用场景。同时，也提供了一种新的思路来处理 Token 级的重要性差异。