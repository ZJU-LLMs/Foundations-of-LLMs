# Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning

**作者**：*Amrith Setlur, Chirag Nagpal1, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant and Aviral Kumar*

**单位**：Google Research, Google DeepMind, Carnegie Mellon University

下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/20/1729425613431-269e8f08-47e1-4b61-8313-687b2382041e.png)

### **ORMs（结果奖励模型）**

1. 训练过程

为了训练ORM，首先从数据集中选取问题，然后收集形如

$$
(x,y\sim\pi(\cdot|x),Rex(y,y_{x}^{*}))
$$

的训练数据。其中$x$是问题，$y$是基础策略$\pi$对问题$x$生成的响应，$Rex(y,y_{x}^{*})$是评估$y$与真实答案$y_{x}^{*}$是否匹配的函数（匹配为1，不匹配为0）。然后训练ORM，使其输入问题 - 响应对$(x,y)$并预测$Rex(y,y_{x}^{*})$。 

2. 特点：

- 只在推理轨迹的最后一步提供反馈，正确则给予高奖励，错误则给予低奖励或零奖励。
- 这种奖励方式提供的正确信号非常稀疏，使得学习困难且搜索效率低。

3. 局限性：

- 由于只关注最终结果，对于中间步骤的探索和改进缺乏指导，难以有效地引导大语言模型在推理过程中逐步优化。


### **PRMs（过程奖励模型）**

1. 训练过程
    对于一个多步响应$y\sim\pi$中的每一步$a_{h}$，定义
    
$$
Q^{\pi}(s_{h},a_{h})=\mathbb{E}_{a_{h + 1},...,a_{H}\sim\pi(\cdot|s_{h},a_{h})}[Rex((a_{1},...,a_{H}),y_{x}^{*})]
$$

  作为状态$s_{h}$下动作$a_{h}$的得分，其中$s_{h}=(x,a_{1},...,a_{h - 1})$。


2. 特点：

- 理论上可以提供更细粒度的监督，对推理过程中的每一个步骤给予奖励。
- 旨在通过奖励中间步骤来促进更有效的推理过程，提高模型的性能。

3. 局限性：

- 人工标注每一步骤的方式难以扩展，成本高昂且不切实际。
- 自动标注训练的 PRM 目前收益有限，难以达到理想的效果。


### **PRMs存在的问题**

仅依赖$Q^{\pi}(s_h,a_h)$来设计奖励存在以下问题： 当从波束中的不同状态采样动作时，如果纯粹基于$Q^{\pi}$的最高值选择下一个状态，就会将来自不同状态的步骤相互比较。

例如，一个动作的预期最终结果的减少，即下面式子的值变小：

$$
Q^{\pi}(s_{1},a_{1,1}) - V^{\pi}(s_{1}) 
$$

意味着$a_{1,1}$自身对从状态$s_{1}$成功的概率有负面影响，而$a_{2,1}$从状态$s_{2}$有正面影响，但基于$Q^{\pi}$的绝对值扩展波束会**保留产生负面进展的动作**，并从波束中**移除可能有积极影响的状态**（如移除状态$s_{2}$，因为波束大小为 2）。换句话说，$Q^{\pi}$未能将对一个动作（步骤）的“评估”与前一个状态所显示的“前景”分离开来。在有限的计算和采样约束下，使用$Q^{\pi}$可能会保留具有潜在不利步骤的状态，从而损害整体成功的可能性。

<p align="center">
<img src="https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/20/1729426008969-326566c5-dea0-4d2a-9355-77940eb11be2.png" style="zoom:40%;" />
</p>




### 优势函数引入与PAV构造

为了解决上述问题，论文采用强化学习中的优势函数，衡量单个步骤的进展：

$$
A^{\pi}\left(s_{h}, a_{h}\right):=Q^{\pi}\left(s_{h}, a_{h}\right)-V^{\pi}\left(s_{h}\right)=Q^{\pi}\left(s_{h}, a_{h}\right)-Q^{\pi}\left(s_{h - 1}, a_{h - 1}\right)
$$

并在ORMs损失函数的基础上：

$$
\ell_{ORM-RL}(\pi):=\mathbb{E}_{x \sim \mathcal{D},\left(a_{1},..., a_{H}\right) \sim \pi(\cdot | x)}\left[Rex\left(\left(x, a_{1},..., a_{H}\right), y_{x}^{*}\right)\right]
$$

结合过程奖励来构建标准的强化学习目标：

$$
\ell_{PAV-RL}^{\pi'}(\pi):=\ell_{ORM-RL}(\pi)+\alpha \cdot \sum_{h=1}^{H} \mathbb{E}_{s_{h} \sim d_{h}^{\pi'}} \mathbb{E}_{a_{h} \sim \pi\left(\cdot | s_{h}\right)}\left[A^{\mu}\left(s_{h}, a_{h}\right)\right]
$$

其中$\ell_{ORM - RL}(\pi)$是标准的RL目标，它表示在数据集$\mathcal{D}$中，策略$\pi$对问题$x$生成响应$(a_{1},...,a_{H})$与真实答案$y_{x}^{*}$匹配的期望。$\alpha$是一个系数，用于平衡两部分的权重。这里策略$\mu$被论文称为证明策略（prover policy）。

 $\sum_{h = 1}^{H}\mathbb{E}_{s_{h}\sim d_{h}^{\pi'}}\mathbb{E}_{a_{h}\sim\pi(\cdot|s_{h})}[A^{\mu}(s_{h},a_{h})]$这部分表示对过程奖励$A^{\mu}$的求和。这里的$d_{h}^{\pi'}$表示在步骤$h$时由旧策略$\pi'$（上一次迭代的策略）访问的状态分布，$\mathbb{E}_{s_{h}\sim d_{h}^{\pi'}}\mathbb{E}_{a_{h}\sim\pi(\cdot|s_{h})}[A^{\mu}(s_{h},a_{h})]$表示在该状态分布下，根据策略$\pi$采取动作$a_{h}$时的过程奖励$A^{\mu}$的期望，对所有步骤$h$从$1$到$H$进行求和。 这种结合方式旨在综合考虑**结果奖励**和**过程奖励**，以更好地优化策略$\pi$。

 PAV的策略梯度为：

$$
\left.\nabla_{\pi} \ell_{PAV-RL}^{\pi'}(\pi)\right|_{\pi'=\pi}=\sum_{h=1}^{H} \nabla_{\pi} \log \pi\left(a_{h} | s_{h}\right) \cdot \underbrace{\left(Q^{\pi}\left(s_{h}, a_{h}\right)+\alpha \cdot A^{\mu}\left(s_{h}, a_{h}\right)\right)}_{\text{effective reward }}
$$

这里$\nabla_{\pi} log \pi\left(a_{h}|s_{h}\right)$是策略$\pi$关于动作$a_{h}$在状态$s_{h}$下的对数概率的梯度。   - $Q^{\pi}\left(s_{h},a_{h}\right)+\alpha\cdot A^{\mu}\left(s_{h},a_{h}\right)$被称为有效奖励。$Q^{\pi}\left(s_{h},a_{h}\right)$是基础策略$\pi$下状态 - 动作对$(s_{h},a_{h})$的价值，$A^{\mu}\left(s_{h},a_{h}\right)$是证明策略$\mu$下的优势，$\alpha$是一个系数，用于平衡两者在有效奖励中的贡献。这个公式展示了如何通过策略梯度来更新策略$\pi$，以优化目标函数$\ell_{PAV - RL}^{\pi'}(\pi)$。


### 实验结果

一、测试时计算的扩展

（一）PAVs的计算效率和准确性优势

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/10/20/1729426416061-e4fbcdb3-4f1f-4cff-9205-362be456bbe0.png)

- 在不同规模的Gemma模型（2B、9B、27B）上进行测试时搜索实验。使用PAVs进行波束搜索，并与使用ORMs进行最佳-of-N搜索进行比较。
- 结果表明，对于不同的波束大小N，PAVs在准确性上比ORMs提高了8 - 10%，在计算效率上比ORMs提高了1.5 - 5倍。例如在Gemma - 2B和9B模型上计算效率提升可达10×，在Gemma - 27B模型上为5×。

（二）证明策略$\mu$的选择影响


![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/10/20/1729426487578-c3a834e5-c165-48f4-a231-2759f28c3ed2.png)

1. 弱/强证明策略的非最优性
- 当以不同强度的Best - of - K策略作为证明策略$\mu$时，太弱（如Bo2）或太强（如Bo32）的证明策略都不是最优的。例如在以Gemma - 2B SFT模型为基础策略时，随着K增加，BoK(π)变强，但在所有N值下，Bo4表现最佳。
2. 不同基础策略$\pi$对应不同最佳证明策略$\mu$
- 在使用三个基础策略（Gemma 2B/9B/27B）作为证明策略训练PAVs的实验中，对于2B和9B基础模型，分别是9B和27B证明策略$\mu$最有效；对于27B模型，较弱的9B策略反而比27B本身更有效，这与理论上证明策略应与基础策略互补的观点相符。

（三）证明策略促进探索

<p align="center">
<img src="https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/20/1729426543946-576488f3-5c6a-4371-a370-3b40c377f567.png" style="zoom:33%;" />
  </p>

- $A_{\mu}$衡量行动的进步，与$Q_{\pi}$衡量特定状态的价值不同。有效奖励$Q_{\pi}+\alpha A_{\mu}$能在**探索新前缀**和利用现有高Q值前缀之间取得更好的平衡。
- 实验表明，与仅使用$Q_{\pi}$的波束搜索和独立同分布采样相比，使用PAVs的波束搜索能提高Pass@N性能，说明证明策略的优势有助于探索。

二、密集奖励RL的扩展

（一）PAV - RL的准确性和样本效率提升


![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/20/1729426594335-6aa8299d-08fd-46f7-b2ce-84fe178103bf.png)

- 在Gemma 2B和9B SFT模型上进行在线RL实验，比较PAV - RL和标准的ORM - RL。
- 结果表明，PAV - RL在测试准确性上比ORM - RL提高了>7%，且采样效率是ORM - RL的6倍。例如对于2B模型，PAV - RL将RFT策略提高了11%，对于9B模型提高了15%。

（二）PAV - RL在重排上的优势


<p align="center">
<img src="https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/20/1729426629037-7ce5ba2c-5fb6-4d9f-89fb-b9f299bb3f4d.png" style="zoom:33%;" />
</p>

- 在Gemma 2B上，PAV - RL的Pass@N性能比ORM - RL更高（对于任何N≤128，提高>7%），且Pass@N提升的速率也更高。这表明PAV - RL能产生更多样化的候选解，避免了ORM - RL中由于下一步分布熵较低导致的非多样化候选问题。

（三）PAVs促进探索和解决新问题

<p align="center">
<img src="https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/10/20/1729426653644-077a96be-8c36-498c-9e0f-723b9ea64bbf.png" style="zoom:33%;" />
  </p>

- 在RL过程中，ORM对不正确展开中的所有步骤同等降权，而PAVs中的有效奖励会对证明策略下有进步的步骤加权，增加了对单个步骤的覆盖，提高了基础策略成功的可能性。
- 实验表明，将PAV - RL策略与测试时波束搜索相结合，能在较小的计算预算（N = 16,32）内解决大量新问题，而SFT策略在大得多的预算（N = 256）下都无法解决这些问题。



---


- 原文链接: https://arxiv.org/abs/2410.08146
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/XXX
- 撰稿：张超