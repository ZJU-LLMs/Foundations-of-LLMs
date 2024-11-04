

# 2D-DPO: Scaling Direct Preference Optimization with 2-Dimensional Supervision

**作者**：Shilong Li, Yancheng He 等

**单位**：*Alibaba Group*


![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/31/1730384064045-0926bb91-f239-481a-acdd-f8d59fe17d0b.png)



本文研究的是**通过二维的多方面的监督信号扩展 DPO 对齐方法，更好地对齐大型语言模型与人类偏好**。现有的DPO方法通常优化一个标量分数或排名奖励，忽略了人类偏好的多维性质。为了解决这一问题，本文提出了一种新的DPO扩展方法 —— **2D-DPO**。

2D-DPO 的核心思想是将偏好优化扩展到两个维度：**片段（segments）**和 **方面（aspects）**。具体来说，本文首先构建了一个名为 HelpSteer-2D 的二维监督数据集，为每个样本都标注了一个**二维评分矩阵**，评估模型回答中的每个片段在多个方面的表现。然后，基于二维的偏好标签，设计了 2D-DPO 损失，进行多片段和多方面的优化。


![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/31/1730384089416-e51d5d71-eae8-4900-a104-59412233cfa1.png)


#### 1. 数据集构建

将每条偏好数据的回答分割为句子级的片段，使用人工制定的评分准则提示 GPT-4 对**每个片段**在**多个方面**（如帮助性、正确性、安全性、完整性和清晰度）进行独立评分，为每个样本标注一个**二维评分矩阵**。

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/10/31/1730384114702-c236a034-f0b2-44fb-b5e9-9997b5b44396.png)


下面是一个 2D 奖励模型与其他类型奖励模型在奖励值分配上的对比示例：

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/31/1730384126143-b0aac5f4-31d2-4c92-bae4-9764e1517530.png)


#### 2. 2D-DPO 损失


![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/10/31/1730384147866-f90b612b-8c7f-4590-8381-94d7ef51b549.png)


在 Token Level 的马尔可夫决策过程（Markov Decision Process，MDP）的视角下，DPO 的损失函数为

$$
\begin{aligned}
&\mathcal{L}_{DPO}(\pi_{\theta};\pi_{ref})= \\
&-\mathbb{E}\left[\log\sigma\left(\beta\sum_{t=0}^{N-1}\log\frac{\pi_\theta(a_w^t\mid s_w^t)}{\pi_{ref}(a_w^t\mid s_w^t)}\right.\right. -\beta\sum_{t=0}^{M-1}\log\frac{\pi_\theta(a_l^t\mid s_l^t)}{\pi_{ref}(a_l^t\mid s_l^t)}\Bigg)\Bigg].
\end{aligned}
$$
其中，$\beta\log\frac{\pi_\theta^*(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{ref}(\mathbf{a}_t|\mathbf{s}_t)}$ 可看作强化学习中的优势函数 $A^*(\mathbf{s}_t,\mathbf{a}_t)$。详细推导见文章附录 A.2 和 A.3。

基于以上优化目标，本文**将不同方面的偏好奖励加权平均**，使用正则化的细粒度奖励作为一个系数，用来调整 Token-Level 的优势函数 $\beta\log\frac{\pi_\theta^*(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{ref}(\mathbf{a}_t|\mathbf{s}_t)}$。此外，**将优势函数项分解为片段级别的累加**，分别进行 Token-Level 的优化。

$$
\begin{aligned}
\mathcal{L}(\pi_\theta&,D)= \\
&- \mathbb{E}_{(\tau_{w},\tau_{l})\sim D}\log\sigma(\beta\sum_{k=0}^{S_{w}-1 }\sum_{t=n_k}^{n_k+l_k}r_{w,k}\log\frac{\pi_{\theta}(\mathbf{a}_{t}^{w}|\mathbf{s}_{t}^{w})}{\pi_{ref}(\mathbf{a}_{t}^{w}|\mathbf{s}_{t}^{w})} \\
&-\beta\sum_{k=0}^{S_l-1}\sum_{t=n_k}^{n_k+l_k}r_{l,k}\log\frac{\pi_\theta(\mathbf{a}_t^l|\mathbf{s}_t^l)}{\pi_{ref}(\mathbf{a}_t^l|\mathbf{s}_t^l)})
\end{aligned}
$$
其中，$r_{w,k}=\mathbf{W}\mathbf{\tilde{r}}_{w,k}$。$W$ 是总和为1的权重，用来对不同方面的奖励进行加权，反映每个方面的重要性；$\mathbf{\tilde{r}}_{w,k}=\{r_{w,k,j}\}_{j=1}^A$，是每个样本中的片段级奖励集合。

考虑到好回答和坏回答中的片段数量可能有显著差异，为了更加关注偏好影响大的关键片段，本文**从好回答中选择得分 最高的 N 个片段，从坏回答中选择得分最低的 N 个片段**，其中 $N = min(S_w, S_l)$，进一步提高了模型对齐训练的效率。

此外，**将所选的好回答和坏回答的片段成对分组，作为偏好对，形成 N 个 Bradley-Terry 模型**，从而在对齐过程中提供更清晰的对比，使模型更容易学习被选择和拒绝响应之间的细粒度差异。（这种重新排列的可行性基于以下事实：单片段 BT 模型的损失可以被视为将其他片段的 $\beta_t$ 设置为 0，如附录A.4所示）

最终，得到了包含细粒度信号的 **Token 级 2D-DPO 损失函数**

$$
\begin{aligned}
&\mathcal{L}_{group}(\pi_{\theta},D)= \\
&-\mathbb{E}_{(\tau_w,\tau_l)\sim D}\left[\sum_{k=0}^{N-1}\log\sigma\left(\beta\sum_{t=n_k}^{n_k+l_k}r_{w,k}\log\frac{\pi_\theta(\mathbf{a}_t^w|\mathbf{s}_t^w)}{\pi_{ref}(\mathbf{a}_t^w|\mathbf{s}_t^w)}\right.\right. \\
&-\beta\sum_{t=n_k}^{n_k+l_k}r_{l,k}\log\frac{\pi_\theta(\mathbf{a}_t^l|\mathbf{s}_t^l)}{\pi_{ref}(\mathbf{a}_t^l|\mathbf{s}_t^l)}\Bigg)\Bigg].& \text{(6)} 
\end{aligned}
$$

#### 3. 实验

在流行基准测试上的实验表明，**2D-DPO** 比使用**标量偏好**或**一维偏好**的方法表现得更好。

![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/10/31/1730384165009-345caaab-34ec-463e-9942-3fa480671b5e.png)


综上所述，2D-DPO通过**引入二维监督信号，实现了对大型语言模型偏好的细粒度优化**。这种方法提高了模型与人类偏好的对齐效果，其监督信号具有更高的质量和可解释性，为细粒度偏好优化研究提供了新的思路。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2410.19720]
- **更多**大模型学习资料，请详见浙大 Daily 实验室 Github 仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：樊怡江，毛玉仁

