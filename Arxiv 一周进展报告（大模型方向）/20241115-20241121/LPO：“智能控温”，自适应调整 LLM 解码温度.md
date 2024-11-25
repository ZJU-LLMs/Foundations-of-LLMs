# Adaptive Decoding via Latent Preference Optimization

**作者**：*Shehzaad Dhuliawala, Ilia Kulikov 等*

**单位**：*Meta AI*

![](https://fastly.jsdelivr.net/gh/bucketio/img14@main/2024/11/24/1732445635647-db5c8266-c4ff-4cce-8c66-47b1fb898fd1.png)



本文研究的是**在语言模型解码过程中动态选择采样温度以优化性能**。现有的语言模型在解码时通常使用固定的温度参数，这可能导致在需要创造性和事实准确性的任务中表现不佳。为了解决这一问题，本文提出了一种新的方法 —— **Adaptive Decoding**，通过在模型中添加一个可学习的层来动态选择解码温度。

Adaptive Decoding 的核心思想是**在推理时动态调整采样温度**，以优化模型在不同任务中的表现。具体来说，本文引入了一个名为 **AdaptiveDecoder** 的模块，该模块可以根据上下文动态选择最优的温度值。为了训练这个模块，本文提出了一种新的训练方法 —— **Latent Preference Optimization (LPO)**，用于优化离散的潜在变量（如温度选择）。



### 1. AdaptiveDecoder 模块

AdaptiveDecoder 模块是一个小型的神经网络，可以附加在现有的语言模型上，文中使用的是一个三层的MLP 。它接收最后一层的隐状态作为输入，并输出一个概率分布，用于选择不同的温度值。具体来说，AdaptiveDecoder 模块可以通过以下方式生成下一个 token：

- **序列级 AdaptiveDecoder**：为整个响应预测一个单一的温度值。
  
$$
\tau\sim\mathrm{AdaptiveDecoder}(h_T)\\y_{t+1}\sim\mathrm{Softmax}(Wh_t/\tau)\quad\mathrm{for}\quad T\leq t<T^{\prime}
$$

- **Token 级 AdaptiveDecoder**：为每个生成的 token 预测一个新的温度值。

$$
\tau_t\sim\mathrm{AdaptiveDecoder}(h_t)\\y_{t+1}\sim\mathrm{Softmax}(Wh_t/\tau_t)\quad\mathrm{for}\quad T\leq t<T^{\prime}
$$

其中， $\boldsymbol{y}^c$ 为选中的响应， $\boldsymbol{y}^r$ 为拒绝的响应， $\boldsymbol{\tau}^c$ 为选中的温度， $\boldsymbol{\tau}^r$ 为拒绝的温度， $P(\cdot)$ 为概率分布， $P_{\text{ref}}(\cdot)$ 为参考模型的概率分布， $\beta$ 为超参数，控制 KL 散度项， $\sigma$ 为 sigmoid 函数。

![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/11/24/1732445779634-ee1fb81e-7fff-4fc4-9bd6-e4f64e2e29d3.png)


### 2. Latent Preference Optimization (LPO)

为了训练 AdaptiveDecoder 模块，本文提出了一种新的偏好优化方法 —— LPO。LPO 通过生成多个响应并对其进行评分（使用**奖励模型**或者通过**结果正确性**进行评分），构建选中和拒绝的偏好对，然后从不同角度构建 DPO 损失来学习 AdaptiveDecoder 模块的最优参数。LPO 损失的具体形式如下：

- **温度作为 Token**：将温度选择视为另一种 token，直接应用 DPO 损失。包含文本 Token 损失项和温度损失项。

$$
\mathcal{L}_{\mathrm{LPO}}=-\log\sigma\left[\beta\log\frac{P(\boldsymbol{y}^{c})}{P_{\mathrm{ref}}(\boldsymbol{y}^{c})}-\beta\log\frac{P(\boldsymbol{y}^{r})}{P_{\mathrm{ref}}(\boldsymbol{y}^{r})}+\beta\log P(\boldsymbol{\tau}^{c})-\beta\log P(\boldsymbol{\tau}^{r})\right]
$$

  其中，$\boldsymbol{y}^c$ 为选中的响应，$\boldsymbol{y}^r$ 为拒绝的响应，$\boldsymbol{\tau}^c$ 为选中的温度，$\boldsymbol{\tau}^r$ 为拒绝的温度

- **温度作为 Token（分离）**：将温度选择视为 token，但仅关注 AdaptiveDecoder 模块的输出，只有温度损失项。

$$
\mathcal{L}_{\text{LPO}} = -\log \sigma \left[ \beta \log P(\boldsymbol{\tau}^c) - \beta \log P(\boldsymbol{\tau}^r) \right]
$$

- **温度作为潜在变量**：将温度选择视为模型的内部变量，通过边际化温度变量来优化 token 概率。生成数据中使用的实际温度 $\boldsymbol{\tau}^c$ 和 $\boldsymbol{\tau}^r$ 在这里无关，因此减少了训练过程中采样温度引起的噪声。

$$
\mathcal{L}_{\text{LPO}} = -\log \sigma \left[ \beta \sum_t \log \frac{\sum_{\tau} P(y^c_t | \tau) P(\tau)}{\sum_{\tau} P_{\text{ref}}(y^c_t | \tau) P_{\text{ref}}(\tau)} - \beta \sum_t \log \frac{\sum_{\tau} P(y^r_t | \tau) P(\tau)}{\sum_{\tau} P_{\text{ref}}(y^r_t | \tau) P_{\text{ref}}(\tau)} \right]
$$

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/11/24/1732445831312-209767f6-36a5-46fd-87b1-3657c5a3ff76.png)


### 3. 实验

1. **减少 N-gram 重复**：  $AdaptiveDecoder_{tok}$ 能学习选择更高温度避免重复，有效减少 42% 的重复率。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/11/24/1732445868063-e66e61e0-9f39-44be-9803-305ab55ee8a5.png)

2. **UltraMathStories 任务**：包含数学、创意写作和一般指令等子任务， $AdaptiveDecoder$ 在该任务上**优于实验中所有固定温度解码**，能根据不同子任务选择合适温度，在该任务上 $AdaptiveDecoder_{seq}$ 表现更好。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/24/1732445886073-8c8d70c9-6ab4-4f9e-96eb-4c808615cbda.png)

3. **受限创意写作**： $AdaptiveDecoder_{tok}$ 可**在单个响应的不同 Token 处动态调整温度**，满足约束的同时，提高故事质量。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/11/24/1732445906806-41c2d467-4587-4391-9dc0-bad22a9ef250.png)

4. **多数投票**： $AdaptiveDecoder_{tok}$ 能学习**为推理链的不同部分分配合适温度**，在单响应和多数投票设置中表现更好。

   ![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/24/1732445929602-c8dae492-71e2-4df3-8e93-363b2d1915b7.png)

综上所述，Adaptive Decoding 通过**动态调整采样温度，实现了对大型语言模型在不同任务中的细粒度优化**。这种方法提高了模型在各种任务中的表现，为语言模型解码策略的研究提供了新的思路。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2411.09661]
- **更多**大模型学习资料，详见浙江大学LLMs GitHub仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：樊怡江，毛玉仁