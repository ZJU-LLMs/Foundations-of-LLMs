# IOPO: Empowering  LLM s with Complex Instruction Following via Input-Output Preference Optimization

**作者**：*Xinghua Zhang, Haiyang Yu 等*

**单位**：*Tongyi Lab*


![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/11/14/1731588823337-4f29693b-f33b-4047-8ee9-7f2649d6df26.png)



本文研究的是**通过输入输出偏好优化（IOPO）方法，提升大型语言模型（ LLM s）遵循复杂指令的能力**。随着 LLM s在各种应用中的广泛使用，指令的复杂性也在迅速增加。然而，现有的复杂指令评估数据有限，且缺乏专门用于提升复杂指令遵循能力的算法。为了解决这一问题，本文提出了**IOPO**方法，并构建了一个名为**TRACE**的基准数据集。

IOPO的核心思想是**同时考虑输入和输出的偏好对**，使 LLM 不仅能够快速适应响应偏好，还能细致地探索指令偏好。具体来说，IOPO不仅将指令作为输入来直接学习响应偏好，还基于相同的响应深入探索指令，以促进对细粒度约束的有效感知。

#### 1. TRACE 基准数据集

本文首先构建了 TRACE 复杂指令数据集，包含120K条训练数据和1K条评估数据。其中每条复杂指令都包含多个约束，涵盖了 26 个**约束维度**和 5 种**约束类型**。


![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/11/14/1731588927191-f35a2fca-a1c5-44db-9775-074213671b01.png)


![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/11/14/1731588937809-f59bc31e-6642-46fc-9287-b2bc2e04c51d.png)


TRACE 复杂指令数据集的构建过程包括以下几个关键步骤：

- **约束分类**：通过 LLM 从大量开源简单指令中**归纳出约束分类**，并由人工专家进一步细化，形成 5 种约束类型和 26 个约束维度。
- **约束扩展**：基于约束分类，通过提示 LLM 将简单指令扩展为**包含多个约束的复杂指令**。
- **指令结构化**：将扩展后的指令文本结构化为任务描述、约束和输入部分。
- **质量控制**：通过提示 LLM 对扩展后的指令进行质量控制，确保指令的有效性。
- **响应生成与评估**：使用 LLM 生成响应，并通过 LLM 评估响应的质量，确保其完全遵循指令中的所有约束。

![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/11/14/1731588890212-4b5b315d-ac85-4860-b3f3-5bcc7e815a17.png)





#### 2. IOPO方法

**IOPO方法通过构建一对包含多项约束内容的指令 $<x_1, x_2>$ 及其对应的响应 $<y_1, y_2>$，其中 $x_2$ 在某些约束上与 $x_1$ 有细微差异，这些差异会导致响应的显著不同。**然后，形成四个输入输出对 $<x_1, y_1>$、$<x_1, y_2>$、$<x_2, y_1>$ 和 $<x_2, y_2>$，构成一个**偏好组对** $G1 ≻ G2$，其中 $G1 = {<x_1, y_1>, <x_2, y_2>}$，$G2 = {<x_1, y_2>, <x_2, y_1>}$。具体的数据构建过程如下：

- **$x_2$ 生成**：通过 “添加”、“删除” 和 “修订” 操作生成新的约束，使响应不再符合新约束，并结合任务描述、新约束和输入形成 $x_2$。
- **$y_2$ 生成**：对于指令 $x_2$，生成相应的响应 $y_2$。
- **响应评估**：评估响应 $y_2$，仅保留完全符合约束的响应。

基于 DPO 的优化目标，通过**最大化偏好组对 $G1 ≻ G2$ 的概率**，推导出 IOPO 的损失函数：

$$
\begin{gathered}
\mathcal{L}_{\mathrm{IOPO}}(\pi_{\theta})=-\mathbb{E}_{x_{1},y_{1},x_{2},y_{2}\sim D} \bigg\{\operatorname{log}\bigg[\sigma\bigg(\frac{1}{2}(2\beta\mathrm{log}\frac{\pi_{\theta}(y_{1}|x_{1})}{\pi_{\mathrm{ref}}(y_{1}|x_{1})}\bigg) \\
-\beta\mathrm{log}\frac{\pi_\theta(y_2|x_1)}{\pi_\mathrm{ref}(y_2|x_1)}-\beta\mathrm{log}\frac{\pi_\theta(y_1|x_2)}{\pi_\mathrm{ref}(y_1|x_2)}+2\beta\mathrm{log}\frac{\pi_\theta(y_2|x_2)}{\pi_\mathrm{ref}(y_2|x_2)} \\
-\left.\beta\mathrm{log}\frac{\pi_{\theta}(y_{1}|x_{2})}{\pi_{\mathrm{ref}}(y_{1}|x_{2})}-\beta\mathrm{log}\frac{\pi_{\theta}(y_{2}|x_{1})}{\pi_{\mathrm{ref}}(y_{2}|x_{1})})\right)\bigg]\bigg\} 
\end{gathered}
$$


#### 3. IOPO方法推导

IOPO方法的核心在于同时优化包含输入和输出的偏好对数据，推导过程类似 DPO：

- **奖励函数表示**：
  奖励函数 $ r(x, y) $ 可以表示为策略模型 $ \pi_r $ 的形式：

$$
r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

​      其中 $ Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right) $。

- **Bradley-Terry模型**：
  Bradley-Terry模型用于估计成对比较的概率：

$$
p(i \succ j) = \frac{p_i}{p_i + p_j}
$$

​     其中 $ p_i $ 是分配给个体 $ i $ 的正实数得分。

- **偏好组对概率**：
  给定一对偏好组 $ \mathcal{G}_1 $ 和 $ \mathcal{G}_2 $，定义 $ p_1 = e^{r(x_1, y_1) + r(x_2, y_2)} $ 和 $ p_2 = e^{r(x_1, y_2) + r(x_2, y_1)} $，则偏好组对概率为：

$$
p(\mathcal{G}_1 \succ \mathcal{G}_2) = \frac{e^{r_{\mathcal{G}_1}}}{e^{r_{\mathcal{G}_1}} + e^{r_{\mathcal{G}_2}}}
$$

​     其中$ r_{\mathcal{G}_1} = r(x_1, y_1) + r(x_2, y_2) $ 和 $ r_{\mathcal{G}_2} = r(x_1, y_2) + r(x_2, y_1) $。

- **优化目标**：
  结合上述公式，优化目标可以进一步推导为：

$$
p(\mathcal{G}_1 \succ \mathcal{G}_2) = \sigma\left(\frac{1}{2}(\Pi_1 + \Pi_2)\right)
$$
​     其中：

$$
\Pi_1 = 2\beta \log \frac{\pi_r(y_1|x_1)}{\pi_{\text{ref}}(y_1|x_1)} - \beta \log \frac{\pi_r(y_2|x_1)}{\pi_{\text{ref}}(y_2|x_1)} - \beta \log \frac{\pi_r(y_1|x_2)}{\pi_{\text{ref}}(y_1|x_2)}
$$

$$
\Pi_2 = 2\beta \log \frac{\pi_r(y_2|x_2)}{\pi_{\text{ref}}(y_2|x_2)} - \beta \log \frac{\pi_r(y_1|x_2)}{\pi_{\text{ref}}(y_1|x_2)} - \beta \log \frac{\pi_r(y_2|x_1)}{\pi_{\text{ref}}(y_2|x_1)}
$$


- **损失函数**：
  最终的损失函数为：

$$
\begin{gathered}
\mathcal{L}_{\mathrm{IOPO}}(\pi_{\theta})=-\mathbb{E}_{x_{1},y_{1},x_{2},y_{2}\sim D} \bigg\{\operatorname{log}\bigg[\sigma\bigg(\frac{1}{2}(2\beta\mathrm{log}\frac{\pi_{\theta}(y_{1}|x_{1})}{\pi_{\mathrm{ref}}(y_{1}|x_{1})}\bigg) \\
-\beta\mathrm{log}\frac{\pi_\theta(y_2|x_1)}{\pi_\mathrm{ref}(y_2|x_1)}-\beta\mathrm{log}\frac{\pi_\theta(y_1|x_2)}{\pi_\mathrm{ref}(y_1|x_2)}+2\beta\mathrm{log}\frac{\pi_\theta(y_2|x_2)}{\pi_\mathrm{ref}(y_2|x_2)} \\
-\left.\beta\mathrm{log}\frac{\pi_{\theta}(y_{1}|x_{2})}{\pi_{\mathrm{ref}}(y_{1}|x_{2})}-\beta\mathrm{log}\frac{\pi_{\theta}(y_{2}|x_{1})}{\pi_{\mathrm{ref}}(y_{2}|x_{1})})\right)\bigg]\bigg\} 
\end{gathered}
$$



#### 4. 实验

在 Trace、IFEval 和 CFBench 三个数据集上的实验表明，IOPO 在复杂指令遵循能力上显著优于现有的 SFT 和 DPO 方法。具体来说，IOPO 在 TRACE 数据集上的单约束和多约束指令遵循能力分别提升了 8.15% 和 2.18%，在 IFEval 和 CFBench 数据集上的表现也有显著提升。

![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/14/1731588985139-6a864ff6-683e-4ee3-9dea-863d8822d3cb.png)


综上所述，IOPO 通过**同时优化输入和输出的偏好，显著提升了 LLM 遵循复杂指令的能力**。这种方法不仅提高了模型在复杂指令场景下的表现，还为复杂指令遵循能力的研究提供了新的思路。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2411.06208]
- **更多**大模型学习资料，详见浙江大学LLMs GitHub仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：樊怡江，毛玉仁