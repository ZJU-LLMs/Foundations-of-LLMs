# Compressed Chain of Thought: Efficient Reasoning through Dense Representations

*Jeffrey Cheng, Benjamin Van Durme 等*

*Hopkins University 等*

本文提出了**压缩链式思维（Compressed Chain-of-Thought, CCoT）**框架，用于提高大语言模型的推理效率和性能。传统链式推理（CoT）通过显式生成推理链来分步完成复杂问题，但需要较高的生成成本和时间。CCoT通过生成压缩的内容丰富的连续表征，减少推理链长度，从而显著降低计算成本，同时保持推理性能。

## 研究内容

研究如何通过压缩的连续表征生成更高效的链式推理机制，从而在推理性能与生成效率之间实现更优的平衡。

## 研究动机

传统链式推理（CoT）虽然能够提升推理能力，但需要生成完整的推理链，导致计算开销大、生成时间长，限制了实际应用中的效率。

![image-20250102161908227](/Users/yuhang/Library/Application Support/typora-user-images/image-20250102161908227.png)

## 技术动机

通过将显式的语言推理链压缩为内容丰富的连续表征（连续嵌入空间），可以减少推理步骤，降低生成成本，同时保留推理链的信息，从而在性能和效率上取得平衡。

## 解决方案

**压缩链式思维（CCoT）框架**旨在通过压缩的内容表征生成推理链，以降低推理开销并提高推理性能。其核心方法包括两个主要模块：**CCOT模块**和**DECODE模块**。

![image-20250102154901997](/Users/yuhang/Library/Application Support/typora-user-images/image-20250102154901997.png)

#### 1. **问题背景与符号**
假设语言模型为预训练的因果解码器 $LM_\theta$，模型权重为 $\theta$。任务输入为一个查询 $w_{1:n}$，其完整推理链为 $t_{1:m}$，目标答案为 $a_{1:o}$。

推理链的压缩比例用 $r$ 表示（$0 < r < 1$），压缩后推理链的长度为 $k = \lceil r \cdot m \rceil$。

#### 2. **CCOT 模块**
**目标**：生成内容丰富的连续嵌入作为压缩推理链的表征，记为 $\hat{z}_{1:k}$，以近似完整推理链的隐状态。

**训练流程**：

1. **生成隐状态**：
   
   - 对输入 $[w_{1:n}; t_{1:m}; a_{1:o}]$ 进行嵌入和隐状态计算：
     $$
     [\bar{w}_{1:n}; \bar{t}_{1:m}; \bar{a}_{1:o}] = \text{EMBED}_\theta([w_{1:n}; t_{1:m}; a_{1:o}])
     $$
     
     $$
     [\hat{w}_{1:n}; \hat{t}_{1:m}; \hat{a}_{1:o}] = \text{ATTN}_\theta([\bar{w}_{1:n}; \bar{t}_{1:m}; \bar{a}_{1:o}])
     $$
     
     
   
2. **选择金标准子集隐状态**：
   
   - 使用评分器（scorer）模块选择 $k$ 个隐状态 $I$ 索引，得到子集隐状态 $z_{1:k} = \hat{t}_I$，作为金标准。
   - 评分器通过线性层实现：
     $$
     I = \text{SCORER}(\hat{t}_{1:m})
     $$
   
3. **近似生成子集隐状态**：
   
   - CCOT模块 $\phi$ 接受输入 $\hat{z}_{0:k-1}$ 生成新的隐状态表征：
     $$
     \hat{z}_{1,k} = \text{CCOT}_\phi(\hat{z}_{0,k-1})
     $$
     其中 $\hat{z}_{0,k-1}$ 是查询 $w_{1:n}$ 的最后一个隐状态。
   
4. **分层训练**：
   
   - 对每一层 $l$，优化目标为缩小生成隐状态与目标隐状态之间的均方误差：
     $$
     \text{LOSS}_\phi = \frac{1}{k} \sum_{i=1}^k \frac{\text{MSE}(z_i^l, \hat{z}_i^l)}{\sigma^2(z_i^l)}
     $$
   - 在训练第 $i$ 层时，只更新该层的参数，冻结之前层的参数，分层细化隐状态近似。

#### 3. **DECODE 模块**
**目标**：利用生成的压缩推理链 $\hat{z}_{1:k}$ 和查询 $w_{1:n}$，解码最终答案 $a_{1:o}$。

**训练流程**：

1. **生成压缩推理链**：
   
   - 使用已训练的 CCOT 模块 $\phi$ 逐步生成推理链表征：
     $$
     \hat{z}_i = \text{CCOT}_\phi(\hat{z}_{i-1})
     $$
   
2. **解码答案**：
   
   - 利用 $w_{1:n}$ 和 $\hat{z}_{1:k}$，基于条件生成模型 $\psi$ 解码答案：
     $$
     p(a_i | a_{1:i-1}, \hat{z}_{1:k}, w_{1:n}) = \text{DECODE}_\psi(\hat{a}_{1:i-1}, \hat{z}_{1:k}, \hat{w}_{1:n})
     $$
   
3. **优化目标**：

- 对解码过程中的生成分布，优化交叉熵损失：
  $$
  \text{LOSS}_\psi = -\sum_{i=2}^o \log p(a_i | a_{1:i-1}, \hat{z}_{1:k}, w_{1:n})
  $$

4. **终止条件预测**：

- 增加一个二分类模块 $\text{END}_\psi$，预测是否需要生成更多的推理链表征，直到达到终止条件。

#### 4. **推理阶段**
推理过程的两个主要阶段：
1. **生成推理链表征**：
   
   - 使用 CCOT 模块自回归生成 $\hat{z}_{1:k}$，每一步基于上一层的隐状态生成下一步：
     $$
     [\hat{w}_{1:n}; \hat{z}_{1:k}] = \text{CCoT}_{\phi}([\bar{w}_{1:n}; \hat{z}_{1:k-1}])
     $$
   
2. **生成答案**：

- 在生成的推理链和原始查询的条件下，通过 DECODE 模块逐步生成答案：
  $$
  \hat{a}_{1:o} = \text{DECODE}_{\psi}([\bar{w}_{1:n}; \hat{z}_{1:k}; \bar{a}_{1:o-1}])
  $$

### 实验结果

本文在 **GSM8K** 数据集上测试模型性能，该数据集包含数学推理任务，要求语言模型具备多步推理能力。模型选择了 LLAMA2-7B-CHAT，使用 **LoRA**（低秩适配）技术对$CCOT_{\phi}$（rank=128）和$DECODE_{\psi}$（rank=64）的参数进行微调。以下是主要实验结果：

<img src="/Users/yuhang/Library/Application Support/typora-user-images/image-20250102161512891.png" alt="image-20250102161512891" style="zoom:50%;" />

以上结果表明，**CCoT** 在 $r=0.05,0.10$ 的低压缩比情况下，以显著低于传统CoT的生成时间实现了较好的准确率，体现了其高效性。

综上， **压缩链式思维（Compressed Chain-of-Thought, CCoT）** 框架，作为传统链式推理（CoT）的高效替代方案。通过生成压缩推理链实现了推理效率与性能的平衡，为复杂推理任务提供了更高效的解决方案，同时保留了推理链的内容表征，具有良好的扩展性和适用性。

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2412.13171]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
  https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：葛宇航，毛玉仁