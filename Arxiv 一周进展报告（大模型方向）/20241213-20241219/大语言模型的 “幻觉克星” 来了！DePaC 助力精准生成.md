# Dehallucinating Parallel Context Extension for Retrieval-Augmented Generation

_Zexiong Ma, Shengnan An 等_

_Peking University; Microsoft; Xi’an Jiaotong University 等_

本文提出了DePaC (**De**hallucinating **Pa**rallel**C**ontext Extension) 方法，DePaC的核心思想是通过采用上下文感知的负训练和信息校准聚合，减少了检索增强生成模型中的幻觉，显著提高了生成响应的准确性和可靠性。

## 研究内容

提高基于RAG的大型语言模型（LLMs）的准确性和可靠性

## 研究动机

之前的方法包括增强检索器性能，迭代RAG以及提示工程等，这些方法不能有效解决信息遗漏的的问题。

## 技术动机

通过负训练，让模型学会在上下文与问题不相关时拒绝回答；通过测量文档提供的信息增量，让模型优先考虑包含有用信息的上下文窗口。

## 解决方案

**背景：**并行上下文扩展（PCE）：PCE的核心思想是将来自多个上下文窗口的信息聚合到一个统一的表示空间中。

​	给定一个问题Q和一系列相关文档${d_1,d_2...d_i}$，PCE首先计算每个上下文窗口的输出分布：$ p_{i,j} = p_\theta(\cdot \mid d_j \oplus \mathcal{Q} \oplus A_{1:i-1})$ ,然后，	这些分布通过某个聚合函数被聚合为单一分布$P_i = AGG(P_{i,1},P_{i,2}..)$，本文使用的聚合函数是最低不确定性聚合函数
​        $p_i = \arg\min_{p_{i,j}} H(p_{i,j})$       $H(p_{i,j}) = -p_{i,j}(\log p_{i,j})^T.$

![method](C:\Users\lyt\Desktop\method.png)

1. **上下文感知负训练（Context-aware Negative Training）：** 它明确地训练主干模型来确定一个问题是否可以基于所提供的文档进行回答。如果没有，我们希望这个模型能拒绝回答这个问题，而不是产生幻觉。

2. **信息-校准聚合（Information-Calibrated Aggregation）：** 仅仅测量最终输出分布的不确定性可能会受到事实遗漏幻觉的严重影响。所以需要测量每个上下文窗口相对于上下文无关的输出分布的信息增量，以反映检索到的文档所提供的信息增量。这里使用KL散度来度量信息增量：$\Delta(\mathbf{p_{i,j}}, \mathbf{p_{i,c}}) = D_{KL}(\mathbf{p_{i,j}} \parallel \mathbf{p_{i,c}})$

3. 将上述两个方法代入最低不确定性函数就可以得到：

   $ p_i = \arg\min_{p_{i,j}} C(p_{i,j}, p_{i,c}) - \gamma \cdot \mathbb{I}(\arg\max_k p_{i,j}^k = t_d)$

   $C(p_{i,j}, p_{i,c}) = H(p_{i,j}) - \beta \cdot \Delta(p_{i,j}, p_{i,c})$

## 实验结果

- DePaC在多个RAG任务上显示出了显著的性能提升。这些任务包括信息检索任务和基于文档的问题回答任务（DocQA）。性能的提升表明DePaC能够有效地利用并行上下文信息，并生成更准确的回答。

![infores](C:\Users\lyt\Desktop\infores.png)

![tableres](C:\Users\lyt\Desktop\tableres.png)

综上，该研究通过引入内部思考过程，提升了LLM在广泛任务中的指令遵循能力，而无需额外的人类数据。 

---

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/html/2412.14905]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
  https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑刘亚川，毛玉仁