# LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigmfor Long-Context Question Answering

**作者**：*Qingfei Zhao, Ruobing Wang , Yukuo Cen* 等    

**单位**： *Institute of Information Engineering, Chinese Academy of Sciences等*

下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/10/26/1729951624779-4413a16a-c37e-4f51-bdee-de04e93dcfa0.png)


本文研究的问题是**提升大型语言模型在处理长文本问答任务时的表现**。之前的方法包含self-RAG以及cRAG等，其中前者依赖自我反思标记来探索全局信息，但这种依赖可能误删含有重要细节的有效检索块；后者逐个评估块以识别事实细节，却忽略了块之间的关联，当关键细节分散在多个块中时，可能造成重要信息的遗漏。
本文的**LongRAG**的核心思想是**通过增强LLM对长文本中全局信息的理解来增强其识别关键信息的能力**
**LongRAG**框架具体流程如下：<br>
**实验方案**<br>

1. **混合检索器（Hybrid Retriever）:** 采用双向编码器进行快速检索，并通过交叉编码器深入理解语义关系，确保检索效率。
2. **LLM增强信息提取器（LLM-augmented Information Extractor）：** 上述检索到的块被固定的窗口截断，难以携带额外的全局信息。此外，当检索到的数据块来自同一段落p时，它们的顺序可能与p中的原始语义顺序不一致，导致向下游llm提供无序的语义信息。将检索到的短文本片段映射回原始长文本段落，提取包含广泛背景和结构知识的全局信息。
   $f_m(p_{c_1}, p_{c_2}, \cdots, p_{c_k}) \rightarrow p_1, p_2, \cdots, p_{k'}$ <br>
   其中$p_{c_1}$表示检索到的块，之后将映射后的段落连接，并输入给大语言模型总结得到全局信息$I_g$。
   $I_g = \text{LLM}\left(\text{prompt}_e\left(q, p_1\left|p_2\right|\cdots\mid p_{k'}\right)\right)$
3. **CoT引导过滤器（CoT-guided Filter）：** 检索到的块通常包含大量的冗余；有些块甚至可以是完全冗余的。这种复杂性使得很难确定一个块是否包含解决多跳问题的关键信息，为了解决上述问题，作者采用两阶段策略，第一阶段基于检索语义空间生成一个具有全局视角的CoT：
   $CoT = \text{LLM}\left(\text{prompt}_c\left(q, p_{c_1}\left|p_{c_2}\right|\cdots\mid p_{c_k}\right)\right)$
   第二阶段利用全局线索（CoT）指导模型精确筛选出包含关键事实细节的文本块$I_d$。

$$
V(q, p_c, \text{CoT}) = 
\begin{cases} 
\text{True,} & \text{if <support} \\
\text{False,} & \text{otherwise}
\end{cases}
$$

$$
I_d = \{p_c \mid V(q, p_c, \text{CoT}) = \text{True}\}
$$

4. **LLM增强生成器（LLM-augmented Generator）：** 结合全局信息和事实细节生成最终答案，提升回答的准确性。


![](https://fastly.jsdelivr.net/gh/bucketio/img14@main/2024/10/24/1729767150354-85a173d1-201b-4286-9047-dc9ae45379cb.png)

**实验结果**<br>
作者选取了三个多跳数据集*HotpotQA*， *2Wiki-MultiHopQA*，*MusiQue*，并与三类方法进行了比较，*Long-Context LLM Methods*，*Advanced RAG Methods*，*RAG-Base (Vanilla RAG)*，分别提升了**6.94%，6.16%，17.25%** 的准确率
![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/24/1729768767012-48349cb3-37c6-4f13-b272-b9b9323e3441.png)

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2406.15319]
- **更多**文章请详见 Github 仓库: **https://github.com/ZJU-LLMs/Foundations-of-LLMS**
- 本文编辑：刘亚川，毛玉仁

