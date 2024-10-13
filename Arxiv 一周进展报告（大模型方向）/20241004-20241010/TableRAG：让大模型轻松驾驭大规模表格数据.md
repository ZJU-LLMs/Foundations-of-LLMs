# TableRAG: Million-Token Table Understanding with Language Models

**作者**：*Si-An Chen, Lesly Miculicich , Julian Martin Eisenschlos* 等    

**单位**： *National Taiwan University, Google Cloud AI Research, Google DeepMind, UC San Diego*

下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2024/10/12/1728720711398-392a27f2-a7d3-4856-9a4f-178df1c811c0.png)


本文研究的问题是**如何提高大语言模型在理解和推理大规模表格数据方面的效率和性能**，之前的方法主要包括基于模式的方法和基于行列检索的方法，前者主要关注schema信息，降低了令牌复杂度，但是会丢失一些有价值的单元格数据；后者通过对行和列整体编码来减少令牌数，但是在处理大表格时仍面临着计算和性能挑战，并且行列压缩也会模糊语义信息。本文提出的**TableRAG**的核心思想是**在回答与表格相关的问题时，其实并不需要将整张表格作为输入提供给大型语言模型。**

**TableRAG**框架具体流程如下：

1. **查询扩展（Tabular Query Expansion）:** 与以前工作使用单一查询不同，作者为模式和单元格值生成单独的查询。例如对于下述问题 *What is the average price for wallets?* 使用大语言模型给出可能的列名: *product* 和 *price* 等，以及相关的单元格值 *wallet*,*seller*等
2. **模式检索（Schema Retrieval）：** 作者使用预先训练的编码器$f_{enc}$对上述查询进行编码，并将它们与编码的列名进行匹配，以确定相关性。检索到的数据包括列名、数据类型（**将列转换为整数、浮点数或日期时数据类型；否则，将它们保留为分类列**）和示例值（**除分类列外，使用最大最小值作为示例值，分类列使用三个最常见的类**）保留每个查询的Top-k个相关的结果，并根据相似性排序
3. **单元格检索（Cell Retrieval）：** 在模式检索之后，构建一个不同列值对的数据库$V = U_{ij}(C_{ij},v_{ij})$，在事实表中，**不同值往往会少于单元格总数**，所以这种方式提高了单元格检索的效率。在最坏的情况下，不同值和单元格总数相当，这里引入编码预算$B$, 如果不同值的数量超过$B$，那么将编码限制在出现最频繁的对。
4. **程序辅助求解器（Program-Aided Solver）：** TableRAG与大语言模型代理兼容，它可以通过**编程方式与表进行交互**。这里作者考虑ReAct（实现表QA基准的最先进方法）
![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/12/1728721111599-896d93cb-7774-451c-943f-46230f3f5723.png)
![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/10/12/1728721111602-b9534ed7-8b3c-4eba-891c-e0e486ff1d00.png)





**在实验方面**，作者开发了两个广泛使用的数据集*ArcadeQA*和*BirdQA*，分别来自*Arcade*和*BIRD-SQL*数据集，此外，作者扩展了*TabFact*数据集，使其包括了从$100\times100 到 1000\times 1000$ 的合成表。

作者将TableRAG与四种不同的表输入方法进行了比较。包括**整表输入**、**模式输入**、**随机行采样**和**行列检索**，在表2所示的跨数据集的评估中，TableRAG始终优于其他方法，在*ArcadeQA*和*BirdQA*上都获得了**最高的准确性**。


![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2024/10/12/1728721136084-7c92b760-4840-40b9-9b9a-186a72f62fba.png)


并且本方法也以更简短的prompt长度取得了较高的准确率，如下图所示。


![](https://fastly.jsdelivr.net/gh/bucketio/img18@main/2024/10/12/1728721147144-bf7d9c8f-bfae-41c3-bf46-69c79f4d641d.png)


此外，为了评估小规模*TableQA*数据集的性能，作者使用常用的*WikiTableQA*基准，与相关的基线方法做了比较，获得了更好的结果。

![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/10/12/1728721147146-ef09ec9a-612b-41e6-a8a1-b09b69e5933c.png)

---

- 原文链接: https://arxiv.org/abs/2410.04739
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/XXX

