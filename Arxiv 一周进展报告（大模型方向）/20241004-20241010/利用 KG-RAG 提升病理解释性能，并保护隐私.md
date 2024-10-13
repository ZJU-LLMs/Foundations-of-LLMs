# LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies

**作者**：*Ameer Hamza, Abdullah, Yong Hyun Ahn, Sungyoung Lee, Seong Tae Kim*

**单位**： *Kyung Hee University, Republic of Korea*



下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/10/13/1728802965111-78daa669-569a-4753-a8a4-4c153471a5c4.png)

## 方法详解


该框架将一个预训练的 LLaVA 模型与一个 CLIP ViT - L 视觉编码器相结合，以提取视觉特征，然后将这些视觉特征投影到语言模型的嵌入空间中。KGR 模块使用 MedCLIP 将输入图像映射到一个共享的潜在空间，并通过 FAISS 库检索相关的 KG 三元组。这些三元组提供了特定领域的上下文，增强了胸部病理准确且信息丰富的自然语言解释（NLE）的生成。模块化设计允许与其他架构（如 Med - XPT 和 Bio - LLaVA）无缝集成，确保在不同的视觉 - 语言任务中具有灵活性和适应性。


![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/10/13/1728803120083-0144cfe8-368f-40c5-ad09-ff790f3ae8ca.png)






### 病理分类任务
运用 MLP 处理医学视觉模型提取的视觉特征，对 10 种病理按存在可能性分为**阴性**、**不确定**、**阳性**三个确定性水平进行预测。得到分类后的结果，跟KG-RAG检索到的知识一起输给大模型。

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/10/13/1728803233951-22468a86-e473-419c-b0d3-a25343a6a34b.png)



### 知识图谱检索
为解决传统 RAG 系统可能暴露患者敏感信息的风险，提出基于 KG 的 RAG 方法。该方法使用由通用医学术语、实体及其相互关系组成的 KG，**避免直接涉及患者特定细节**，降低隐私暴露风险。

**数据存储的构建**

- 构建一个包含 KG 三元组的数据存储，这些三元组来自 MIMIC - CXR 训练集，通过 RadGraph 模型提取。形式为：“疾病实体 - 关系描述 - 相关实体” 的形式，如 “肺炎 - 暗示 - 肺部阴影” 等
- 仅使用具有 “暗示” 关系的三元组，因为它们与解释病理更直接相关。
- 三元组的嵌入使用 MED-CLIP 模型生成，并仅存储文本信息，排除图像特征，以便实现跨模态检索。

**知识检索过程**

- 对于每个查询图像，使用 MED-CLIP 模型提取视觉特征，该模型将视觉和三元组特征映射到统一特征空间。
- 通过计算查询图像视觉特征与存储的三元组嵌入之间的余弦相似度，从 KG 数据存储中检索出最相似的前 k 个三元组。

### 视觉与语言模型

**视觉模型**：MedCLIP 和 ViT - L/14 CLIP

**语言模型**：LLaVA 或者 Viccuna 

最后集成信息输入，向语言模型提供病理及其确定性水平（不确定、阳性）以及检索到的知识。这些元素被集成到一个结构化的提示模板中。然后将这个提示输入到解码器中，解码器根据图像特征、病理和检索到的知识生成自然语言解释（NLE）。

### 实验结果

**与其他方法比较**

在 MIMIC - NLE 测试集上，将 KG - LLaVA 框架与 RATCHET、TieNet、DPT 等方法比较。KG - LLaVA 在 AUC（83.0）、BLEU - 4（7.2）、METEOR（15.1）、ROUGE - L（25.0）和 CIDEr（62.2）等指标上均优于现有方法，表明其在准确分类和生成胸部病理相关解释方面的有效性。

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/10/13/1728803608036-8cb96249-43ff-4b12-984e-58d77a486b4a.png)


**不同 LLMs 比较**

对 KG - LLaVA、Med - XPT 和 Bio - LLaVA 三个框架比较。KG - LLaVA 在 BLEU - 4、METEOR 和 ROUGE - L 指标上表现最佳，能生成准确且内容丰富的解释；

**不同 RAG 方法影响**

比较 Med - XPT 和 KG - LLaVA 在无 RAG、基于标准 NLE 的 RAG 以及基于 KG 检索模块的 RAG 三种配置下的性能。在 KG 配置下性能提升最显著，KG - LLaVA 在多个指标上领先，Med - XPT 在 CIDEr 指标上表现出色，证明了 KG - RAG 模块的重要性。

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/13/1728803689789-8e3ac7b8-548d-40ab-8a42-55608b6b5cbd.png)




---

- 原文链接: https://arxiv.org/abs/2410.04749
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/XXX
