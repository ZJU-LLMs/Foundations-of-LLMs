- ## Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation

  作者：Chengyue Wu Xiaokang 等

  单位：DeepSeek-AI  The University of Hong Kong等

  ## 研究框图

  下图给出此文的的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。
  ![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/10/27/1729995459777-fb317cb2-1f65-4ddb-b265-8607204ee4a5.png)

  本文研究内容是如何将多模态理解和生成统一到一个模型中，此前的研究使用单一的视觉编码器来同时处理两项任务的输入，然而，多模态理解和生成任务所需的表征存在显著差异。在多模态理解任务中，视觉编码器的目的是提取高层语义信息（例如图像中的对象类别或视觉属性）。理解任务的输出不仅涉及从图像中提取信息，还包括复杂的语义推理。相比之下，在视觉生成任务中，主要关注的是生成图像的局部细节和保持全局一致性。在这种情况下，所需的表征必须是能够表达细粒度空间结构和纹理细节的低维编码。为了解决这一问题，本文提出了Janus，一个解耦视觉编码的统一多模态框架，专用于多模态理解和生成。

  其框架图如下图所示：


  ![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/10/27/1729995498985-7523eb5d-4cbf-4c5d-acfe-87ab0c172f0e.png)


  ##### 具体而言：

  **在文本理解方面**，使用LLM的tokenizer将文本转换为离散ID，并获得与每个ID对应的特征表示。

  **在多模态理解中**，采用SigLIP 编码器从图像中提取高维语义特征。这些特征从二维网格展平为一维序列，并通过一个adaptor将这些图像特征映射到LLM的输入空间。

  **在视觉生成任务中**，使用 VQ tokenizer将图像转换为离散ID。将ID序列展平为一维后，使用adaptor将与每个ID对应的代码簿嵌入映射到LLM的输入空间。然后，我们将这些特征序列连接起来，形成一个多模态特征序列，并将其输入LLM进行处理。

  在纯文本理解和多模态理解任务中，LLM的内置预测头用于文本预测，而在视觉生成任务中，使用一个随机初始化的预测头进行图像预测。

  ##### 模型训练上，分为三个阶段，如下图所示：

  ![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/10/27/1729995530775-5cdb018d-6709-41da-bc1c-038cae0536cf.png)


  **第一阶段：训练adaptor和Image Head**

  此阶段的主要目标是在嵌入空间中建立**视觉和语言元素之间的概念联系**，使LLM能够理解图像中的实体，并具备初步的视觉生成能力。在此阶段，**冻结视觉编码器和LLM**，仅更新两个adaptor和Image Head中的可训练参数。

  **第二阶段：统一预训练**

  在该阶段，使用**多模态语料**进行统一预训练，使模型Janus学习多模态理解和生成能力。在这一过程中们解冻LLM并使用所有类型的训练数据：纯文本数据、多模态理解数据和视觉生成数据。

  **第三阶段：监督微调**

  在此阶段，通过指令微调数据对预训练模型进行**监督微调**，以提升其**指令跟随能力和对话能力**。在微调过程中，微调除Gen. Encoder 外的所有参数，并侧重于监督答案的生成，同时屏蔽系统和用户的提示信息。

  ##### 实验方面，如下表所示：

  在**多模态理解性能**上，Janus 在相同规模的模型中取得了整体最佳结果，相较于之前最佳的统一模型Show-o ，Janus 在MME和GQA数据集上分别提升了41%（949 → 1338）和30%（48.7 → 59.1）,同时，Janus 在多个数据集（如POPE、MMBench、SEED Bench 和 MM-Vet）上超越了LLaVA-v1.5 (7B)。

  在**视觉生成性能**上，Janus 在GenEval上取得了61%的总体准确率，超过了之前最佳的统一模型Show-o (53%) 和一些流行的仅生成方法，如SDXL (55%) 和 DALL-E 2 (52%）。

  ![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/10/27/1729995544210-366b640f-50b1-416a-b74c-7ec3e278731d.png)

  ![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/27/1729995551472-4ff10a8c-ad74-46f9-a67f-9871132ae71b.png)


  综上所述，本文介绍了Janus，这是一种简单、统一且可扩展的多模态理解与生成模型。Janus的核心理念是将多模态理解和生成的视觉编码解耦，以缓解理解和生成对视觉编码器提出的不同需求所引发的冲突。大量实验验证了Janus的有效性及其领先的性能。

  ---

  - 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/pdf/2410.13848]
  - **更多**文章请详见 Github 仓库: 
    **https://github.com/ZJU-LLMs/XXX**
  - 本文编辑：胡中豪，毛玉仁
