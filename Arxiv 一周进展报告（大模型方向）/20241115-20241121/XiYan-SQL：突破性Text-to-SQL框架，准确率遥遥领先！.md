# XiYan-SQL: A Multi-Generator Ensemble Framework For Text-to-SQL

**作者**：*Yingqi Gao, Yifu Liu等*

**单位**：*Alibaba Group等*



下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/11/21/1732195373231-6a37cd2b-8ce7-4a9c-9420-b7335a42df43.png)

自然语言到SQL（NL2SQL）的技术，即将自然语言查询转换为结构化查询语言（SQL）的能力，是数据库访问方面的重大进步。它极大地促进了非专家和高级用户从大量数据存储中进行数据洞察。然而，尽管大型语言模型（LLM）的进步显著提高了NL2SQL应用的效力和准确性，但现有的解决方案仍面临一些挑战。基于LLM的NL2SQL解决方案通常采用提示词工程（prompt engineering）和有监督的微调（SFT）两种方法。提示词工程利用模型的内在能力，通过优化提示词来生成多样化的SQL查询，但这种方法依赖于多路径生成和self-consistency，带来了巨大的推理开销。基于SFT的方法试图在NL2SQL任务上微调参数规模较小的模型，以生成更可控的SQL查询，但由于其参数量有限，这些方法在复杂的NL2SQL推理和跨领域数据库的迁移方面表现不佳。

为了应对这些挑战，本文提出了XiYan-SQL，这是一个全新的框架，采用多生成器集成的策略来提高候选SQL的质量。本文的动机在于结合提示词工程和SFT方法的优势，生成高质量和多样化的候选SQL查询。具体来说，本文希望通过以下几个方面来提升NL2SQL的性能：

1. **增强模型对数据库结构的理解能力**：提出M-Schema，一种半结构化的数据库schema表示方法，旨在增强模型对于数据库结构的理解能力。
2. **提高生成的候选SQL查询的质量和多样性**：结合ICL方法的巨大潜力和SFT方法的高可控性，提出一系列训练策略，以微调模型生成高质量且具有不同偏好的候选。
3. **优化生成的SQL查询**：通过Refiner纠正逻辑或语法错误来进一步优化每个候选。
4. **识别最佳候选**：微调一个选择模型，用来区分候选SQL查询之间的细微差别，从而选择最终的SQL。


### 方法详细介绍

本文提出的XiYan-SQL框架由三个主要组件组成：Schema Linking、Candidate Generation和Candidate Selection。每个组件都有其独特的方法和策略，以确保生成的SQL查询既高质量又多样化。


![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/11/21/1732195423768-acbd3126-cee3-4abd-b5ba-ca580f7015d0.png)

#### 1. Schema Linking

Schema Linking的目的是将自然语言查询关联到数据库中的元素，包括表、列和值。这一过程由两个主要模块组成：检索模块和列选择器。

- **检索模块**：
  - **关键词和实体识别**：首先通过few-shot的方法提示LLM来识别用户问题中的关键词以及实体。
  - **列检索器**：基于关键词和列描述之间的语义相似性排序，每个关键词检索出Top-K的列。
  - **值检索器**：采用基于局部敏感哈希（LSH）和语义相似性的两阶段检索策略，以识别数据库中的相似值。

- **列选择器**：
  
  - **组织和评估**：从前一步骤中检索到的schema被组织为M-Schema的样式提供给LLM，然后采用few-shot的方式来提示LLM评估每个列与用户查询之间的相关性。
  
  - **选择必要列**：仅选择必要的列供生成器使用，以最小化SQL生成所需的表和列。
  

![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2024/11/21/1732195459346-f056e29c-ed2b-4c61-877d-41c17a591f90.png)

#### 2. Candidate Generation

Candidate Generation采用多生成器来生成高质量和多样化的候选SQL。这一过程分为两个主要部分：微调SQL生成器和ICL SQL生成器。

- **微调SQL生成器**：
  - **两阶段多任务训练**：
    - **基本语法训练**：使用基础和较为单一的SQL模式和语法微调预训练模型，训练目标是开发一个基础模型，激活SQL生成能力，并可以过渡到不同的SQL任务。
    - **生成增强训练**：在第一阶段训练之后，结合各种多任务数据和语法偏好数据来获得增强模型。具体任务包括将问题转换为SQL查询、将SQL转换为问题、从SQL到参考信息（evidence）的任务、SQL判别和再生成任务等。
  - **多样化的语法风格**：利用不同的LLM以多种方式改写原始查询，从而在训练阶段指导模型学习这些数据形式。

- **ICL SQL生成器**：
  - **骨架相似性选择示例**：使用NLTK工具识别问题中的所有命名实体，并将相同类型的命名实体替换为统一的特殊标记。根据修改后的问题计算embedding，并选择与目标问题最相似的前K个训练集样本。
  - **示例选择策略**：对于涉及多个表操作的问题，仅选择涉及多个表操作的SQL查询作为示例。每个问题在生成SQL时最多使用5个示例。

- **SQL Refiner**：
  - **优化生成的SQL**：基于与schema相关的上下文、生成的SQL查询和执行结果（包括潜在错误信息），使模型能够进行第二轮纠正生成。原始SQL和再生成的SQL可以通过选择模型进行最优选择，此过程可以迭代执行。

#### 3. Candidate Selection

Candidate Selection的目的是从候选池中选择正确和合理的SQL查询。这一过程通过计算SQL执行结果的一致性并对其进行分组，利用选择模型根据提供的上下文信息和候选集选择最合理的候选。

- **选择模型**：
  - **微调选择模型**：专门微调一个模型作为SQL选择器，来更好地区分候选SQL查询的细微差别。
  - **训练数据增广**：对选择模型的训练数据进行了特定增广，以与候选SQL的不同语法风格偏好保持一致。
  
  

### 实验与评估

文章的实验部分旨在全面评估XiYan - SQL框架在不同数据库和数据集上的性能表现，通过与多种方法对比以及消融实验，验证其有效性和各组件的作用。
1. **实验设置**
    - **数据集**：使用了Spider、Bird、SQL - Eval和NL2GQL四个数据集，涵盖关系型和非关系型数据库，具体信息如下：
        - **Spider**：包含1981个问题，使用SQLite，涉及39个数据库，是广泛认可的跨域数据集。
        - **Bird**：有1534个问题，基于SQLite，包含11个数据库，由于测试集不可用，在开发集上进行实验。
        - **SQL - Eval**：为开源PostgreSQL评估数据集，包含304个问题和11个数据库，由Defog发布，基于Spider构建。
        - **NL2GQL**：基于图数据库，包含288个问题和3个数据库，用于评估XiYan - SQL在非关系型数据集上的有效性。
    - **评估指标**：采用执行准确率（Execution Accuracy，EX）来评估生成SQL查询的有效性，通过比较预测SQL查询和参考SQL查询在特定数据库实例上的执行结果进行计算。
    
2. **实验结果**
   
    - **Bird开发基准结果**：XiYan - SQL在Bird开发基准上达到72.23%的准确率，高于GPT - 4o的57.95%。与其他先进方法相比，CHASE - SQL框架采用多链思维提示技术和二进制投票机制，准确率为73.14%，XiYan - SQL通过在5个候选中投票获得了有竞争力的性能。同时，基于SFT的方法ExSL + Granite - 34B - Code以72.43%的准确率位居第二，表明小模型通过先进训练技术也能有效生成复杂SQL查询，XiYan - SQL结合了SFT和ICL方法平衡了测试时间和系统整体性能。
    

	![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/11/21/1732195488480-c80ebcba-988a-4592-855d-988fa0be6cb7.png)

    - **Spider数据集结果**：在Spider数据集上，GPT - 4o准确率为83.54%，XiYan - SQL刷新了当前最优执行准确率，达到89.65%，相比之前领先模型仅有0.05%的边际优势，表明底层骨干模型能力的提升对性能有显著贡献。

	![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/11/21/1732195497169-313c8d64-3db3-4ec5-9fcd-93faab3b38fd.png)

    - **SQL - Eval数据集结果**：SQL - Eval提供多个参考SQL查询，XiYan - SQL选择第一个作为计算指标的真实值，在该数据集上获得了69.86%的最高得分，大幅领先于SQL - Coder - 8B（60.20%），比闭源骨干模型高出2 - 5个百分点，体现了XiYan - SQL在PostgreSQL上SQL生成的通用性。

	![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2024/11/21/1732195504224-b2cdd18f-2dea-4785-898a-386b4a4d5743.png)

    - **NL2GQL数据集结果**：在评估XiYan - SQL在非关系型图数据集有效性的实验中，从NL2GQL数据集中抽取288个示例，XiYan - SQL实现了41.20%的执行准确率，远超GPT - 4o（4.86%）、DeepSeek（18.06%）、Gemini 1.5 Pro（6.60%）和Claude 3.5 Sonnet（3.12%），表现出最佳性能。

	![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2024/11/21/1732195510140-752d884d-7096-4230-b52b-c0423c9813d6.png)


3. **消融实验**
    - **M - Schema**：在Bird开发基准上进行消融实验，使用DeepSeek、Claude 3.5 Sonnet、Gemini 1.5 Pro和GPT - 4o四个强大的LLM作为NL2SQL生成器，比较不同模式表示对端到端SQL生成性能的影响。结果显示，与DDL Schema相比，使用M - Schema作为数据库模式表示时，所有四个模型的性能均有提升，平均提高2.03%。尽管M - Schema与MAC - SQL Schema结构相似，但GPT - 4o和Claude 3.5 Sonnet分别有0.65%和0.78%的性能提升，而DeepSeek和Gemini 1.5 Pro有轻微的准确率下降（分别为0.13%和0.26%），表明M - Schema是比DDL Schema和MAC - SQL Schema更好的表示方法，具有强大的通用性。
    

	![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/11/21/1732195571906-2d90aa10-8130-4d20-8acd-4932b699975d.png)

    - **Schema Linking**：通过消融实验评估模式链接的有效性，利用召回率和精确率指标评估基于校正SQL查询（作为真实值）选择列的正确性，使用GPT - 4o作为NL2SQL生成器分析模式链接对端到端EX指标的影响。实验结果表明，不使用模式链接时，精确率为10.14%，执行准确率为57.95%；采用报告中的模式链接方法后，精确率达到74.74%，召回率略有下降，执行准确率提高了2.15%，证明了模式链接的有效性。
    
	![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/11/21/1732195617753-de3a26c8-2d10-4339-af0a-589b0b7150e5.png)

	- **Candidate Generation和Selection**：对XiYan - SQL进行了多种消融实验，以评估候选生成和选择的有效性及影响。具体如下：
	
	![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/11/21/1732195626430-6bb96bd5-257e-4255-ae84-c9399379a8c1.png)
	
	 - 去除微调生成器后，XiYan - SQL性能显著下降，表明微调生成器能生成高质量和多样的候选SQL查询。
	 - 移除ICL生成器和优化器也导致性能下降。
	 - 在候选选择方面，不使用选择模型而仅依赖自一致性进行候选选择时，XiYan - SQL性能降低约三个百分点，突出了选择模型的有效性。
	 - 当SQL候选数量增加到五个时，XiYan - SQL的准确率进一步提高到72.23%。





---


- 查看 Arxiv 原文链接请点击“**阅读原文**”
[https://arxiv.org/abs/2411.08599]
- **更多**模型学习资料，请详见浙大 Daily 实验室 Github 仓库：**https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：张超 毛玉仁



