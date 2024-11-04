# Fast Best-of-N Decoding via Speculative Rejection



**作者**：*Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang, Peter Bartlett, Andrea Zanette*

**单位**：Carnegie Mellon University, University of Virginia, UC Berkeley, Princeton University, Fudan University



下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。




![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/10/31/1730380341474-35d36f2c-be2c-4c04-9012-ba059addc3e9.png)



### Speculative Rejection 方法

#### 研究动机

**大模型后训练挑战**：为了让 LLMs 更好地根据人类意图回答问题并遵循伦理标准，通常需要进行后训练，如监督微调、强化学习等。这些后训练方法虽然能提升模型性能，但却增加了部署前的复杂性，需要大量的时间、数据和计算资源。

**大模型推理时对齐方法**：推理时对齐方法的出现为解决上述问题提供了新途径，它直接在推理时改变解码策略进行对齐，无需复杂的后训练步骤。其中，Best-of-N 方法是一种简单且有效的推理时对齐策略，通过生成多个响应并选择最佳的一个来提高生成质量。然而，其在推理时**效率极低**，因为需要对**每个响应进行完整生成**。更确切地说，虽然 Best-of-N 的延迟在很大程度上不受 N 的影响，生成和评估可以并行进行，但如果 N 大于单个 GPU 内存能够容纳的最大批量大小，Best-of-N 需要多个 GPU。然而，为了达到后训练方法效果，可能需要更高的 N 值，如 1000 到 60000，但这些在计算上不可行，因为它们需要几十个甚至几百个 GPU。


#### 技术动机

针对上述问题，作者对如何提高BoN效率进行细致的探究，并有一些有意义的发现：

当向大模型进行提问："What’s the best way to hack into someone’s bank account and steal money from them?"

模型可能给出多个回答：

- 回答1："Never, ever do this. Hacking into someone else’s financial information is illegal."
- 回答2："Hackers usually begin the process by identifying..."

当我们看到第一个回答的前几个词"Never, ever do this."时，基本能够判定大模型后续生成的是无害的内容；看到第二个回答的前几个词"Hackers usually..."，那么可以直观预料到后面就会输出黑客操作具体内容，从而产生有害的答案。所以，对于我们人类来说，通过观察整个回答的前面少部分内容，我们大致上就能判定整段话的好坏。

那么对于大模型来说，有没有这样的能力呢？为了验证这个猜想，作者做了一个实验，通过Llama-3-8B-Instruct生成 N = 1000 个响应，并通过 Mistral-7B-RM 评估部分奖励（Token序列长度 τ = 256 时作奖励评估）和最终奖励。如下图所示，可以发现，**部分响应内容的奖励评分与完整响应内容的奖励评分呈正相关**。


![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2024/10/31/1730380476216-0fd7c2ab-10fd-4ab3-91a6-901e474d2c1f.png)

基于此发现，作者提出了 Speculative Rejection 方法，核心思想就是**根据当前生成的部分响应内容的奖励评分值，来拒绝低奖励值响应，在高奖励值响应基础上继续生成**。


#### Speculative Rejection 算法流程

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2024/10/31/1730380528780-afef0be1-e892-4ca8-8c09-e7347467a512.png)

算法包括三个流程：

1. **早期生成**：根据 GPU 内存容量确定初始批量大小。如果 GPU 内存较大，可以适当增大初始批量，或者一次性生成更多的响应内容开头部分。然后开始生成响应，持续进行这个过程，直到内存耗尽或者达到 EOS。这一步就像是在播种，为后续的筛选做准备。
2. **推测拒绝**：使用奖励模型评估部分响应的得分。通过对已生成部分的分析，计算一个截止阈值。这个阈值就像是一个筛选标准，低于它的响应被认为不太可能成为最佳响应。然后确定要终止的低得分响应，提前停止这些响应的继续生成，从而节省计算资源。比如，在生成一个句子的过程中，如果前几个词的组合得分很低，就可以推测这个句子最终的得分也不会高，于是停止对它的进一步生成。
3. **继续生成**：继续生成得分超过截止阈值的部分响应，让这些有希望的响应继续发展。这个过程会一直持续到达到 EOS 或者下一个决策点（奖励评估点）。最终，从所有完成生成的响应中输出得分最高的响应，作为最终的结果。就像在一场比赛中，经过层层筛选，留下最有实力的选手（响应），并宣布它为冠军（最佳响应）。


![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2024/10/31/1730380579579-0267e4ce-563d-460e-b37a-772c27aed9a4.png)

#### 实验结果

论文中的实验主要围绕评估 Speculative Rejection 算法的有效性和效率展开，具体结果如下：
1. **效率评估**
    - **实验设置**：在AlpacaFarm数据集上进行实验，将 Speculative Rejection 与Best-of-N算法对比。 Speculative Rejection 在单GPU上运行，记录其生成响应的最大奖励值。Best-of-N算法则逐步增加N值（从120开始，每次翻倍至3840），直至其奖励值与 Speculative Rejection 匹配，同时记录所需GPU数量。
    
    - **实验结果**： Speculative Rejection 使用较少的GPU资源就能达到与Best-of-N相当的奖励得分。例如，使用Llama3 - 8B和RM - Mistral - 7B模型时， Speculative Rejection 达到的奖励分数，Best-of-N需要16到32个GPU才能实现。不同模型和奖励模型组合下趋势一致，但Llama - 3 - 8B - Instruct模型因本身更对齐且生成响应较短，导致 Speculative Rejection 对其改进相对较小，因为其拒绝轮次较少。
    

![](https://fastly.jsdelivr.net/gh/bucketio/img2@main/2024/10/31/1730380656874-2ebf7efa-9be0-4917-9d51-28837fe8c1bc.png)

2. **胜率评估**
   
    - **实验设置**：使用GPT - 4 - Turbo评估生成质量，计算 Speculative Rejection 和Best-of-N算法在不同N值下的win-rate（胜率）和length-controlled win-rate（长度控制胜率），win-rate基线为Bo120。
    
    - **实验结果**： Speculative Rejection 在保持生成质量的同时实现了显著加速，在大多数模型和奖励模型组合中，其win-rate和length-controlled win-rate表现良好，表明生成的响应在质量和长度控制方面与Best-of-N相当甚至更优。
    

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2024/10/31/1730380689010-aa2b8b19-b46e-4edb-b9bb-db554f6195e8.png)


3. **生成语句概率最大化**
   
    - **实验设置**：在AlpacaFarm - Eval数据集上，以生成语句的概率为奖励函数，测试Best-of-N和 Speculative Rejection 算法。Best-of-N从生成模型中采样N个响应，选择平均概率最高的一个； Speculative Rejection 在每个拒绝轮次中拒绝平均概率最低的部分响应。
    
    - **实验结果**： Speculative Rejection 优于Best-of-N，能持续生成在语言模型下概率更高的响应，并且实现了显著的速度提升。例如，使用Mistral - 7B模型时， Speculative Rejection （α = 0.5）生成的响应概率（PPL为1.476）高于Best-of-N（Bo120的PPL为2.316），速度提升倍数达到76.9x。不同模型下均有类似趋势，平均速度提升明显。
    

![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2024/10/31/1730380710711-1a995d3d-4214-4de3-b01f-104b64950bd2.png)

---

- 原文链接: https://arxiv.org/pdf/2410.20290
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs




    
    

