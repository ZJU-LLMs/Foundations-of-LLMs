# TREEBON: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling



**作者**：*Jiahao Qiu, Yifu Lu, Yifan Zeng, Jiacheng Guo, Jiayi Geng, Huazheng Wang, Kaixuan Huang, Yue Wu, and Mengdi Wang*

**单位**：Princeton University, University of Michigan, Oregon State University



下图给出此文的整体逻辑框架。首先，对文章进行一句话总结，然后简要介绍研究内容、研究动机、技术动机、解决方案以及优势与潜力，以便读者快速了解文章脉络。


![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2024/10/25/1729870055198-94516a51-2f3a-49e6-b9c9-68f8b4b8b617.png)

论文提出了 TreeBoN 方法，通过将投机树搜索策略与 Best-of-N（BoN）采样相结合，并利用从 DPO 隐式奖励修改而来的 token 级奖励引导，来提高大语言模型推理时的对齐性能和效率。以下是 TreeBoN 方法的详细细节：

### TreeBoN方法

在大语言模型推理过程中，使模型输出符合人类意图和伦理标准的对齐至关重要。目前已有一些方法，如 BoN 采样通过生成多个回答并选择最优来尝试提高推理性能，但这种方法存在明显缺陷。

**BoN 采样的问题**：其计算成本高，因为它需要生成大量回答，计算成本随生成回答数量的增加呈线性增长，这在实际应用中效率低下，尤其是对于大规模的语言模型和复杂任务，会带来巨大的计算开销和延迟，限制了其在实时或对回答速度有要求场景中的应用。

**加速方法的局限性**：像 Speculative BoN 等加速方法，试图通过对部分回答（如前 K 个 Token）进行评分来预测整体回答的质量，但由于奖励模型通常是在完整回答上训练的，对部分回答的评分不准确，导致预测结果与实际评分有较大偏差，无法有效提升性能，反而可能影响最终的推理效果。

为了克服 BoN 采样效率低下的问题，TreeBoN 采用层次化策略，将长序列生成过程拆分为多个子序列。通过在树结构中逐层生成候选回答片段，避免了一次性生成大量完整回答带来的高额计算成本。TreeBoN通过在树结构中逐层生成候选回答片段来工作。算法从一组初始根回答片段开始，在每一层，选择高奖励的回答片段并扩展为多个子回答片段。这种对树空间的推测性搜索提高了效率和最终回答质量。


![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/10/26/1729876214352-89036c0f-00a1-4463-a0cf-3a3631dd696a.png)

具体来说，TreeBoN生成过程包含以下步骤：

- **初始候选生成**：使用基础策略$π_{base}$生成$N$个长度为$l_1$的候选回答片段$C_1$，总最大回答长度$l_{max}$被分割为多个长度为$l_i$的段。
- **部分奖励评分**：在每层$i$，使用奖励模型或部分奖励函数$r(y|x)$计算候选回答片段$y \in C_i$的奖励分数，在生成长度为$l_i$的回答片段后进行评分。
- **剪枝和选择**：根据奖励分数，从当前层中选择前$N / N_{children}$个候选回答片段形成活动集$P_i$，这些高奖励的父回答片段将用于在下一层生成回答片段。
- **回答片段扩展**：对于每个父回答片段$y \in P_i$，TreeBoN从基础策略$π_{base}$中采样$N_{children}$个回答片段，每个回答片段的最大新token长度为$l_{i + 1}$，从而生成下一层的候选集$C_{i + 1}$。整个过程中，候选集大小始终为$N$，活动集$P_i$大小始终为$N / N_{children}$，以确保在不增加计算预算的情况下生成相同数量的总token。
- **最终选择**：在生成所有层的候选回答片段后，使用奖励模型计算最后一层候选回答片段$C_{N_{layer}}$的最终奖励，选择奖励最高的回答$y*$作为最终输出。


在这个过程中，TreeBoN使用**加权隐式奖励函数**来评估部分回答片段。对于序列y的前K个token，部分奖励计算为

$$
r_{partial}(y_{:K}|x)=\sum_{k=0}^{K - 1}w_k log\frac{\pi^{*}(y_k|x,y_{:k})}{\pi(y_k|x,y_{:k})}
$$

其中$w_k=\frac{1}{|y_k|}$是加权因子，用于调整每个token级别的对数似然比的贡献。

这种加权奖励有助于早期剪枝低质量回答片段，并在整个树扩展过程中鼓励继续生成更高质量的候选回答片段。通过利用DPO策略模型的隐式奖励，TreeBoN能够更准确地评估部分回答片段，从而提高整体的回答质量。


### 实验结果

论文通过一系列实验评估了TreeBoN方法在不同数据集上的性能表现，包括与Baseline方法的对比、不同树结构和参数设置的影响、效率评估以及对不同隐式奖励的探索等方面。具体实验结果如下：

1.**在不同数据集上的改进**
   
**评估方法**：使用GPT4 win-rate评估方法，在AlpacaFarm、UltraFeedback、HH - RLHF和TutorEval等数据集上，针对100个随机选择的提示，对比TreeBoN与Baseline方法（Best-of-N采样，N = 128）的性能。对于数学推理数据集GSM8K，报告零样本pass@1解决率。
    


![](https://fastly.jsdelivr.net/gh/bucketio/img10@main/2024/10/26/1729876350955-4f754095-8a40-4f56-9bdf-e6736c81dcf4.png)

      
- 在AlpacaFarm、UltraFeedback、HH - RLHF和TutorEval数据集上，TreeBoN在最大长度为192和384 tokens时，始终优于Baseline方法。例如，在192 tokens时，TreeBoN在AlpacaFarm上达到64%的win-rate，在其他数据集上至少达到60%的win-rate；在384 tokens时，在AlpacaFarm上保持62%的win-rate，在其他数据集上至少54%，使用SFR模型时在所有数据集上达到60% - 65%的win-rate。
- 在GSM8K数据集上，TreeBoN在最大回答长度为576 tokens时，pass@1解决率比BoN高出9%，表明TreeBoN的分层结构有助于处理需要长CoT推理的数学推理任务。
    
2.**不同树结构的影响**
   
**实验设置**：在保持计算成本不变（$N = 128$和$l_{max}$相同）的情况下，分别改变树层数(Number of Layers)和每个节点的子节点数量(Number of Children)，在AlpacaFarm数据集上计算TreeBoN相对于BoN的win-rate。
    
    
    
![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2024/10/26/1729876397886-74292db3-e428-4c2e-b121-ec29af2611ae.png)


![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/10/26/1729876427865-eb65d3bc-3830-454b-aee5-9ceda3b8cef1.png)

  
      
- 增加树层数能持续提高性能，如在192和384 tokens的最大长度下，随着树层数增加，win-rate有所提升。
- 对于不同的最大生成长度，最佳子节点数量不同。但总体而言，无论树结构如何变化，TreeBoN相对于Baseline方法的win-rate保持在约60%左右，显示了方法的有效性和稳健性，同时表明未来可通过针对不同任务探索更多超参数来进一步提高性能。
    
3.**效率评估**
- 在实验设置中，计算成本仅由根样本数量$N$和最大生成长度$l_{max}$控制，这里Baseline方法BoN设置同时生成128个回答，并从中利用奖励模型选择最佳结果。


![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2024/10/26/1729876876219-519489a4-10a1-4b0d-8477-2653a4c70e19.png)


      
 - 随着计算预算增加（即$N$增加），TreeBoN相对于BoN的win-rate也增加，表明TreeBoN比Baseline方法更具可扩展性，能更有效地利用额外计算预算。例如，在AlpacaFarm数据集上，当$N$从8增加到128时，TreeBoN的win-rate逐渐提高，即使在$N = 8$（仅为BoN计算成本的$8/128=6.3\%$）时，TreeBoN仍能以$55\%$的win-rate优于BoN。
    
4.**不同隐式奖励的探索**

 **实验设置**：测试不同的隐式奖励，包括DPO隐式奖励、加权隐式奖励、加权隐式奖励（指数衰减）、长度归一化DPO隐式奖励、DPO策略对数概率和、SimPO奖励等，在AlpacaFarm数据集上使用默认配置的TreeBoN进行实验。
 
 
![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/10/26/1729876957829-94847f91-dea7-4317-81c7-f0c4370a2880.png)


**加权隐式奖励**在树搜索设置中表现最佳，达到最高的GPT4 win-rate，证明了该奖励设计在TreeBoN方法中的有效性。




5.**与其他方法对比讨论**

**与SBoN对比**：SBoN依赖于**部分奖励分数与回答奖励正相关的假设**，但由于奖励模型通常在完整回答上训练，对部分回答的评分不准确，导致性能欠佳。TreeBoN通过使用更精确的**DPO策略模型的隐式奖励信号**解决了这一问题，显著提高了部分奖励近似的可靠性。此外，TreeBoN的分层树结构能更全面地探索回答空间，在扩展有希望的候选回答片段的同时有效地剪枝低质量回答片段，是SBoN的一种广义形式（当Nchildren = 1且Nlayer = 2时，TreeBoN可简化为SBoN的两层结构）。

**与传统BoN对比**：传统BoN在生成候选回答片段时没有分层结构，只是简单地探索回答空间。TreeBoN采用**更结构化的探索策略**，通过逐层生成和优化回答，使用更少的总样本更有效地搜索回答空间，从而在速度和性能上都有所改进，更好地平衡了探索与利用之间的权衡。并且，TreeBoN可以利用键值缓存机制进一步加速，在树结构中，父token的键和值可被其子节点重用，提高了计算效率。

---


- 原文链接: https://arxiv.org/abs/2410.16033
- 更多文章请详见 Github 仓库: https://github.com/ZJU-LLMs/Foundations-of-LLMs




    
    

