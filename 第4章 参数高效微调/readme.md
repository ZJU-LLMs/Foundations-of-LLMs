# 参数高效微调

- [参数高效微调](#参数高效微调)
  - [参数高效微调简介](#参数高效微调简介)
  - [参数附加方法](#参数附加方法)
  - [参数选择方法](#参数选择方法)
  - [低秩适配方法](#低秩适配方法)
  - [实践与应用](#实践与应用)


## <img src="../figure/star.svg" width="25" height="25" />参数高效微调简介

1. **Efficient Large Language Models: A Survey.** `arXiv`  
   *Zhongwei Wan et al.* [[PDF](https://arxiv.org/abs/2312.03863)], 2023.

2. **A Survey for In-context Learning.** `CoRR`  
   *Qingxiu Dong et al.* [[PDF](https://arxiv.org/abs/2301.00234)], 2023.

3. **Instruction Tuning for Large Language Models: A Survey.** `arXiv`  
   *Shengyu Zhang et al.* [[PDF](https://arxiv.org/abs/2308.10792)], 2023.

4. **Finetuned language models are zero-shot learners.** `arXiv`  
   *Jason Wei et al.* [[PDF](https://arxiv.org/abs/2109.01652)], 2021.

5. **Multitask Prompted Training Enables Zero-Shot Task Generalization.** `2021`  
   *Victor Sanh et al.*.

6. **Instruction in the Wild: A User-based Instruction Dataset.** `GitHub`  
   *Jinjie Ni et al.* [[GitHub](https://github.com/XueFuzhao/InstructionWild)], 2023.

7. **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** `ACL`  
   *Yizhong Wang et al.* [[PDF](https://arxiv.org/abs/2303.08284)], 2023.

8. **Llama 2: Open foundation and fine-tuned chat models.** `arXiv`  
   *Hugo Touvron et al.* [[PDF](https://arxiv.org/abs/2307.09288)], 2023.

9. **Parameter-Efficient Transfer Learning for NLP.** `ICML`  
   *Neil Houlsby et al.* [[PDF](https://arxiv.org/abs/1902.00751)], 2019.

10. **The Power of Scale for Parameter-Efficient Prompt Tuning.** `EMNLP`  
    *Brian Lester, Rami Al-Rfou, and Noah Constant* [[PDF](https://arxiv.org/abs/2104.08691)], 2021.

11. **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** `ACL`  
    *Xiang Lisa Li and Percy Liang* [[PDF](https://arxiv.org/abs/2101.00190)], 2021.

12. **Tuning Language Models by Proxy.** `arXiv`  
    *Alisa Liu et al.* [[PDF](https://arxiv.org/abs/2401.08565)], 2024.

13. **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.** `ACL`  
    *Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel* [[PDF](https://arxiv.org/abs/2106.10199)], 2022.

14. **Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning.** `EMNLP`  
    *Runxin Xu et al.* [[PDF](https://arxiv.org/abs/2109.07785)], 2021.

15. **Training Neural Networks with Fixed Sparse Masks.** `NIPS`  
    *Yi-Lin Sung, Varun Nair, and Colin Raffel* [[PDF](https://arxiv.org/abs/2101.10358)], 2021.

16. **LoRA: Low-Rank Adaptation of Large Language Models.** `ICLR`  
   *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* [[PDF](https://arxiv.org/abs/2106.09685)] [[Code](https://github.com/microsoft/LoRA)], 2022

17. **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** `ICLR`  
    *Qingru Zhang et al.* [[PDF](https://arxiv.org/abs/2207.04283)], 2023.

18. **DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation.** `EACL`  
    *Mojtaba Valipour et al.* [[PDF](https://arxiv.org/abs/2304.01011)], 2023.

19. **DoRA: Weight-Decomposed Low-Rank Adaptation.** `arXiv`  
    *Shih-Yang Liu et al.* [[PDF](https://arxiv.org/abs/2402.09353)], 2024.

## <img src="../figure/star.svg" width="25" height="25" />参数附加方法

1. **The Power of Scale for Parameter-Efficient Prompt Tuning.** `EMNLP`  
   *Brian Lester, Rami Al-Rfou, and Noah Constant* [[PDF](https://arxiv.org/abs/2104.08691)], 2021.

2. **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** `ACL`  
   *Xiang Lisa Li and Percy Liang* [[PDF](https://arxiv.org/abs/2101.00190)], 2021.

3. **Parameter-Efficient Transfer Learning for NLP.** `ICML`  
   *Neil Houlsby et al.* [[PDF](https://arxiv.org/abs/1902.00751)], 2019.

4. **ReLoRA: High-Rank Training Through Low-Rank Updates.** `NIPS Workshop`  
   *Vladislav Lialin et al.* 2023.

5. **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters.** `Findings of EMNLP`  
   *Shwai He et al.* [[PDF](https://arxiv.org/abs/2107.00054)], 2022.

6. **Counter-Interference Adapter for Multilingual Machine Translation.** `Findings of EMNLP`  
   *Yaoming Zhu et al.* [[PDF](https://arxiv.org/abs/2207.04364)], 2021.

7. **Tuning Language Models by Proxy.** `arXiv`  
   *Alisa Liu et al.* [[PDF](https://arxiv.org/abs/2401.08565)], 2024.

8. **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters.** `Findings of EMNLP`  
   *Shwai He et al.* [[PDF](https://arxiv.org/abs/2107.00054)], 2022.

## <img src="../figure/star.svg" width="25" height="25" />参数选择方法

1. **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.** `ACL`  
   *Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel* [[PDF](https://arxiv.org/abs/2106.10199)], 2022.

2. **What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning.** `arXiv`  
   *Jaejun Lee, Raphael Tang, and Jimmy Lin* [[PDF](https://arxiv.org/abs/1911.03090)], 2019.

3. **On the Effectiveness of Parameter-Efficient Fine-Tuning.** `AAAI`  
   *Zihao Fu et al.* [[PDF](https://arxiv.org/abs/2212.02929)], 2023.

4. **Parameter-Efficient Fine-Tuning without Introducing New Latency.** `ACL`  
   *Baohao Liao, Yan Meng, and Christof Monz* [[PDF](https://arxiv.org/abs/2209.04510)], 2023.

5. **Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning.** `EMNLP`  
   *Runxin Xu et al.* [[PDF](https://arxiv.org/abs/2109.07785)], 2021.

6. **Masking as an Efficient Alternative to Finetuning for Pre-trained Language Models.** `EMNLP`  
   *Mengjie Zhao et al.* [[PDF](https://arxiv.org/abs/2110.10392)], 2020.

7. **Composable Sparse Fine-Tuning for Cross-Lingual Transfer.** `ACL`  
   *Alan Ansell et al.* [[PDF](https://arxiv.org/abs/2109.04336)], 2022.

8. **GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.** `ICLR`  
   *Alex Wang et al.* [[PDF](https://arxiv.org/abs/1804.07461)], 2019.

9. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** `ICLR`  
      *Jonathan Frankle and Michael Carbin* [[PDF](https://arxiv.org/abs/1803.03635)], 2019.

10. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** `ICLR`  
    *Jonathan Frankle and Michael Carbin* [[PDF](https://arxiv.org/abs/1803.03635)], 2019.

## <img src="../figure/star.svg" width="25" height="25" />低秩适配方法

1. **LoRA: Low-Rank Adaptation of Large Language Models.** `ICLR`  
   *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* [[PDF](https://arxiv.org/abs/2106.09685)] [[Code](https://github.com/microsoft/LoRA)], 2022

2. **Towards a Unified View of Parameter-Efficient Transfer Learning.** `ICLR`  
   *Junxian He et al.* [[PDF](https://arxiv.org/abs/2106.04643)], 2022.

3. **A Note on LoRA.** `arXiv`  
   *Vlad Fomenko et al.* [[PDF](https://arxiv.org/abs/2404.05086)], 2024.

4. **Parameter-Efficient Model Adaptation for Vision Transformers.** `AAAI`  
   *Xuehai He et al.* [[PDF](https://arxiv.org/abs/2210.05525)], 2023.

5. **DoRA: Weight-Decomposed Low-Rank Adaptation.** `arXiv`  
   *Shih-Yang Liu et al.* [[PDF](https://arxiv.org/abs/2402.09353)], 2024.

6. **S-LoRA: Serving Thousands of Concurrent LoRA Adapters.** `arXiv`  
   *Ying Sheng et al.* [[PDF](https://arxiv.org/abs/2311.03285)], 2023.

7. **Sparse Low-rank Adaptation of Pre-trained Language Models.** `EMNLP`  
   *Ning Ding et al.* [[PDF](https://arxiv.org/abs/2210.07265)], 2023.

8. **DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution.** `arXiv`  
   *Yulong Mao et al.* [[PDF](https://arxiv.org/abs/2405.17357)], 2024.

9. **ReLoRA: High-Rank Training Through Low-Rank Updates.** `NIPS Workshop`  
   *Vladislav Lialin et al.* 2023.

10. **SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining.** `arXiv`  
    *Andi Han et al.* [[PDF](https://arxiv.org/abs/2406.02214)], 2024.

11. **Pissa: Principal singular values and singular vectors adaptation of large language models.** `arXiv`  
    *Fanxu Meng, Zhaohui Wang, and Muhan Zhang* [[PDF](https://arxiv.org/abs/2404.02948)], 2024.

12. **MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning.** `arXiv`  
    *Hanqing Wang et al.* [[PDF](https://arxiv.org/abs/2406.09044)], 2024.

13. **A Survey on LoRA of Large Language Models.** `arXiv`  
    *Yuren Mao et al.* [[PDF](https://arxiv.org/abs/2407.11046)], 2024.

14. **Parameter-efficient fine-tuning of large-scale pre-trained language models.** `Nat. Mac. Intell.`  
    *Ning Ding et al.* 5.3 (2023): 220-235.

15. **LoTR: Low Tensor Rank Weight Adaptation.** `arXiv`  
    *Daniel Bershatsky et al.* [[PDF](https://arxiv.org/abs/2402.01376)], 2024.

16. **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Ting Jiang et al.* [[PDF](https://arxiv.org/abs/2405.12130)], 2024.

17. **Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning.** `arXiv`  
    *Wenhan Xia, Chengwei Qin, and Elad Hazan* [[PDF](https://arxiv.org/abs/2401.04151)], 2024.

18. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** `ACL/IJCNLP`  
    *Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer* 2021.

19. **Sparse Low-rank Adaptation of Pre-trained Language Models.** `EMNLP`  
    *Ning Ding et al.* 2023.

20. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** `ACL/IJCNLP`  
    *Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer* 2021.

21. **Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Pengjie Ren et al.* [[PDF](https://arxiv.org/abs/2402.17263)], 2024.

22. **Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Pengjie Ren et al.* [[PDF](https://arxiv.org/abs/2402.17263)], 2024.

23. **Sparse Low-rank Adaptation of Pre-trained Language Models.** `EMNLP`  
    *Ning Ding et al.* 2023.

24. **Sparse Low-rank Adaptation of Pre-trained Language Models.** `EMNLP`  
    *Ning Ding et al.* 2023.

25. **ReLoRA: High-Rank Training Through Low-Rank Updates.** `NIPS Workshop`  
    *Vladislav Lialin et al.* 2023.

26. **Parameter-efficient fine-tuning of large-scale pre-trained language models.** `Nat. Mac. Intell.`  
    *Ning Ding et al.* 5.3 (2023): 220-235.

27. **LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning.** `arXiv`  
    *Rui Pan et al.* [[PDF](https://arxiv.org/abs/2403.17919)], 2024.

28. **Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning.** `arXiv`  
    *Wenhan Xia, Chengwei Qin, and Elad Hazan* [[PDF](https://arxiv.org/abs/2401.04151)], 2024.

29. **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** `ICLR`  
    *Qingru Zhang et al.* [[PDF](https://arxiv.org/abs/2207.04283)], 2023.

30. **LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.** `arXiv`  
    *Chengsong Huang et al.* [[PDF](https://arxiv.org/abs/2307.13269)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />实践与应用

1. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis.** `SIGMOD`  
   *Chao Zhang et al.* 2024.

2. **TabLLM: Few-shot Classification of Tabular Data with Large Language Models.** `AISTATS`  
   *Stefan Hegselmann et al.* 2023.
