# 参数高效微调

- [参数高效微调](#参数高效微调)
  - [参数高效微调简介](#参数高效微调简介)
  - [参数附加方法](#参数附加方法)
  - [参数选择方法](#参数选择方法)
  - [低秩适配方法](#低秩适配方法)
  - [实践与应用](#实践与应用)


## <img src="../figure/star.svg" width="25" height="25" />参数高效微调简介

1. **Efficient Large Language Models: A Survey.** `arXiv`  
   *Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, Mi Zhang.* [[PDF](https://arxiv.org/abs/2312.03863)], 2023.

2. **A Survey for In-context Learning.** `CoRR`  
   *Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, Zhifang Sui.* [[PDF](https://arxiv.org/abs/2301.00234)], 2023.

3. **Instruction Tuning for Large Language Models: A Survey.** `arXiv`  
   *Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, Guoyin Wang.* [[PDF](https://arxiv.org/abs/2308.10792)], 2023.

4. **Finetuned language models are zero-shot learners.** `arXiv`  
   *Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le.* [[PDF](https://arxiv.org/abs/2109.01652)], 2021.

5. **Multitask Prompted Training Enables Zero-Shot Task Generalization.** `ICLR`  
   *Victor Sanh et al.* [[PDF](https://arxiv.org/abs/2110.08207)], 2022.

6. **Instruction in the Wild: A User-based Instruction Dataset.** `GitHub`  
   *Jinjie Ni and Fuzhao Xue and Kabir Jain and Mahir Hitesh Shah and Zangwei Zheng and Yang You.* [[GitHub](https://github.com/XueFuzhao/InstructionWild)], 2023.

7. **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** `ACL`  
   *Yizhong Wang et al.* [[PDF](https://arxiv.org/abs/2212.10560)], 2023.

8. **Llama 2: Open foundation and fine-tuned chat models.** `arXiv`  
   *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.* [[PDF](https://arxiv.org/abs/2307.09288)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />参数附加方法

1. **The Power of Scale for Parameter-Efficient Prompt Tuning.** `EMNLP`  
   *Brian Lester, Rami Al-Rfou, and Noah Constant* [[PDF](https://arxiv.org/abs/2104.08691)], 2021.

2. **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** `ACL`  
   *Xiang Lisa Li and Percy Liang* [[PDF](https://arxiv.org/abs/2101.00190)], 2021.

3. **Parameter-Efficient Transfer Learning for NLP.** `ICML`  
   *Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.* [[PDF](https://arxiv.org/abs/1902.00751)], 2019.

4. **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters.** `Findings of EMNLP`  
   *Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao.* [[PDF](https://arxiv.org/abs/2210.04284)], 2022.

5. **Counter-Interference Adapter for Multilingual Machine Translation.** `Findings of EMNLP`  
   *Yaoming Zhu, Jiangtao Feng, Chengqi Zhao, Mingxuan Wang, Lei Li.* [[PDF](https://arxiv.org/abs/2104.08154)], 2021.

6. **Tuning Language Models by Proxy.** `arXiv`  
   *Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, Noah A. Smith.* [[PDF](https://arxiv.org/abs/2401.08565)], 2024.

7. **Training Neural Networks with Fixed Sparse Masks.** `NIPS`  
    *Yi-Lin Sung, Varun Nair, Colin Raffel* [[PDF](https://arxiv.org/abs/2111.09839)], 2021.

## <img src="../figure/star.svg" width="25" height="25" />参数选择方法

1. **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.** `ACL`  
   *Elad Ben Zaken, Shauli Ravfogel, Yoav Goldberg* [[PDF](https://arxiv.org/abs/2106.10199)], 2022.

2. **What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning.** `arXiv`  
   *Jaejun Lee, Raphael Tang, and Jimmy Lin* [[PDF](https://arxiv.org/abs/1911.03090)], 2019.

3. **On the Effectiveness of Parameter-Efficient Fine-Tuning.** `AAAI`  
   *Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, Nigel Collier.* [[PDF](https://arxiv.org/abs/2211.15583)], 2023.

4. **Parameter-Efficient Fine-Tuning without Introducing New Latency.** `ACL`  
   *Baohao Liao, Yan Meng, and Christof Monz* [[PDF](https://arxiv.org/abs/2305.16742)], 2023.

5. **Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning.** `EMNLP`  
   *Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, Fei Huang.* [[PDF](https://arxiv.org/abs/2109.05687)], 2021.

6. **Masking as an Efficient Alternative to Finetuning for Pre-trained Language Models.** `EMNLP`  
   *Mengjie Zhao, Tao Lin, Fei Mi, Martin Jaggi, Hinrich Schütze.* [[PDF](https://arxiv.org/abs/2004.12406)], 2020.

7. **Composable Sparse Fine-Tuning for Cross-Lingual Transfer.** `ACL`  
   *Alan Ansell, Edoardo Maria Ponti, Anna Korhonen, Ivan Vulić.* [[PDF](https://arxiv.org/abs/2110.07560)], 2022.

8. **GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.** `ICLR`  
   *Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman.* [[PDF](https://arxiv.org/abs/1804.07461)], 2019.

9. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** `ICLR`  
    *Jonathan Frankle and Michael Carbin* [[PDF](https://arxiv.org/abs/1803.03635)], 2019.

## <img src="../figure/star.svg" width="25" height="25" />低秩适配方法

1. **LoRA: Low-Rank Adaptation of Large Language Models.** `ICLR`  
   *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* [[PDF](https://arxiv.org/abs/2106.09685)] [[Code](https://github.com/microsoft/LoRA)], 2022

2. **Towards a Unified View of Parameter-Efficient Transfer Learning.** `ICLR`  
   *Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig.* [[PDF](https://arxiv.org/abs/2110.04366)], 2022.

3. **A Note on LoRA.** `arXiv`  
   *Vlad Fomenko, Han Yu, Jongho Lee, Stanley Hsieh, Weizhu Chen.* [[PDF](https://arxiv.org/abs/2404.05086)], 2024.

4. **Parameter-Efficient Model Adaptation for Vision Transformers.** `AAAI`  
   *Xuehai He,Chunyuan Li,Pengchuan Zhang,Jianwei Yang,Xin Eric Wang.* [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25160/24932)], 2023.

5. **DoRA: Weight-Decomposed Low-Rank Adaptation.** `arXiv`  
   *Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen.* [[PDF](https://arxiv.org/abs/2402.09353)], 2024.

6. **S-LoRA: Serving Thousands of Concurrent LoRA Adapters.** `arXiv`  
   *Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica.* [[PDF](https://arxiv.org/abs/2311.03285)], 2023.

7. **Sparse Low-rank Adaptation of Pre-trained Language Models.** `EMNLP`  
   *Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun.* [[PDF](https://arxiv.org/abs/2210.07265)], 2023.

8. **DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution.** `arXiv`  
   *Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, Jinan Xu* [[PDF](https://arxiv.org/abs/2405.17357)], 2024.

9. **ReLoRA: High-Rank Training Through Low-Rank Updates.** `NIPS Workshop`  
   *Vladislav Lialin, Namrata Shivagunde, Sherin Muckatira, Anna Rumshisky.* [[PDF](https://arxiv.org/abs/2307.05695)],2023.

10. **SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining.** `arXiv`  
    *Andi Han, Jiaxiang Li, Wei Huang, Mingyi Hong, Akiko Takeda, Pratik Jawanpuria, Bamdev Mishra.* [[PDF](https://arxiv.org/abs/2406.02214)], 2024.

11. **Pissa: Principal singular values and singular vectors adaptation of large language models.** `arXiv`  
    *Fanxu Meng, Zhaohui Wang, Muhan Zhang* [[PDF](https://arxiv.org/abs/2404.02948)], 2024.

12. **MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning.** `arXiv`  
    *Hanqing Wang, Zeguan Xiao, Yixia Li, Shuo Wang, Guanhua Chen, Yun Chen.* [[PDF](https://arxiv.org/abs/2406.09044)], 2024.

13. **A Survey on LoRA of Large Language Models.** `arXiv`  
    *Yuren Mao, Yuhang Ge, Yijiang Fan, Wenyi Xu, Yu Mi, Zhonghao Hu, Yunjun Gao.* [[PDF](https://arxiv.org/abs/2407.11046)], 2024.

14. **Parameter-efficient fine-tuning of large-scale pre-trained language models.** `Nat. Mac. Intell.`  
    *Ding, Ning, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu.* [[PDF](https://www.nature.com/articles/s42256-023-00626-4.pdf)], 2023.

15. **LoTR: Low Tensor Rank Weight Adaptation.** `arXiv`  
    *Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, Ivan Oseledets.* [[PDF](https://arxiv.org/abs/2402.01376)], 2024.

16. **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Ting Jiang, Shaohan Huang, Shengyue Luo, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, Fuzhen Zhuang.* [[PDF](https://arxiv.org/abs/2405.12130)], 2024.

17. **Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning.** `arXiv`  
    *Wenhan Xia, Chengwei Qin, Elad Hazan.* [[PDF](https://arxiv.org/abs/2401.04151)], 2024.

18. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** `ACL/IJCNLP`  
    *Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta.* [[PDF](https://arxiv.org/abs/2012.13255)],2021.

19. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** `ACL/IJCNLP`  
    *Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer.* [[PDF](https://arxiv.org/abs/2012.13255)], 2021.

20. **Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Pengjie Ren, Chengshun Shi, Shiguang Wu, Mengqi Zhang, Zhaochun Ren, Maarten de Rijke, Zhumin Chen, Jiahuan Pei* [[PDF](https://arxiv.org/abs/2402.17263)], 2024.

21. **LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning.** `arXiv`  
    *Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, Tong Zhang.* [[PDF](https://arxiv.org/abs/2403.17919)], 2024.

22. **Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning.** `arXiv`  
    *Wenhan Xia, Chengwei Qin, and Elad Hazan* [[PDF](https://arxiv.org/abs/2401.04151)], 2024.

23. **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** `ICLR`  
    *Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao.* [[PDF](https://arxiv.org/abs/2303.10512)], 2023.

24. **LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.** `arXiv`  
    *Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin.* [[PDF](https://arxiv.org/abs/2307.13269)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />实践与应用

1. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis.** `SIGMOD`  
   *Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin.* [[PDF](https://arxiv.org/abs/2401.10506)], 2024.

2. **TabLLM: Few-shot Classification of Tabular Data with Large Language Models.** `AISTATS`  
   *Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, David Sontag.* [[PDF](https://arxiv.org/abs/2401.10506)], 2023.

