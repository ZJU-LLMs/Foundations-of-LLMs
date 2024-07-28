# 参数高效微调

- [参数高效微调](#参数高效微调)
  - [参数高效微调简介](#参数高效微调简介)
  - [参数附加方法](#参数附加方法)
  - [参数选择方法](#参数选择方法)
  - [低秩适配方法](#低秩适配方法)
  - [实践与应用](#实践与应用)


## <img src="../figure/star.svg" width="25" height="25" />参数高效微调简介

1. **Efficient Large Language Models: A Survey.** `arXiv`  
   *Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, Mi Zhang.* [[PDF](https://arxiv.org/abs/2312.03863)] [[Code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)], 2023.

2. **A Survey for In-context Learning.** `arXiv`  
   *Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, Zhifang Sui.* [[PDF](https://arxiv.org/abs/2301.00234)] [[Code](https://github.com/dqxiu/ICL_PaperList)], 2023.

3. **Instruction Tuning for Large Language Models: A Survey.** `arXiv`  
   *Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, Guoyin Wang.* [[PDF](https://arxiv.org/abs/2308.10792)] [[Code](https://github.com/dqxiu/ICL_PaperList)], 2023.

4. **Finetuned language models are zero-shot learners.** `arXiv`  
   *Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le.* [[PDF](https://arxiv.org/abs/2109.01652)] [[Code](https://github.com/google-research/flan)], 2021.

5. **Multitask Prompted Training Enables Zero-Shot Task Generalization.** `ICLR`  
   *Victor Sanh et al.* [[PDF](https://arxiv.org/abs/2110.08207)] [[Code](https://github.com/bigscience-workshop/promptsource.git)], 2022.

6. **Instruction in the Wild: A User-based Instruction Dataset.** `GitHub`  
   *Jinjie Ni and Fuzhao Xue and Kabir Jain and Mahir Hitesh Shah and Zangwei Zheng and Yang You.* [[Code](https://github.com/XueFuzhao/InstructionWild)], 2023.

7. **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** `ACL`  
   *Yizhong Wang et al.* [[PDF](https://arxiv.org/abs/2212.10560)] [[Code](https://github.com/yizhongw/self-instruct.git)], 2023.

8. **Llama 2: Open foundation and fine-tuned chat models.** `arXiv`  
   *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.* [[PDF](https://arxiv.org/abs/2307.09288)] [[Code](https://github.com/meta-llama/llama.git)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />参数附加方法

1. **The Power of Scale for Parameter-Efficient Prompt Tuning.** `EMNLP`  
   *Brian Lester, Rami Al-Rfou, and Noah Constant* [[PDF](https://arxiv.org/abs/2104.08691)] [[Code](https://github.com/mkshing/Prompt-Tuning.git)], 2021.

2. **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** `ACL`  
   *Xiang Lisa Li and Percy Liang* [[PDF](https://arxiv.org/abs/2101.00190)] [[Code](https://github.com/XiangLi1999/PrefixTuning.git)], 2021.

3. **Parameter-Efficient Transfer Learning for NLP.** `ICML`  
   *Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.* [[PDF](https://arxiv.org/abs/1902.00751)] [[Code](https://github.com/google-research/adapter-bert.git)], 2019.

4. **AdapterFusion: Non-Destructive Task Composition for Transfer Learning** ``
   *Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych.* [[PDF](https://arxiv.org/abs/2005.00247)] [[Code]()], 2020.

5. **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters.** `Findings of EMNLP`  
   *Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao.* [[PDF](https://arxiv.org/abs/2210.04284)] [[Code](https://github.com/Shwai-He/SparseAdapter.git)], 2022.

6. **Counter-Interference Adapter for Multilingual Machine Translation.** `Findings of EMNLP`  
   *Yaoming Zhu, Jiangtao Feng, Chengqi Zhao, Mingxuan Wang, Lei Li.* [[PDF](https://arxiv.org/abs/2104.08154)] [[Code](https://github.com/Yaoming95/CIAT.git)], 2021.

7. **Tuning Language Models by Proxy.** `arXiv`  
   *Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, Noah A. Smith.* [[PDF](https://arxiv.org/abs/2401.08565)] [[Code](https://github.com/alisawuffles/proxy-tuning)], 2024.

8. **Training Neural Networks with Fixed Sparse Masks.** `NIPS`  
    *Yi-Lin Sung, Varun Nair, Colin Raffel* [[PDF](https://arxiv.org/abs/2111.09839)] [[Code](https://github.com/VITA-Group/ToST)], 2021.

## <img src="../figure/star.svg" width="25" height="25" />参数选择方法

1. **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.** `ACL`  
   *Elad Ben Zaken, Shauli Ravfogel, Yoav Goldberg* [[PDF](https://arxiv.org/abs/2106.10199)] [[Code](https://github.com/benzakenelad/BitFit.git)], 2022.

2. **What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning.** `arXiv`  
   *Jaejun Lee, Raphael Tang, and Jimmy Lin* [[PDF](https://arxiv.org/abs/1911.03090)], 2019.

3. **On the Effectiveness of Parameter-Efficient Fine-Tuning.** `AAAI`  
   *Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, Nigel Collier.* [[PDF](https://arxiv.org/abs/2211.15583)] [[Code](https://github.com/fuzihaofzh/AnalyzeParameterEfficientFinetune.git)], 2023.

4. **Parameter-Efficient Fine-Tuning without Introducing New Latency.** `ACL`  
   *Baohao Liao, Yan Meng, and Christof Monz* [[PDF](https://arxiv.org/abs/2305.16742)], 2023.

5. **Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning.** `EMNLP`  
   *Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, Fei Huang.* [[PDF](https://arxiv.org/abs/2109.05687)] [[Code](https://github.com/pkunlp-icler/ChildTuning.git)], 2021.

6. **Masking as an Efficient Alternative to Finetuning for Pre-trained Language Models.** `EMNLP`  
   *Mengjie Zhao, Tao Lin, Fei Mi, Martin Jaggi, Hinrich Schütze.* [[PDF](https://arxiv.org/abs/2004.12406)], 2020.

7. **Composable Sparse Fine-Tuning for Cross-Lingual Transfer.** `ACL`  
   *Alan Ansell, Edoardo Maria Ponti, Anna Korhonen, Ivan Vulić.* [[PDF](https://arxiv.org/abs/2110.07560)] [[Code](https://github.com/cambridgeltl/composable-sft.git)], 2022.

8. **GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.** `ICLR`  
   *Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman.* [[PDF](https://arxiv.org/abs/1804.07461)] [[Code](https://github.com/nyu-mll/GLUE-baselines.git)], 2019.

9. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** `ICLR`  
    *Jonathan Frankle and Michael Carbin* [[PDF](https://arxiv.org/abs/1803.03635)], 2019.

10. **Unified Low-Resource Sequence Labeling by Sample-Aware Dynamic Sparse Finetuning** `EMNLP`
    *Sarkar Snigdha Sarathi Das, Ranran Haoran Zhang, Peng Shi, Wenpeng Yin, Rui Zhang.* [[PDF](https://arxiv.org/abs/2311.03748)] [[Code](https://github.com/psunlpgroup/FISH-DIP.git)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />低秩适配方法

1. **LoRA: Low-Rank Adaptation of Large Language Models.** `ICLR`  
   *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* [[PDF](https://arxiv.org/abs/2106.09685)] [[Code](https://github.com/microsoft/LoRA)], 2022

2. **Towards a Unified View of Parameter-Efficient Transfer Learning.** `ICLR`  
   *Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig.* [[PDF](https://arxiv.org/abs/2110.04366)] [[Code](https://github.com/jxhe/unify-parameter-efficient-tuning.git)], 2022.

3. **A Note on LoRA.** `arXiv`  
   *Vlad Fomenko, Han Yu, Jongho Lee, Stanley Hsieh, Weizhu Chen.* [[PDF](https://arxiv.org/abs/2404.05086)], 2024.

4. **KronA: Parameter Efficient Tuning with Kronecker Adapter** `arXiv`
   *Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J. Clark, Mehdi Rezagholizadeh.* [[PDF](https://arxiv.org/abs/2212.10650)], 2022.

5. **Parameter-Efficient Model Adaptation for Vision Transformers.** `AAAI`  
   *Xuehai He,Chunyuan Li,Pengchuan Zhang,Jianwei Yang,Xin Eric Wang.* [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25160/24932)], 2023.

6. **DoRA: Weight-Decomposed Low-Rank Adaptation.** `arXiv`  
   *Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen.* [[PDF](https://arxiv.org/abs/2402.09353)] [[Code](https://github.com/nbasyl/DoRA.git)], 2024.

7. **LoRA Learns Less and Forgets Less** `arXiv`
   *Dan Biderman, Jose Gonzalez Ortiz, Jacob Portes, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham.* [[PDF](https://arxiv.org/abs/2405.09673)], 2024.

8. **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection** `arXiv`
   *Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian.* [[PDF](https://arxiv.org/abs/2403.03507)], 2024.

9. **S-LoRA: Serving Thousands of Concurrent LoRA Adapters.** `arXiv`  
   *Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica.* [[PDF](https://arxiv.org/abs/2311.03285)] [[Code](https://github.com/S-LoRA/S-LoRA.git)], 2023.

10. **Sparse Low-rank Adaptation of Pre-trained Language Models.** `EMNLP`  
    *Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun.* [[PDF](https://arxiv.org/abs/2311.11696)] [[Code](https://github.com/TsinghuaC3I/SoRA)], 2023.

11. **DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution.** `arXiv`  
   *Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, Jinan Xu* [[PDF](https://arxiv.org/abs/2405.17357)] [[Code](https://github.com/MIkumikumi0116/DoRA)], 2024.

12. **ReLoRA: High-Rank Training Through Low-Rank Updates.** `NIPS Workshop`  
   *Vladislav Lialin, Namrata Shivagunde, Sherin Muckatira, Anna Rumshisky.* [[PDF](https://arxiv.org/abs/2307.05695)] [[Code](https://github.com/Guitaricet/relora)],2023.

13. **SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining.** `arXiv`  
    *Andi Han, Jiaxiang Li, Wei Huang, Mingyi Hong, Akiko Takeda, Pratik Jawanpuria, Bamdev Mishra.* [[PDF](https://arxiv.org/abs/2406.02214)] [[Code](https://github.com/andyjm3/SLTrain)], 2024.

14. **Pissa: Principal singular values and singular vectors adaptation of large language models.** `arXiv`  
    *Fanxu Meng, Zhaohui Wang, Muhan Zhang* [[PDF](https://arxiv.org/abs/2404.02948)] [[Code](https://github.com/GraphPKU/PiSSA)], 2024.

15. **MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning.** `arXiv`  
    *Hanqing Wang, Zeguan Xiao, Yixia Li, Shuo Wang, Guanhua Chen, Yun Chen.* [[PDF](https://arxiv.org/abs/2406.09044)], 2024.

16. **A Survey on LoRA of Large Language Models.** `arXiv`  
    *Yuren Mao, Yuhang Ge, Yijiang Fan, Wenyi Xu, Yu Mi, Zhonghao Hu, Yunjun Gao.* [[PDF](https://arxiv.org/abs/2407.11046)] [[Code](https://github.com/ZJU-LLMs/Awesome-LoRAs.git)], 2024.

17. **Parameter-efficient fine-tuning of large-scale pre-trained language models.** `Nat. Mac. Intell.`  
    *Ding, Ning, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu.* [[PDF](https://www.nature.com/articles/s42256-023-00626-4.pdf)], 2023.

18. **LoTR: Low Tensor Rank Weight Adaptation.** `arXiv`  
    *Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, Ivan Oseledets.* [[PDF](https://arxiv.org/abs/2402.01376)], 2024.

19. **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Ting Jiang, Shaohan Huang, Shengyue Luo, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, Fuzhen Zhuang.* [[PDF](https://arxiv.org/abs/2405.12130)] [[Code](https://github.com/kongds/MoRA)], 2024.

20. **Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning.** `arXiv`  
    *Wenhan Xia, Chengwei Qin, Elad Hazan.* [[PDF](https://arxiv.org/abs/2401.04151)], 2024.

21. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** `ACL/IJCNLP`  
    *Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta.* [[PDF](https://arxiv.org/abs/2012.13255)],2021.

22. **Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning.** `arXiv`  
    *Pengjie Ren, Chengshun Shi, Shiguang Wu, Mengqi Zhang, Zhaochun Ren, Maarten de Rijke, Zhumin Chen, Jiahuan Pei* [[PDF](https://arxiv.org/abs/2402.17263)], 2024.

23. **LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning.** `arXiv`  
    *Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, Tong Zhang.* [[PDF](https://arxiv.org/abs/2403.17919)] [[Code](https://github.com/OptimalScale/LMFlow)], 2024.

24. **Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning.** `arXiv`  
    *Wenhan Xia, Chengwei Qin, and Elad Hazan* [[PDF](https://arxiv.org/abs/2401.04151)], 2024.

25. **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** `ICLR`  
    *Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao.* [[PDF](https://arxiv.org/abs/2303.10512)] [[Code](https://github.com/QingruZhang/AdaLoRA.git)], 2023.

26. **LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.** `CoLM`  
    *Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin.* [[PDF](https://arxiv.org/abs/2307.13269)] [[Code](https://github.com/sail-sg/lorahub.git)], 2023.

27. **Dylora: Parameter efficient tuning of pre-trained models using dynamic search-free low-rank adaptation** `EACL`
    *Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, Ali Ghodsi.* [[PDF](https://arxiv.org/abs/2210.07558)] [[Code](github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA)], 2023.

28. **DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dy-namic Rank Distribution** `arXiv` 
    *Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, Jinan Xu.* [[PDF](https://arxiv.org/abs/2405.17357)] [[Code](https://github.com/MIkumikumi0116/DoRA.git)],2023.

## <img src="../figure/star.svg" width="25" height="25" />实践与应用

1. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis.** `SIGMOD`  
   *Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin.* [[PDF](https://arxiv.org/abs/2401.10506)], 2024.

2. **TabLLM: Few-shot Classification of Tabular Data with Large Language Models.** `AISTATS`  
   *Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, David Sontag.* [[PDF](https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf)], 2023.

