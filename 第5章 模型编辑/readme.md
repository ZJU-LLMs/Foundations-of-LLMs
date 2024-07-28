# 模型编辑

- [模型编辑](#模型编辑)
  - [模型编辑简介](#模型编辑简介)
  - [模型编辑经典方法](#模型编辑经典方法)
  - [附加参数法：T-Patcher](#附加参数法t-patcher)
  - [定位编辑法：ROME](#定位编辑法rome)
  - [模型编辑应用](#模型编辑应用)


## <img src="../figure/star.svg" width="25" height="25" />模型编辑简介

1. **Knowledge Editing for Large Language Models: A Survey.** `arXiv`  
   *Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, Jundong Li.* [[PDF](https://arxiv.org/abs/2310.16218)], 2023

2. **A Comprehensive Study of Knowledge Editing for Large Language Models.** `arXiv`  
   *Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen.* [[PDF](https://arxiv.org/abs/2401.01286)][[Code](https://github.com/zjunlp/EasyEdit)], 2024

3. **Editing Large Language Models: Problems, Methods, and Opportunities.** `EMNLP`  
   *Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang.* [[PDF](https://arxiv.org/abs/2305.13172)][[Code](https://github.com/zjunlp/EasyEdit)], 2023

4. **A Survey on Knowledge Editing of Neural Networks.** `arXiv`  
   *Vittorio Mazzia, Alessandro Pedrani, Andrea Caciolai, Kay Rottmann, Davide Bernardi.* [[PDF](https://arxiv.org/abs/2310.19704)], 2023

## <img src="../figure/star.svg" width="25" height="25" />模型编辑经典方法

1. **Memory-Based Model Editing at Scale.** `ICML`  
   *Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn.* [[PDF](https://proceedings.mlr.press/v162/mitchell22a.html?_hsenc=p2ANqtz-8PcBZg33YLCBAVdcZ55PYZXm2xs6OJ8qM1z5cu9NWDbYyx8ey70v--e65rovexQfK34-tjgKdTMqKyU1nNVowzXjY-bA&_hsmi=226067236&utm_source=pocket_mylist)][[Code](https://sites.google.com/view/serac-editing)], 2022

2. **Fixing Model Bugs with Natural Language Patches.** `EMNLP`  
   *Shikhar Murty, Christopher D. Manning, Scott M. Lundberg, Marco Túlio Ribeiro.* [[PDF](https://arxiv.org/abs/2211.03318)], 2022

3. **Calibrating Factual Knowledge in Pretrained Language Models.** `EMNLP`  
   *Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, Lei Li.* [[PDF](https://arxiv.org/abs/2210.03329)][[Code](https://github.com/dqxiu/CaliNet)], 2022

4. **Transformer-Patcher: One Mistake Worth One Neuron.** `ICLR`  
   *Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, Zhang Xiong.* [[PDF](https://arxiv.org/abs/2301.09785)][[Code](https://github.com/ZeroYuHuang/Transformer-Patcher)], 2023

5. **Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.** `NeurIPS`  
   *Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi.* [[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html)][[Code](https://github.com/thartvigsen/grace)], 2023

6. **Meta-learning in neural networks: A survey.** `IEEE transactions on pattern analysis and machine intelligence`  
   *Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey.* [[PDF](https://ieeexplore.ieee.org/abstract/document/9428530)], 2021

7. **Editable Neural Networks.** `ICLR`  
   *Anton Sinitsin, Vsevolod Plokhotnyuk, Dmitry V. Pyrkin, Sergei Popov, Artem Babenko.* [[PDF](https://arxiv.org/abs/2004.00345)][[Code](https://github.com/xtinkt/editable)], 2020

8. **Editing Factual Knowledge in Language Models.** `EMNLP`  
   *Nicola De Cao, Wilker Aziz, Ivan Titov.* [[PDF](https://arxiv.org/abs/2104.08164)][[Code](https://github.com/nicola-decao/KnowledgeEditor)], 2021

9. **Fast Model Editing at Scale.** `ICLR`  
   *Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning.* [[PDF](https://arxiv.org/abs/2110.11309)][[Code](https://sites.google.com/view/mend-editing)], 2022

10. **Transformer Feed-Forward Layers Are Key-Value Memories.** `EMNLP`  
    *Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy.* [[PDF](https://arxiv.org/abs/2012.14913)][[Code](https://github.com/mega002/ff-layers/)], 2021

11. **Knowledge Neurons in Pretrained Transformers.** `ACL`  
    *Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei.* [[PDF](https://arxiv.org/abs/2104.08696)][[Code](https://github.com/Hunter-DDM/knowledge-neurons)], 2022

12. **Locating and Editing Factual Associations in GPT.** `NeurIPS`  
    *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov.* [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)][[Code](https://github.com/kmeng01/rome)], 2022

13. **Mass-Editing Memory in a Transformer.** `ICLR`  
    *Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, David Bau.* [[PDF](https://arxiv.org/abs/2210.07229)][[Code](https://github.com/kmeng01/memit)], 2023

## <img src="../figure/star.svg" width="25" height="25" />附加参数法：T-Patcher

1. **Transformer-Patcher: One Mistake Worth One Neuron.** `ICLR`  
   *Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, Zhang Xiong.* [[PDF](https://arxiv.org/pdf/2301.09785)][[Code](https://github.com/ZeroYuHuang/Transformer-Patcher)], 2023

## <img src="../figure/star.svg" width="25" height="25" />定位编辑法：ROME

1. **Locating and Editing Factual Associations in GPT.** `NeurIPS`  
   *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov.* [[PDF](https://arxiv.org/pdf/2202.05262)][[Code](https://github.com/kmeng01/rome)], 2022

2. **Mass-Editing Memory in a Transformer.** `ICLR`  
   *Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, David Bau.* [[PDF](https://arxiv.org/pdf/2210.07229)][[Code](https://github.com/kmeng01/memit)], 2023

## <img src="../figure/star.svg" width="25" height="25" />模型编辑应用

1. **Scalable Extraction of Training Data from (Production) Language Models.** `arXiv`  
   *Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee.* [[PDF](https://arxiv.org/pdf/2311.17035)], 2023

2. **DEPN: Detecting and Editing Privacy Neurons in Pretrained Language Models.** `arXiv`  
   *Xinwei Wu, Junzhuo Li, Minghui Xu, Weilong Dong, Shuangzhi Wu, Chao Bian, Deyi Xiong.* [[PDF](https://arxiv.org/pdf/2310.20138)][[Code](
   https://github.com/flamewei123/DEPN)], 2023

3. **Transformer Feed-Forward Layers Build Predictions by
Promoting Concepts in the Vocabulary Space.** `arXiv`  
   *Mor Geva, Avi Caciularu, Kevin Ro Wang, Yoav Goldberg.* [[PDF](https://arxiv.org/pdf/2203.14680)][[Code]( https://github.com/aviclu/ffn-values.
)], 2022

4. **Locating and Mitigating Gender Bias in Large Language Models.** `arXiv`  
   *Yuchen Cai, Ding Cao, Rongxi Guo, Yaqin Wen, Guiquan Liu, Enhong Chen.* [[PDF](https://arxiv.org/pdf/2403.14409)], 2024

5. **Debiasing Algorithm through Model Adaptation.** `arXiv`  
   *Tomasz Limisiewicz, David Mareček, Tomáš Musil.* [[PDF](https://arxiv.org/pdf/2310.18913)][[Code](https://github.com/tomlimi/DAMA)], 2023

