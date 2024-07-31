# 检索增强生成

- [检索增强生成](#检索增强生成)
  - [检索增强生成简介](#检索增强生成简介)
  - [检索增强生成架构](#检索增强生成架构)
  - [知识检索](#知识检索)
  - [生成增强](#生成增强)
  - [实践与应用](#实践与应用)


## <img src="../figure/star.svg" width="25" height="25" />检索增强生成简介

1. **No free lunch theorems for optimization.** `IEEE Transactions on Evolutionary Computation`   
   *David H. Wolp ert, William G. Macready* [[PDF](https://ieeexplore.ieee.org/document/585893)], 1997
   
1. **Retrieval-augmented generation for knowledge-intensive nlp tasks.** `NeurIPS`   
   *Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela* [[PDF](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)], 2020

## <img src="../figure/star.svg" width="25" height="25" />检索增强生成架构

1. **In-context retrieval-augmented language models.** `Transactions of the Association for Computational Linguistics`  
    *Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham.* [[PDF](https://arxiv.org/pdf/2302.00083)][[Code](https://github.com/ai21labs/in-context-ralm)], 2023

2. **Replug: Retrieval-augmented black-box language models.** `arXiv`  
    *Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih.* [[PDF](https://arxiv.org/abs/2301.12652)], 2023

3. **Atlas: Few-shot learning with retrieval augmented language models.** `Journal of Machine Learning Research`  
    *Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, Edouard Grave.*  [[PDF](https://arxiv.org/pdf/2208.03299)][[Code](https://github.com/facebookresearch/atlas)], 2023

4. **Improving language models by retrieving from trillions of tokens.** `ICML` 
    *Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark.* [[PDF](https://arxiv.org/pdf/2112.04426)][[Code](https://github.com/lucidrains/RETRO-pytorch)], 2022

5. **Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In.** `arXiv`
    *Zichun Yu, Chenyan Xiong, Shi Yu, Zhiyuan Liu*. [[PDF](https://arxiv.org/abs/2305.17331)][[Code](https://github.com/openmatch/augmentation-adapted-retriever)], 2023

6. **Self-rag: Learning to retrieve, generate, and critique through self-reflection.** `arXiv`  
   *Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi.* [[PDF](https://arxiv.org/abs/2310.11511)][[Code](https://github.com/AkariAsai/self-rag)], 2023



## <img src="../figure/star.svg" width="25" height="25" />知识检索

1. **The Chronicles of RAG: The Retriever, the Chunk and the Generator.** `arXiv`  
    *Paulo Finardi, Leonardo Avila, Rodrigo Castaldoni, Pedro Gengo, Celio Larcher, Marcos Piau, Pablo Costa, Vinicius Carid{\'a}*. [[PDF](https://arxiv.org/abs/2401.07883)], 2024

2. **LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding.** `arXiv`  
    *Mingrui Wu, Sheng Cao*. [[PDF](https://arxiv.org/abs/2404.05825)], 2024

3. **Generate rather than retrieve: Large language models are strong context generators.** `ICLR`  
    *Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, Meng Jiang.* [[PDF](https://arxiv.org/pdf/2209.10063)][[Code](https://github.com/wyu97/GenRead)], 2023    

4. **An information-theoretic perspective of tf--idf measures.** `IPM`  
    *Akiko Aizawa.* [[PDF](https://doi.org/10.1016/S0306-4573(02)00021-3)], 2003

5. **The probabilistic relevance framework: BM25 and beyond.** `Foundations and Trends in Information Retrieval`  
    *Stephen Robertson, Hugo Zaragoza.* [[PDF](https://dl.acm.org/doi/10.1561/1500000019)], 2009

6. **Investigating the Effects of Sparse Attention on Cross-Encoders.** `ECIR`  
    *Ferdinand Schlatt, Maik Fr{\"o}be, Matthias Hagen.* [[PDF](https://arxiv.org/pdf/2312.17649)][[Code](https://github.com/webis-de/ecir-24)], 2024

7. **A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE.** `arXiv`  
    *Herv{\'e} D{\'e}jean, St{\'e}phane Clinchant, Thibault Formal.* [[PDF](https://arxiv.org/abs/2403.10407)], 2024

8. **Dense passage retrieval for open-domain question answering.** `EMNLP`  
   *Vladimir Karpukhin, Barlas O{\u{g}}uz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.* [[PDF](https://arxiv.org/pdf/2004.04906)][[Code](https://github.com/facebookresearch/DPR)], 2020

9. **Colbert: Efficient and effective passage search via contextualized late interaction over bert.** `SIGIR`  
    *Omar Khattab, Matei Zaharia.* [[PDF](https://arxiv.org/pdf/2004.12832)][[Code](https://github.com/stanford-futuredata/ColBERT)], 2020

10. **Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring.** `arXiv`  
    *Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, Jason Weston.* [[PDF](https://arxiv.org/abs/1905.01969)][[Code](https://github.com/sfzhou5678/PolyEncoder)], 2019

11. **Transformer memory as a differentiable search index.** `Advances in Neural Information Processing Systems`  
    *Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta.* [[PDF](https://arxiv.org/pdf/2202.06991)][[Code](https://github.com/ArvinZhuang/DSI-transformers)], 2022

12. **From matching to generation: A survey on generative information retrieval.** `arXiv`  
    *Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, Zhicheng Dou.* [[PDF](https://arxiv.org/abs/2404.14851)], 2024

13. **A Neural Corpus Indexer for Document Retrieval.** `arXiv`  
    *Yujing Wang, Ying Hou, Hong Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, Mao Yang.* [[PDF](https://arxiv.org/abs/2206.02743)], 2022

14. **Multidimensional binary search trees used for associative searching.** `Communications of the ACM`  
    *Jon Louis Bentley.* [[PDF](https://dl.acm.org/doi/10.1145/361002.361007)], 1975

15. **Ball\*-tree: Efficient spatial indexing for constrained nearest-neighbor search in metric spaces.** `arXiv`  
    *Mohamad Dolatshah, Ali Hadian, Behrouz Minaei-Bidgoli.* [[PDF](https://arxiv.org/abs/1511.00628)], 2015

16. **Approximate nearest neighbor algorithm based on navigable small world graphs.** `Information Systems`  
    *Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, Vladimir Krylov.* [[PDF](https://doi.org/10.1016/j.is.2013.10.006)], 2014

17. **Non-metric similarity graphs for maximum inner product search.** `Advances in Neural Information Processing Systems`  
    *Stanislav Morozov, Artem Babenko.* [[PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/229754d7799160502a143a72f6789927-Paper.pdf)][[Code](https://github.com/stanis-morozov/ip-nsw)], 2018

18. **Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs.** `IEEE Transactions on Pattern Analysis and Machine Intelligence`  
    *Yu A Malkov, Dmitry A Yashunin.* [[PDF](https://arxiv.org/pdf/1603.09320)][[Code](https://github.com/nmslib/hnswlib)], 2018

19. **Product quantization for nearest neighbor search.** `IEEE Transactions on Pattern Analysis and Machine Intelligence`  
    *Herve Jegou, Matthijs Douze, Cordelia Schmid.* [[PDF](https://ieeexplore.ieee.org/document/5432202)], 2010

20. **Optimized product quantization for approximate nearest neighbor search.** `CVPR`  
    *Tiezheng Ge, Kaiming He, Qifa Ke, Jian Sun.* [[PDF](https://ieeexplore.ieee.org/document/6619223)], 2013

21. **Searching in one billion vectors: re-rank with source coding.** `ICASSP`  
    *Herv{\'e} J{\'e}gou, Romain Tavenard, Matthijs Douze, Laurent Amsaleg.* [[PDF](https://arxiv.org/pdf/1102.3828)], 2011

22. **Is ChatGPT good at search? Investigating large language models as re-ranking agent.** `arXiv`  
    *Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, Zhaochun Ren.* [[PDF](https://arxiv.org/abs/2304.09542)][[Code](https://github.com/sunnweiwei/rankgpt)], 2023



## <img src="../figure/star.svg" width="25" height="25" />生成增强
1. **Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models.** `EMNLP`   
   *Potsawee Manakul, Adian Liusie, Mark JF Gales* [[PDF](https://aclanthology.org/2023.emnlp-main.557.pdf)] [[Code](https://github.com/potsawee/selfcheckgpt)], 2023
   
2. **Predicting Question-Answering Performance of Large Language Models through Semantic Consistency.** `arXiv`   
   *Ella Rabinovich, Samuel Ackerman, Orna Raz, Eitan Farchi, Ateret Anaby Tavor* [[PDF](https://arxiv.org/pdf/2311.01152)], 2020
   
3. **Large language models struggle to learn long-tail knowledge.** `ICML`   
   *Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel* [[PDF](https://proceedings.mlr.press/v202/kandpal23a/kandpal23a.pdf)] [[Code](https://github.com/nkandpa2/long_tail_knowledge)], 2023
   
4. **When not to trust language models: Investigating effectiveness of parametric and non-parametric memories.** `ACL`   
   *Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, Daniel Khashabi* [[PDF](https://aclanthology.org/2023.acl-long.546)] [[Code](https://github.com/AlexTMallen/adaptive-retrieval)], 2023

5. **Locating and editing factual associations in GPT.** `NeurIPS`   
   *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov* [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf)] [[Code](https://github.com/kmeng01/rome)], 2022

6. **Learning to trust your feelings: Leveraging self-awareness in llms for hallucination mitigation.** `arXiv`   
   *Yuxin Liang, Zhuoyang Song, Hao Wang, Jiaxing Zhang* [[PDF](https://arxiv.org/pdf/2401.15449)][[Code](https://github.com/liangyuxin42/dreamcatcher)], 2024
   
7. **Improving Language Models via Plug-and-Play Retrieval Feed-back.** `arXiv`   
   *Wenhao Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang, Ashish Sabharwal* [[PDF](https://arxiv.org/pdf/2305.14002)], 2023
   
8. **Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp.** `arXiv`   
   *Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, Matei Zaharia* [[PDF](https://arxiv.org/abs/2212.14024)][[Code](https://github.com/stanfordnlp/dsp)], 2022
   
9. **Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models.** `EMNLP`   
   *Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, Jaewoo Kang* [[PDF](https://aclanthology.org/2023.emnlp-main.63/)][[Code](https://github.com/gankim/tree-of-clarifications)], 2023
   
10. **Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression.** `arXiv`   
      *Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu* [[PDF](https://arxiv.org/abs/2310.06839)][[Code](https://github.com/microsoft/LLMLingua)], 2023
    
11. **FIT-RAG: Black-Box RAG with Factual Information and Token Reduction.** `ACM Transactions on Information Systems`   
      *Yuren Mao, Xuemei Dong, Wenyi Xu, Yunjun Gao, Bin Wei, Ying Zhang* [[PDF](https://dl.acm.org/doi/pdf/10.1145/3676957)], 2024
    
12. **Prca: Fitting black-box large language models for retrieval question answering via pluggable reward-driven contextual adapter.** `EMNLP`   
      *Haoyan Yang, Zhitao Li, Yong Zhang, Jianzong Wang, Ning Cheng, Ming Li, Jing Xiao* [[PDF](https://aclanthology.org/2023.emnlp-main.326/)], 2023
    
13. **Triforce: Lossless acceleration of long sequence generation with hierarchical speculative decodingr.** `arXiv`   
      *Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, Beidi Chen* [[PDF](https://arxiv.org/abs/2404.11912)][[Code](https://github.com/Infini-AI-Lab/TriForce)], 2024
    
14. **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented
      Generation.** `arXiv`   
      *Chao Jin, Zili Zhang,  Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin* [[PDF](https://arxiv.org/html/2404.12457v1)], 2024

## <img src="../figure/star.svg" width="25" height="25" />实践与应用
1. **A survey on large language model based autonomous agents.** `Frontiers of Computer Science`   
      *Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, Beidi Chen* [[PDF](https://link.springer.com/article/10.1007/s11704-024-40231-1)][[Code](https://github.com/Paitesanshi/LLM-Agent-Survey)], 2024
      
2. **Multimodal prompt retrieval for generative visual question answering.** `ACL`   
      *Timothy Ossowski, Junjie Hu* [[PDF](https://aclanthology.org/2023.findings-acl.158.pdf)][[Code](https://github.com/tossowski/MultimodalPromptRetrieval)], 2023

3. **FinTextQA: A Dataset for Long-form Financial Question Answering.** `arXiv`   
      *Jian Chen, Peilin Zhou, Yining Hua, Yingxin Loh, Kehui Chen, Ziyuan Li, Bing Zhu, Junwei Liang* [[PDF](https://arxiv.org/pdf/2405.09980)], 2024
      
4. **Retrieval-based controllable molecule generation.** `ICLR`   
      *Zichao Wang, Weili Nie, Zhuoran Qiao, Chaowei Xiao, Richard Baraniuk, Anima Anandkumarn* [[PDF](https://openreview.net/pdf?id=vDFA1tpuLvk)][[Code](https://github.com/NVlabs/RetMol)], 2022
      
5. **Re-imagen: Retrieval-augmented text-to-image generator.** `arXiv`   
      *Wenhu Chen, Hexiang Hu, Chitwan Saharia, William W. Cohen* [[PDF](https://arxiv.org/pdf/2209.14491)], 2022
      
6. **Using external off-policy speech-to-text mappings in contextual end-to-end automated speech recognition.** `arXiv`   
      *David M. Chan, Shalini Ghosh, Ariya Rastrow, Björn Hoffmeister* [[PDF](https://arxiv.org/pdf/2301.02736)], 2023

7. **Language models with image descriptors are strong few-shot video-language learners.** `NeurIPS`   
      *Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, Shih-Fu Chang, Mohit Bansal, Heng Ji* [[PDF](https://papers.neurips.cc/paper_files/paper/2022/file/381ceeae4a1feb1abc59c773f7e61839-Paper-Conference.pdf)][[Code](https://github.com/MikeWangWZHL/VidIL)], 2022
