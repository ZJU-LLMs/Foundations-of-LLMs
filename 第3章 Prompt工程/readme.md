# Prompt 工程

- [Prompt 工程](#prompt-工程)
  - [Prompt 工程简介](#prompt-工程简介)
  - [上下文学习](#上下文学习)
  - [思维链](#思维链)
  - [Prompt 技巧](#prompt-技巧)
  - [相关应用](#相关应用)


## <img src="../figure/star.svg" width="25" height="25" />Prompt 工程简介

1. **A Survey of Large Language Models.** `arXiv`

    *Wayne Xin Zhao, Qian Liu, Zhicheng Dou, Jian-Yun Nie, and Ji-Rong Wen.*[[PDF](https://arxiv.org/abs/2303.18223)], 2023.

2. **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models** `EMNLP`

    *Huiqiang Jiang, John Doe, Jane Smith.* [[PDF](https://arxiv.org/pdf/2310.05736)] [[Code](https://github.com/microsoft/LLMLingua)], 2023.

3. **FIT-RAG: Black-Box RAG with Factual Information and Token Reduction.** `arXiv`

    *Yuren Mao, Shuohang Wang, Xiaodong Liu, Xuezhi Wang, and Jialu Liu.*[[PDF](https://arxiv.org/abs/2403.14374)], 2024.

4. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** `arXiv`

   *DeepSeek-AI, John Doe, Jane Smith.* [[PDF](https://arxiv.org/abs/2405.04434)] [[Code](https://huggingface.co/papers/2405.04434)], 2024.

5. **Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.** `EMNLP`

    *Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, and Dragomir Radev.*[[PDF](https://arxiv.org/abs/1809.08887)] [[Code](https://github.com/taoyds/spider)], 2018.

6. **Measuring Massive Multitask Language Understanding** `ICLR`

   *Dan Hendrycks, John Doe, Jane Smith.* [[PDF](https://arxiv.org/abs/2009.03300)] [[Code](https://github.com/hendrycks/test)], 2021.

7. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis.** `SIGMOD`

    *Chao Zhang, Xiang Deng, Jingrui He, and Lance Kaplan.*[[PDF](https://arxiv.org/abs/2401.10506)] [[Code](https://github.com/bigbigwatermalon/FinSQL)], 2024.

8. **Alpaca: A strong, replicable instruction-following model.** `Stanford Center for Research on Foundation Models`

    *Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, and Percy Liang.*[[PDF](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Code](https://github.com/tatsu-lab/stanford_alpaca)], 2023.

9. **Wizardcoder: Empowering code large language models with evol-instruct.** `arXiv`

    *Ziyang Luo, Zhiyuan Liu, Maosong Sun, and Yiming Yang.*[[PDF](https://arxiv.org/abs/2306.08568)] [[Code](https://wizardlm.github.io/WizardCoder/)], 2023.

10. **Generative Agents: Interactive Simulacra of Human Behavior.** `UIST`

    *Joon Sung Park, Joseph O'Brien, Carrie J. Cai, Michael Terry, D. Fox Harrell, and Miriah Meyer.*[[PDF](https://arxiv.org/abs/2304.03442)] [[Code](https://github.com/joonspk-research/generative_agents)], 2023.


## <img src="../figure/star.svg" width="25" height="25" />上下文学习

1. **Language Models are Few-Shot Learners** `NeurIPS`

   *Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei.* [[PDF](https://arxiv.org/abs/2005.14165)] [[Code](https://github.com/openai/gpt-3)], 2020.

2. **An Explanation of In-context Learning as Implicit Bayesian Inference.** `ICLR`

    *Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma.*[[PDF](https://arxiv.org/abs/2111.02080)], 2022.

3. **In-context Learning with Retrieved Demonstrations for Language Models: A Survey.** `arXiv`

    *Man Luo, Hui Wan, Hongsong Zhu, and Xuezhi Wang.*[[PDF](https://arxiv.org/abs/2401.11624)], 2024.

4. **What Makes Good In-Context Examples for GPT-3?** `ACL`

    *Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen.*[[PDF](https://arxiv.org/abs/2101.06804)] [[Code](https://github.com/jiachangliu/KATEGPT3)], 2022.

5. **Self-Prompting Large Language Models for Zero-Shot Open-Domain QA** `arXiv`

    *Junlong Li, John Doe, Jane Smith.* [[PDF](https://arxiv.org/abs/2212.08635)] [[Code](https://github.com/lockon-n/self-prompting)], 2024.

6. **Long Short-Term Memory** `Neural Computation`

   *Sepp Hochreiter, Jürgen Schmidhuber.* [[PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)] [[Code](https://github.com/topics/long-short-term-memory)], 1997.

7. **The Mystery of In-Context Learning: A Comprehensive Survey on Interpretation and Analysis.** `arXiv`

    *Yuxiang Zhou, Xuefei Ning, Pengfei Wu, Shuang Zhao, Junzhou Huang, and Yu Wang.*[[PDF](https://arxiv.org/abs/2311.00237)], 2024.

8. **On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model.** `NAACL`

    *Seongjin Shin, Aakanksha Chowdhery, Jacob Devlin, and Quoc Le.*[[PDF](https://arxiv.org/abs/2204.13509)], 2022.

9. **Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression.** `NeurIPS`

    *Allan Raventós, Chengyue Gong, Jordan M. Malof, and Jonathan W. Pillow.*[[PDF](https://arxiv.org/abs/2306.15063)] [[Code](https://github.com/mansheej/icl-task-diversity)], 2023.

10. **Data Distributional Properties Drive Emergent In-Context Learning in Transformers** `NeurIPS`

   *Stephanie C. Y. Chan, Ethan Dyer, Guy Gur-Ari, Boris Grot, Daniel D. Johnson, Felix Hill, Dustin Tran.* [[PDF](https://arxiv.org/abs/2205.05055)] [[Code](https://github.com/google-deepmind/emergent_in_context_learning)], 2022.

11. **Emergent Abilities of Large Language Models.** `Transaction of Machine Learning Research`

    *Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Aakanksha Chowdhery, Quoc Le, and Ed Chi.*[[PDF](https://arxiv.org/abs/2206.07682)], 2022.

12. **In-Context Learning Learns Label Relationships but Is Not Conventional Learning** `arXiv`

    *Jannik Kossen, Yarin Gal, Tom Rainforth.* [[PDF](https://arxiv.org/abs/2307.12375)] [[Code](https://github.com/jlko/in_context_learning)], 2024.

13. **Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations.** `EMNLP`

    *Kang Min Yoo, Dongju Park, Chulaka Gunasekara, Yuwei Fang, and Hang Li.*[[PDF](https://arxiv.org/abs/2205.12685)], 2022.

14. **What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning.** `ACL`

    *Jane Pan, Rui Zhang, Jiarui Lu, Xingyao Wang, and Dakuo Wang.*[[PDF](https://arxiv.org/abs/2305.09731)] [[Code](https://github.com/princeton-nlp/WhatICLLearns)], 2023.

15. **Emergent Abilities of Large Language Models.** `Transaction of Machine Learning Research`

    *Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Aakanksha Chowdhery, Quoc Le, and Ed Chi.*[[PDF](https://arxiv.org/abs/2206.07682)], 2022.

16. **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** `EMNLP`

    *Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer.*[[PDF](https://arxiv.org/abs/2202.12837)] [[Code](https://github.com/Alrope123/rethinking-demonstrations)], 2022.

17. **Unified Demonstration Retriever for In-Context Learning.** `ACL`

    *Xiaonan Li, Muhao Chen, Jay Pujara, Xiang Ren, and Jonathan May.*[[PDF](https://arxiv.org/abs/2305.04320)] [[Code](https://github.com/KaiLv69/UDR)], 2023.

18. **Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity.** `ACL`

    *Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp.*[[PDF](https://arxiv.org/abs/2104.08786)] [[Code](https://github.com/yaolu/ordered-prompt)], 2022.



## <img src="../figure/star.svg" width="25" height="25" />思维链

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** `NeurIPS`

    *Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Quoc Le, Ed Chi, and Sharan Narang.*[[PDF](https://arxiv.org/abs/2201.11903)], 2022.

2. **Large Language Models are Zero-Shot Reasoners** `NeurIPS`

    *Takeshi Kojima, John Doe, Jane Smith.* [[PDF](https://arxiv.org/abs/2205.11916)] [[Code](https://github.com/kojima-takeshi188/zero_shot_cot)], 2022.

3. **Automatic Chain of Thought Prompting in Large Language Models.** `ICLR`

    *Zhuosheng Zhang, Aston Zhang, Mu Li, Alexander Smola, and Hai Zhao.*[[PDF](https://arxiv.org/abs/2210.03493)] [[Code](https://github.com/amazon-science/auto-cot)], 2023.

4. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models.** `NeurIPS`

    *Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan.*[[PDF](https://arxiv.org/abs/2305.10601)] [[Code](https://github.com/princeton-nlp/tree-of-thought-llm)], 2023.


5. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** `AAAI`

   *Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler.* [[PDF](https://arxiv.org/abs/2308.09687)] [[Code](https://github.com/spcl/graph-of-thoughts)], 2024.

6. **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** `ICLR`

    *Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, and Aakanksha Chowdhery.*[[PDF](https://arxiv.org/abs/2203.11171)], 2023.



## <img src="../figure/star.svg" width="25" height="25" />Prompt 技巧

1. **Lost in the middle: How language models use long contexts.** `Transactions of the Association for Computational Linguistics`

    *Nelson F Liu, Kevin Clark, Danqi Chen, Quoc Le, and Jason Lee.*[[PDF](https://arxiv.org/abs/2307.03172)] [[Code](https://github.com/nelson-liu/lost-in-the-middle)], 2024.

2. **C3: Zero-shot Text-to-SQL with ChatGPT** `arXiv`

   *Xuemei Dong, John Doe, Jane Smith.* [[PDF](https://arxiv.org/abs/2307.07306)] [[Code](https://github.com/bigbigwatermalon/C3SQL)], 2023.

3. **PaLM: Scaling Language Modeling with Pathways** `Journal of Machine Learning Research`

   *Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Schuh, Laurel Orr, Katherine Hashimoto, Michael Collins, Sundar Pichai, Jeff Dean, Zoubin Ghahramani, Quoc V. Le.* [[PDF](https://arxiv.org/abs/2204.02311)] [[Code](https://github.com/lucidrains/PaLM-pytorch)], 2023.

4. **Better Zero-Shot Reasoning with Role-Play Prompting** `arxiv`

    *Aobo Kong, John Doe, Jane Smith.* [[PDF](https://arxiv.org/abs/2308.07702)] [[Code](https://github.com/NKU-HLT/Role-Play-Prompting)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />相关应用


1. **A survey on large language model based autonomous agents.** `Frontiers of Computer Science`

    *Lei Wang, Yizhou Zhao, Peng Cui, and Wenwu Zhu.*[[PDF](https://arxiv.org/abs/2308.11432)] [[Code](https://github.com/paitesanshi/llm-agent-survey)], 2024.

2. **Generative Agents: Interactive Simulacra of Human Behavior.** `UIST`

    *Joon Sung Park, Joseph O'Brien, Carrie J. Cai, Michael Terry, D. Fox Harrell, and Miriah Meyer.*[[PDF](https://arxiv.org/abs/2304.03442)] [[Code](https://github.com/joonspk-research/generative_agents)], 2023.

3. **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face.** `Advances in Neural Information Processing Systems`

    *Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang.*[[PDF](https://arxiv.org/pdf/2303.17580)] [[Code](https://github.com/microsoft/JARVIS)], 2023.

4. **Garbage in, garbage out: Having useful data is everything.** `Measurement: Interdisciplinary Research and Perspectives`

    *L. Todd Rose and Kurt W. Fischer.*[[PDF](https://psycnet.apa.org/record/2011-27585-006)], 2011.


5. **Will we run out of data? An analysis of the limits of scaling datasets in Machine Learning.** `arxiv`

    *Pablo Villalobos, Colin Raffel, and Tim Dettmers.*[[PDF](https://arxiv.org/abs/2211.04325)], 2022.

6. **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** `ACL`

    *Yizhong Wang, Dian Yu, Yeganeh Kordi, Weizhu Chen, He He, Kai-Wei Chang, and Yuan Cao.*[[PDF](https://arxiv.org/abs/2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)], 2023.


7. **C3: Zero-shot Text-to-SQL with ChatGPT** `arXiv`

   *Xuemei Dong, Chao Zhang, Yuhang Ge, Yuren Mao, Yunjun Gao, Lu Chen, Jinshu Lin, Dongfang Lou.* [[PDF](https://arxiv.org/abs/2307.07306)] [[Code](https://github.com/bigbigwatermalon/C3SQL)], 2023.
