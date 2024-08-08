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

    *Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu.* [[PDF](https://arxiv.org/pdf/2310.05736)] [[Code](https://github.com/microsoft/LLMLingua)], 2023.

3. **FIT-RAG: Black-Box RAG with Factual Information and Token Reduction.** `arXiv`

    *Yuren Mao, Xuemei Dong, Wenyi Xu, Yunjun Gao, Bin Wei, Ying Zhang.*[[PDF](https://arxiv.org/abs/2403.14374)], 2024.

4. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** `arXiv`

   *DeepSeek-AI.* [[PDF](https://arxiv.org/abs/2405.04434)] [[Code](https://huggingface.co/papers/2405.04434)], 2024.

5. **Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.** `EMNLP`

    *Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev.*[[PDF](https://arxiv.org/abs/1809.08887)] [[Code](https://github.com/taoyds/spider)], 2018.

6. **Measuring Massive Multitask Language Understanding** `ICLR`

   *Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt.* [[PDF](https://arxiv.org/abs/2009.03300)] [[Code](https://github.com/hendrycks/test)], 2021.

7. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis.** `SIGMOD`

    *Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin.*[[PDF](https://arxiv.org/abs/2401.10506)] [[Code](https://github.com/bigbigwatermalon/FinSQL)], 2024.

8. **Alpaca: A strong, replicable instruction-following model.** `Stanford Center for Research on Foundation Models`

    *Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, and Percy Liang.*[[PDF](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Code](https://github.com/tatsu-lab/stanford_alpaca)], 2023.

9. **Wizardcoder: Empowering code large language models with evol-instruct.** `arXiv`

    *Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang.*[[PDF](https://arxiv.org/abs/2306.08568)] [[Code](https://wizardlm.github.io/WizardCoder/)], 2023.

10. **Generative Agents: Interactive Simulacra of Human Behavior.** `UIST`

    *Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein.*[[PDF](https://arxiv.org/abs/2304.03442)] [[Code](https://github.com/joonspk-research/generative_agents)], 2023.


## <img src="../figure/star.svg" width="25" height="25" />上下文学习

1. **Language Models are Few-Shot Learners** `NeurIPS`

   *Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei.* [[PDF](https://arxiv.org/abs/2005.14165)] [[Code](https://github.com/openai/gpt-3)], 2020.

2. **An Explanation of In-context Learning as Implicit Bayesian Inference.** `ICLR`

    *Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma.*[[PDF](https://arxiv.org/abs/2111.02080)], 2022.

3. **In-context Learning with Retrieved Demonstrations for Language Models: A Survey.** `arXiv`

    *Man Luo, Xin Xu, Yue Liu, Panupong Pasupat, Mehran Kazemi.*[[PDF](https://arxiv.org/abs/2401.11624)], 2024.

4. **What Makes Good In-Context Examples for GPT-3?** `ACL`

    *Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, Weizhu Chen.*[[PDF](https://arxiv.org/abs/2101.06804)] [[Code](https://github.com/jiachangliu/KATEGPT3)], 2022.

5. **Self-Prompting Large Language Models for Zero-Shot Open-Domain QA** `arXiv`

    *Junlong Li, Jinyuan Wang, Zhuosheng Zhang, Hai Zhao.* [[PDF](https://arxiv.org/abs/2212.08635)] [[Code](https://github.com/lockon-n/self-prompting)], 2024.

6. **Long Short-Term Memory** `Neural Computation`

   *Sepp Hochreiter, Jürgen Schmidhuber.* [[PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)] [[Code](https://github.com/topics/long-short-term-memory)], 1997.

7. **The Mystery of In-Context Learning: A Comprehensive Survey on Interpretation and Analysis.** `arXiv`

    *Yuxiang Zhou, Jiazheng Li, Yanzheng Xiang, Hanqi Yan, Lin Gui, Yulan He.*[[PDF](https://arxiv.org/abs/2311.00237)], 2024.

8. **On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model.** `NAACL`

    *Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woomyoung Park, Jung-Woo Ha, Nako Sung.*[[PDF](https://arxiv.org/abs/2204.13509)], 2022.

9. **Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression.** `NeurIPS`

    *Allan Raventós, Mansheej Paul, Feng Chen, Surya Ganguli.*[[PDF](https://arxiv.org/abs/2306.15063)] [[Code](https://github.com/mansheej/icl-task-diversity)], 2023.

10. **Data Distributional Properties Drive Emergent In-Context Learning in Transformers** `NeurIPS`

   *Stephanie C.Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, Felix Hill.* [[PDF](https://arxiv.org/abs/2205.05055)] [[Code](https://github.com/google-deepmind/emergent_in_context_learning)], 2022.

11. **Emergent Abilities of Large Language Models.** `Transaction of Machine Learning Research`

    *Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus.*[[PDF](https://arxiv.org/abs/2206.07682)], 2022.

12. **In-Context Learning Learns Label Relationships but Is Not Conventional Learning** `arXiv`

    *Jannik Kossen, Yarin Gal, Tom Rainforth.* [[PDF](https://arxiv.org/abs/2307.12375)] [[Code](https://github.com/jlko/in_context_learning)], 2024.

13. **Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations.** `EMNLP`

    *Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang-goo Lee, Taeuk Kim.*[[PDF](https://arxiv.org/abs/2205.12685)], 2022.

14. **What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning.** `ACL`

    *Jane Pan, Tianyu Gao, Howard Chen, Danqi Chen.*[[PDF](https://arxiv.org/abs/2305.09731)] [[Code](https://github.com/princeton-nlp/WhatICLLearns)], 2023.

15. **Emergent Abilities of Large Language Models.** `Transaction of Machine Learning Research`

    *Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus.*[[PDF](https://arxiv.org/abs/2206.07682)], 2022.

16. **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** `EMNLP`

    *Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer.*[[PDF](https://arxiv.org/abs/2202.12837)] [[Code](https://github.com/Alrope123/rethinking-demonstrations)], 2022.

17. **Unified Demonstration Retriever for In-Context Learning.** `ACL`

    *Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, Xipeng Qiu.*[[PDF](https://arxiv.org/abs/2305.04320)] [[Code](https://github.com/KaiLv69/UDR)], 2023.

18. **Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity.** `ACL`

    *Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp.*[[PDF](https://arxiv.org/abs/2104.08786)] [[Code](https://github.com/yaolu/ordered-prompt)], 2022.



## <img src="../figure/star.svg" width="25" height="25" />思维链

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** `NeurIPS`

    *Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou.*[[PDF](https://arxiv.org/abs/2201.11903)], 2022.

2. **Large Language Models are Zero-Shot Reasoners** `NeurIPS`

    *Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa.* [[PDF](https://arxiv.org/abs/2205.11916)] [[Code](https://github.com/kojima-takeshi188/zero_shot_cot)], 2022.

3. **Automatic Chain of Thought Prompting in Large Language Models.** `ICLR`

    *Zhuosheng Zhang, Aston Zhang, Mu Li, Alex Smola.*[[PDF](https://arxiv.org/abs/2210.03493)] [[Code](https://github.com/amazon-science/auto-cot)], 2023.

4. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models.** `NeurIPS`

    *Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan.*[[PDF](https://arxiv.org/abs/2305.10601)] [[Code](https://github.com/princeton-nlp/tree-of-thought-llm)], 2023.


5. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** `AAAI`

   *Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler.* [[PDF](https://arxiv.org/abs/2308.09687)] [[Code](https://github.com/spcl/graph-of-thoughts)], 2024.

6. **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** `ICLR`

    *Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou.*[[PDF](https://arxiv.org/abs/2203.11171)], 2023.



## <img src="../figure/star.svg" width="25" height="25" />Prompt 技巧

1. **Lost in the middle: How language models use long contexts.** `Transactions of the Association for Computational Linguistics`

    *Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang.*[[PDF](https://arxiv.org/abs/2307.03172)] [[Code](https://github.com/nelson-liu/lost-in-the-middle)], 2024.

2. **C3: Zero-shot Text-to-SQL with ChatGPT** `arXiv`

   *Xuemei Dong, Chao Zhang, Yuhang Ge, Yuren Mao, Yunjun Gao, Lu Chen, Jinshu Lin, Dongfang Lou.* [[PDF](https://arxiv.org/abs/2307.07306)] [[Code](https://github.com/bigbigwatermalon/C3SQL)], 2023.

3. **PaLM: Scaling Language Modeling with Pathways** `Journal of Machine Learning Research`

   *Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel.* [[PDF](https://arxiv.org/abs/2204.02311)] [[Code](https://github.com/lucidrains/PaLM-pytorch)], 2023.

4. **Better Zero-Shot Reasoning with Role-Play Prompting** `arxiv`

    *Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, Xiaohang Dong.* [[PDF](https://arxiv.org/abs/2308.07702)] [[Code](https://github.com/NKU-HLT/Role-Play-Prompting)], 2023.

## <img src="../figure/star.svg" width="25" height="25" />相关应用


1. **A survey on large language model based autonomous agents.** `Frontiers of Computer Science`

    *Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen.*[[PDF](https://arxiv.org/abs/2308.11432)] [[Code](https://github.com/paitesanshi/llm-agent-survey)], 2024.

2. **Generative Agents: Interactive Simulacra of Human Behavior.** `UIST`

    *Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein.*[[PDF](https://arxiv.org/abs/2304.03442)] [[Code](https://github.com/joonspk-research/generative_agents)], 2023.

3. **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face.** `Advances in Neural Information Processing Systems`

    *Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang.*[[PDF](https://arxiv.org/pdf/2303.17580)] [[Code](https://github.com/microsoft/JARVIS)], 2023.

4. **Garbage in, garbage out: Having useful data is everything.** `Measurement: Interdisciplinary Research and Perspectives`

    *L. Todd Rose and Kurt W. Fischer.*[[PDF](https://psycnet.apa.org/record/2011-27585-006)], 2011.


5. **Will we run out of data? Limits of LLM scaling based on human-generated data.** `arxiv`

    *Pablo Villalobos, Colin Raffel, and Tim Dettmers.*[[PDF](https://arxiv.org/abs/2211.04325)], 2022.

6. **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** `ACL`

    *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.*[[PDF](https://arxiv.org/abs/2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)], 2023.


7. **C3: Zero-shot Text-to-SQL with ChatGPT** `arXiv`

   *Xuemei Dong, Chao Zhang, Yuhang Ge, Yuren Mao, Yunjun Gao, Lu Chen, Jinshu Lin, Dongfang Lou.* [[PDF](https://arxiv.org/abs/2307.07306)] [[Code](https://github.com/bigbigwatermalon/C3SQL)], 2023.
