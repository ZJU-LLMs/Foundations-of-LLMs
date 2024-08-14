# 大模型基础论文列表

- [语言模型基础](#语言模型基础)
  - [基于统计方法的语言模型](#基于统计方法的语言模型)
  - [基于 RNN 的语言模型](#基于-rnn-的语言模型)
  - [基于 Transformer 的语言模型](#基于-transformer-的语言模型)
  - [语言模型的采样方法](#语言模型的采样方法)
  - [语言模型的评测](#语言模型的评测)

- [大语言模型](#大语言模型)
  - [大数据+大模型→新智能](#大数据大模型新智能)
  - [大语言模型架构概览](#大语言模型架构概览)
  - [基于 Encoder-only 架构的大语言模型](#基于-encoder-only-架构的大语言模型)
  - [基于 Encoder-Decoder 架构的大语言模型](#基于-encoder-decoder-架构的大语言模型)
  - [基于 Decoder-only 架构的大语言模型](#基于-decoder-only-架构的大语言模型)
  - [非 Transformer 架构](#非-transformer-架构)

- [Prompt 工程](#prompt-工程)
  - [Prompt 工程简介](#prompt-工程简介)
  - [上下文学习](#上下文学习)
  - [思维链](#思维链)
  - [Prompt 技巧](#prompt-技巧)
  - [相关应用](#相关应用)

- [参数高效微调](#参数高效微调)
  - [参数高效微调简介](#参数高效微调简介)
  - [参数附加方法](#参数附加方法)
  - [参数选择方法](#参数选择方法)
  - [低秩适配方法](#低秩适配方法)
  - [实践与应用](#实践与应用)

- [模型编辑](#模型编辑)
  - [模型编辑简介](#模型编辑简介)
  - [模型编辑经典方法](#模型编辑经典方法)
  - [附加参数法：T-Patcher](#附加参数法t-patcher)
  - [定位编辑法：ROME](#定位编辑法rome)
  - [模型编辑应用](#模型编辑应用)

- [检索增强生成](#检索增强生成)
  - [检索增强生成简介](#检索增强生成简介)
  - [检索增强生成架构](#检索增强生成架构)
  - [知识检索](#知识检索)
  - [生成增强](#生成增强)
  - [实践与应用](#实践与应用)



## 语言模型基础


### <img src="..\figure\star.svg" width="25" height="25" />基于统计方法的语言模型

1. **Foundations of statistical natural language processing.** `BOOK`  
   *Chris Manning, Hinrich Sch{\"{u}}tze* [[PDF](https://nlp.stanford.edu/fsnlp/)], 1999
2. **Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics and Speech Recognition.Third Edition.** `BOOK`  
   *Daniel Jurafsky, James H. Martin* [[PDF](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)], 2023



### <img src="..\figure\star.svg" width="25" height="25" />基于 RNN 的语言模型

1. **A learning algorithm for continually running fully recurrent neural networks.** `Neural computation`  
   *RJ Williams, D Zipser.* [[PDF](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)], 1989

2. **Long Short-Term Memory.** `Neural Computing`  
   *Sepp Hochreiter, J{\"{u}}rgen Schmidhuber* [[PDF](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf)], 1997

3. **On the difficulty of training Recurrent Neural Networks.** `ICML`  
   *Razvan Pascanu, Tomas Mikolov, Yoshua Bengio.* [[PDF](https://arxiv.org/abs/1211.5063)], 2012

4. **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.** `arXiv`  
   *Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio* [[PDF](https://arxiv.org/abs/1412.3555)], 2014 

5. **Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.** `NeurIPS`  
   *Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer* [[PDF](https://arxiv.org/abs/1506.03099)], 2015    



### <img src="..\figure\star.svg" width="25" height="25" />基于 Transformer 的语言模型

1. **Layer Normalization.** `arXiv`  
   *Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton* [[PDF](https://arxiv.org/abs/1607.06450)], 2016

2. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.** `JMLR`  
   *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.* [[PDF](https://arxiv.org/abs/1910.10683)], 2019

3. **Transformer Feed-Forward Layers Are Key-Value Memories.** `EMNLP`  
   *Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy* [[PDF](https://arxiv.org/abs/2012.14913)], 2021

4. **ResiDual: Transformer with Dual Residual Connections.** `arXiv`  
   *Shufang Xie, Huishuai Zhang, Junliang Guo, Xu Tan, Jiang Bian, Hany Hassan Awadalla, Arul Menezes, Tao Qin, Rui Yan.* [[PDF](https://arxiv.org/abs/2304.14802)], 2023




### <img src="..\figure\star.svg" width="25" height="25" />语言模型的采样方法

1. **Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models.** `AAAI`  
   *Ashwin K Vijayakumar, Michael Cogswell, Ramprasath R. Selvaraju, Qing Sun, Stefan Lee, David Crandall, Dhruv Batra.* [[PDF](https://arxiv.org/abs/1610.02424)], 2018

2. **The Curious Case of Neural Text Degeneration.** `ICLR`  
   *Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi* [[PDF](https://arxiv.org/abs/1904.09751)], 2020




### <img src="..\figure\star.svg" width="25" height="25" />语言模型的评测


1. **Perplexity—a Measure of the Difficulty of Speech Recognition Tasks.** `JASA`  
   *F. Jelinek, R. L. Mercer, L. R. Bahl, J. K. Baker* [[PDF](https://pubs.aip.org/asa/jasa/article/62/S1/S63/642598/Perplexity-a-measure-of-the-difficulty-of-speech)], 1997

2. **ROUGE: A Package for Automatic Evaluation of Summaries.** `ACL`  
   *Chin-Yew Lin* [[PDF](https://aclanthology.org/W04-1013/)], 2004

3. **BLEU might be Guilty but References are not Innocent.** `EMNLP`  
   *Markus Freitag, David Grangier, Isaac Caswell* [[PDF](https://arxiv.org/abs/2004.06063)], 2020

4. **BERTScore: Evaluating Text Generation with BERT.** `ICLR`  
   *Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi.* [[PDF](https://arxiv.org/abs/1904.09675)], 2020

5. **Leveraging Large Language Models for NLG Evaluation: Advances and Challenges.** `arXiv`  
   *Zhen Li, Xiaohan Xu, Tao Shen, Can Xu, Jia-Chen Gu, Yuxuan Lai, Chongyang Tao, Shuai Ma* [[PDF](https://arxiv.org/abs/2401.07103)], 2024

6. **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.** `EMNLP`  
   *Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu* [[PDF](https://arxiv.org/abs/2303.16634)], 2023

7. **INSTRUCTSCORE: Towards Explainable Text Generation Evaluation with Automatic Feedback.** `EMNLP`  
   *Wenda Xu, Danqing Wang, Liangming Pan, Zhenqiao Song, Markus Freitag, William Wang, Lei Li.* [[PDF](https://aclanthology.org/2023.emnlp-main.365/)], 2023





## 大语言模型




### <img src="..\figure\star.svg" width="25" height="25" />大数据+大模型→新智能

1.  **Scaling laws for neural language models.** `arXiv`    
    *Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei.* [[PDF](https://arxiv.org/pdf/2001.08361)], 2020.

2.  **Training Compute-Optimal Large Language Models** `arXiv`    
    *Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre.* [[PDF](https://arxiv.org/pdf/2203.15556)], 2022.

3.  **PaLM 2 Technical Report.** `arXiv`    
    *Google.* [[PDF](https://arxiv.org/pdf/2305.10403)], 2023.



### <img src="..\figure\star.svg" width="25" height="25" />大语言模型架构概览

1.  **Attention is all you need.** `NeurIPS`  
    *Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia.* [[PDF](https://arxiv.org/pdf/1706.03762)], 2017.



### <img src="..\figure\star.svg" width="25" height="25" />基于 Encoder-only 架构的大语言模型

1.  **A survey on contextual embeddings.** `arXiv`    
    *Qi Liu, Matt J. Kusner, Phil Blunsom.* [[PDF](https://arxiv.org/pdf/2003.07278)], 2020.

2.  **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** `NAACL`
    *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.* [[PDF](https://aclanthology.org/N19-1423.pdf)][[Code](https://github.com/google-research/bert)], 2018.

3.  **RoBERTa: A Robustly Optimized BERT Pretraining Approach.** `arXiv`
    *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.* [[PDF](https://arxiv.org/abs/1907.11692)][[Code](https://github.com/pytorch/fairseq)], 2019.

4.  **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.** `arXiv`
    *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.* [[PDF](https://arxiv.org/pdf/1909.11942)][[Code](https://github.com/google-research/ALBERT)], 2019.

5.  **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.** `arXiv`
    *Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.* [[PDF](https://arxiv.org/pdf/2003.10555)][[Code](https://github.com/google-research/electra)], 2020.



### <img src="..\figure\star.svg" width="25" height="25" />基于 Encoder-Decoder 架构的大语言模型

1.  **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.** `arXiv`  
    *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.* [[PDF](https://arxiv.org/pdf/1910.10683)][[Code](https://github.com/google-research/text-to-text-transfer-transformer)], 2019.

2.  **Multitask Prompted Training Enables Zero-Shot Task Generalization.** `arXiv`  
    *Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Tali Bers, Stella Biderman, Leo Gao, Thomas Wolf, Alexander M. Rush.* [[PDF](https://arxiv.org/pdf/2110.08207)][[Code](https://github.com/bigscience-workshop/promptsource)], 2021.

3.  **mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer.** `NAACL`  
    *Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.* [[PDF](https://aclanthology.org/2021.naacl-main.41.pdf)][[Code](https://goo.gle/mt5-code)], 2021.

4.  **Scaling Instruction-Finetuned Language Models.** `Journal of Machine Learning Research`  
    *Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei.* [[PDF](https://www.jmlr.org/papers/volume25/23-0870/23-0870.pdf)][[Code](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)], 2024.

5.  **Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.** `ACL`  
    *Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer.* [[PDF](https://aclanthology.org/2020.acl-main.703.pdf)][[Code](https://github.com/facebookresearch/fairseq/blob/main/examples/bart)], 2020.

6.  **Multilingual denoising pre-training for neural machine translation.** `Transactions of the Association for Computational Linguistics`  
    *Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.* [[PDF](https://arxiv.org/pdf/2001.08210)][[Code](https://github.com/facebookresearch/fairseq/blob/main/examples/mbart)], 2020.



### <img src="..\figure\star.svg" width="25" height="25" />基于 Decoder-only 架构的大语言模型

1.  **Improving language understanding by generative pre-training.** `Online`  
    *Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever.* [[PDF](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)], 2018.

2.  **Language models are unsupervised multitask learners.** `Online`  
    *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.* [[PDF](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)], 2019.

3.  **Language models are few-shot learners.** `NeurIPS`  
    *Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei.* [[PDF](https://papers.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)], 2020.

4.  **Evaluating Large Language Models Trained on Code.** `arXiv`  
    *Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba.* [[PDF](https://arxiv.org/pdf/2107.03374)], 2021.

5.  **WebGPT: Browser-assisted question-answering with human feedback.** `arXiv`  
    *Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, John Schulman.* [[PDF](https://arxiv.org/pdf/2112.09332)], 2021.

6.  **Training language models to follow instructions with human feedback.** `NeurIPS`  
    *Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe.* [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)], 2022.

7.  **Introducing chatgpt.** `Online`  
    *OpenAI.* [[PDF](https://openai.com/blog/chatgpt)], 2023.

8.  **Gpt-4 technical report.** `Online`  
    *OpenAI.* [[PDF](https://openai.com/index/gpt-4-research)], 2023.

9.  **Gpt-4 technical report.** `Online`  
    *OpenAI.* [[PDF](https://openai.com/index/hello-gpt-4o)], 2024.

10.  **Gpt-4 technical report.** `Online`  
     *OpenAI.* [[PDF](https://openai.com/index/hello-gpt-4o)], 2024.

11.  **LLaMA: Open and Efficient Foundation Language Models.** `arXiv`  
     *Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.* [[PDF](https://arxiv.org/pdf/2302.13971)][[Code](https://github.com/facebookresearch/llama)], 2023.

12.  **Llama 2: Open Foundation and Fine-Tuned Chat Models.** `arXiv`  
     *Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom.* [[PDF](https://arxiv.org/pdf/2307.09288)][[Code](https://github.com/facebookresearch/llama/)], 2023.

13.  **Introducing Meta Llama 3: The most capable openly available LLM to date.** `Online`  
     *Meta AI.* [[PDF](https://ai.meta.com/blog/meta-llama-3/)][[Code](https://github.com/meta-llama/llama3)], 2024.

14.  **Alpaca: A Strong, Replicable Instruction-Following Model.** `Online`  
     *Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto.* [[PDF](https://crfm.stanford.edu/2023/03/13/alpaca.html)][[Code](https://github.com/tatsu-lab/stanford_alpaca)], 2023.

15.  **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%\* ChatGPT Quality.** `Online`  
     *The Vicuna Team.* [[PDF](https://lmsys.org/blog/2023-03-30-vicuna)][[Code](https://github.com/lm-sys/FastChat)], 2023.

16.  **QLoRA: Efficient Finetuning of Quantized LLMs.** `arXiv`  
     *Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer.* [[PDF](https://arxiv.org/pdf/2305.14314)][[Code](https://github.com/artidoro/qlora)], 2023.

17.  **Code Llama: Open Foundation Models for Code.** `arXiv`  
     *Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.* [[PDF](https://arxiv.org/pdf/2308.12950)][[Code](https://github.com/facebookresearch/codellama)], 2023.

18.  **A Brief Report on LawGPT 1.0: A Virtual Legal Assistant Based on GPT-3.** `arXiv`  
     *Ha-Thanh Nguyen.* [[PDF](https://arxiv.org/pdf/2302.05729)], 2023.

19.  **Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks.** `arXiv`  
     *Tiedong Liu, Bryan Kian Hsiang Low.* [[PDF](https://arxiv.org/pdf/2305.14201)][[Code](https://github.com/liutiedong/goat)], 2023.

20.  **Visual instruction tuning.** `NeurIPS`  
     *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee.* [[PDF](https://papers.nips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)][[Code](https://llava-vl.github.io/)], 2023.

21.  **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models.** `arXiv`  
     *Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny.* [[PDF](https://arxiv.org/pdf/2304.10592)][[Code](https://minigpt-4.github.io/)], 2023.



### <img src="..\figure\star.svg" width="25" height="25" />非 Transformer 架构

1. **Efficiently modeling long sequences with structured state spaces.** `arXiv`  
   *Albert Gu, Karan Goel, Christopher Ré.* [[PDF](https://arxiv.org/abs/2111.00396)][[Code](https://github.com/state-spaces/s4)], 2021.

2. **On the Parameterization and Initialization of Diagonal State Space Models.** `NeurIPS`  
   *Albert Gu, Karan Goel, Ankit Gupta, Christopher Ré.* [[PDF](https://arxiv.org/abs/2206.11893)], 2022.

3. **RWKV: Reinventing RNNs for the Transformer Era.** `EMNLP`  
   *Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan S. Wind, Stanislaw Wozniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, Rui-Jie Zhu* [[PDF](https://arxiv.org/abs/2305.13048)][[Code](https://github.com/BlinkDL/RWKV-LM)], 2023.

4. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** `arXiv`  
   *Albert Gu, Tri Dao.* [[PDF](https://arxiv.org/abs/2312.00752)][[Code](https://github.com/state-spaces/mamba)], 2023.

5. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** `arXiv`  
   *Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al.* [[PDF](https://arxiv.org/abs/2407.04620)][[Code](https://github.com/test-time-training/ttt-lm-pytorch)], 2024.





## Prompt 工程


### <img src="..\figure\star.svg" width="25" height="25" />Prompt 工程简介

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



### <img src="..\figure\star.svg" width="25" height="25" />上下文学习

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



### <img src="..\figure\star.svg" width="25" height="25" />思维链

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



### <img src="..\figure\star.svg" width="25" height="25" />Prompt 技巧

1. **Lost in the middle: How language models use long contexts.** `Transactions of the Association for Computational Linguistics`

   *Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang.*[[PDF](https://arxiv.org/abs/2307.03172)] [[Code](https://github.com/nelson-liu/lost-in-the-middle)], 2024.

2. **C3: Zero-shot Text-to-SQL with ChatGPT** `arXiv`

   *Xuemei Dong, Chao Zhang, Yuhang Ge, Yuren Mao, Yunjun Gao, Lu Chen, Jinshu Lin, Dongfang Lou.* [[PDF](https://arxiv.org/abs/2307.07306)] [[Code](https://github.com/bigbigwatermalon/C3SQL)], 2023.

3. **PaLM: Scaling Language Modeling with Pathways** `Journal of Machine Learning Research`

   *Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel.* [[PDF](https://arxiv.org/abs/2204.02311)] [[Code](https://github.com/lucidrains/PaLM-pytorch)], 2023.

4. **Better Zero-Shot Reasoning with Role-Play Prompting** `arxiv`

   *Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, Xiaohang Dong.* [[PDF](https://arxiv.org/abs/2308.07702)] [[Code](https://github.com/NKU-HLT/Role-Play-Prompting)], 2023.



### <img src="..\figure\star.svg" width="25" height="25" />相关应用


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







## 参数高效微调


### <img src="..\figure\star.svg" width="25" height="25" />参数高效微调简介

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



### <img src="..\figure\star.svg" width="25" height="25" />参数附加方法

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



### <img src="..\figure\star.svg" width="25" height="25" />参数选择方法

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



### <img src="..\figure\star.svg" width="25" height="25" />低秩适配方法

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



### <img src="..\figure\star.svg" width="25" height="25" />实践与应用

1. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis.** `SIGMOD`  
   *Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin.* [[PDF](https://arxiv.org/abs/2401.10506)], 2024.

2. **TabLLM: Few-shot Classification of Tabular Data with Large Language Models.** `AISTATS`  
   *Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, David Sontag.* [[PDF](https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf)], 2023.





## 模型编辑

### <img src="..\figure\star.svg" width="25" height="25" />模型编辑简介

1. **Knowledge Editing for Large Language Models: A Survey.** `arXiv`  
   *Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, Jundong Li.* [[PDF](https://arxiv.org/abs/2310.16218)], 2023

2. **A Comprehensive Study of Knowledge Editing for Large Language Models.** `arXiv`  
   *Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen.* [[PDF](https://arxiv.org/abs/2401.01286)][[Code](https://github.com/zjunlp/EasyEdit)], 2024

3. **Editing Large Language Models: Problems, Methods, and Opportunities.** `EMNLP`  
   *Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang.* [[PDF](https://arxiv.org/abs/2305.13172)][[Code](https://github.com/zjunlp/EasyEdit)], 2023

4. **A Survey on Knowledge Editing of Neural Networks.** `arXiv`  
   *Vittorio Mazzia, Alessandro Pedrani, Andrea Caciolai, Kay Rottmann, Davide Bernardi.* [[PDF](https://arxiv.org/abs/2310.19704)], 2023



### <img src="..\figure\star.svg" width="25" height="25" />模型编辑经典方法

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



### <img src="..\figure\star.svg" width="25" height="25" />附加参数法：T-Patcher

1. **Transformer-Patcher: One Mistake Worth One Neuron.** `ICLR`  
   *Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, Zhang Xiong.* [[PDF](https://arxiv.org/pdf/2301.09785)][[Code](https://github.com/ZeroYuHuang/Transformer-Patcher)], 2023



### <img src="..\figure\star.svg" width="25" height="25" />定位编辑法：ROME

1. **Locating and Editing Factual Associations in GPT.** `NeurIPS`  
   *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov.* [[PDF](https://arxiv.org/pdf/2202.05262)][[Code](https://github.com/kmeng01/rome)], 2022

2. **Mass-Editing Memory in a Transformer.** `ICLR`  
   *Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, David Bau.* [[PDF](https://arxiv.org/pdf/2210.07229)][[Code](https://github.com/kmeng01/memit)], 2023



### <img src="..\figure\star.svg" width="25" height="25" />模型编辑应用

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





## 检索增强生成


### <img src="..\figure\star.svg" width="25" height="25" />检索增强生成简介

1. **No free lunch theorems for optimization.** `IEEE Transactions on Evolutionary Computation`   
   *David H. Wolp ert, William G. Macready* [[PDF](https://ieeexplore.ieee.org/document/585893)], 1997

1. **Retrieval-augmented generation for knowledge-intensive nlp tasks.** `NeurIPS`   
   *Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela* [[PDF](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)], 2020



### <img src="..\figure\star.svg" width="25" height="25" />检索增强生成架构

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



### <img src="..\figure\star.svg" width="25" height="25" />知识检索

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



### <img src="..\figure\star.svg" width="25" height="25" />生成增强

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



### <img src="..\figure\star.svg" width="25" height="25" />实践与应用

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



















