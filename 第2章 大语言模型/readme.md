# 大语言模型

- [大语言模型](#大语言模型)
  - [大数据+大模型→新智能](#大数据大模型新智能)
  - [大语言模型架构概览](#大语言模型架构概览)
  - [基于 Encoder-only 架构的大语言模型](#基于-encoder-only-架构的大语言模型)
  - [基于 Encoder-Decoder 架构的大语言模型](#基于-encoder-decoder-架构的大语言模型)
  - [基于 Decoder-only 架构的大语言模型](#基于-decoder-only-架构的大语言模型)
  - [非 Transformer 架构](#非-transformer-架构)


## <img src="../figure/star.svg" width="25" height="25" />大数据+大模型→新智能
1.  **Scaling laws for neural language models.** `arXiv`    
    *Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei.* [[PDF](https://arxiv.org/pdf/2001.08361)], 2020.

2.  **Training Compute-Optimal Large Language Models** `arXiv`    
    *Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre.* [[PDF](https://arxiv.org/pdf/2203.15556)], 2022.

3.  **PaLM 2 Technical Report.** `arXiv`    
    *Google.* [[PDF](https://arxiv.org/pdf/2305.10403)], 2023.



## <img src="../figure/star.svg" width="25" height="25" />大语言模型架构概览
1.  **Attention is all you need.** `NeurIPS`  
    *Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia.* [[PDF](https://arxiv.org/pdf/1706.03762)], 2017.

## <img src="../figure/star.svg" width="25" height="25" />基于 Encoder-only 架构的大语言模型
1.  **A survey on contextual embeddings.** `arXiv`    
    *Qi Liu, Matt J. Kusner, Phil Blunsom.* [[PDF](https://arxiv.org/pdf/2003.07278)], 2020.

2.  **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** `NAACL`
    *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.* [[PDF](https://aclanthology.org/N19-1423.pdf)][[Code](https://github.com/google-research/bert)], 2018.
    
4.  **RoBERTa: A Robustly Optimized BERT Pretraining Approach.** `arXiv`
    *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.* [[PDF](https://arxiv.org/abs/1907.11692)][[Code](https://github.com/pytorch/fairseq)], 2019.
    
5.  **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.** `arXiv`
    *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.* [[PDF](https://arxiv.org/pdf/1909.11942)][[Code](https://github.com/google-research/ALBERT)], 2019.
    
6.  **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.** `arXiv`
    *Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.* [[PDF](https://arxiv.org/pdf/2003.10555)][[Code](https://github.com/google-research/electra)], 2020.


## <img src="../figure/star.svg" width="25" height="25" />基于 Encoder-Decoder 架构的大语言模型
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


## <img src="../figure/star.svg" width="25" height="25" />基于 Decoder-only 架构的大语言模型
1.  


## <img src="../figure/star.svg" width="25" height="25" />非 Transformer 架构
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
