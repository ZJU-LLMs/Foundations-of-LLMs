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

9. **Gpt-4 technical report.** `Online`  
    *OpenAI.* [[PDF](https://openai.com/index/hello-gpt-4o)], 2024.

10. **Gpt-4 technical report.** `Online`  
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
