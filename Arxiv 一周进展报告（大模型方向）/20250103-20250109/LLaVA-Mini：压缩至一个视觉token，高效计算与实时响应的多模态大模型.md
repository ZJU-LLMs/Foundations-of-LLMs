# LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token

Shaolei Zhang, Qingkai Fang等

中国科学院智能信息处理重点实验室， 中国科学院计算技术研究所等

本文提出了LLaVA-Mini，通过对多模态大模型注意力矩阵的逐层分析，发现视觉token主要在模型的前几层被利用，基于这一发现，文章引入了模态预融合技术，将视觉信息提前融入文本token，将输入LLM主干的视觉token压缩至一个token。

## 研究内容

多模态大模型的视觉token压缩

## 研究动机



现有方法表现不佳：现有方法依赖于预定义规则来减少视觉编码器输出的token数量，或专注于LLM主干小型化，或者其他方法，仍会导致视觉信息的大量丢失。

## 技术动机



**多模态大模型是如何理解视觉token的？**

通过提出这一疑问，本文对模型进行逐层分析，发现视觉token主要在模型的前几层被利用，随着层级的加深，关注视觉token的注意力急剧减少。

## 解决方案

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2025/01/11/1736578337919-6891e1bf-7390-4da2-90c8-1dd016b40639.png)


基于上面的发现——视觉token在模型的浅层中对融合视觉信息至关重要，LLaVA-Mini在LLM主干网络之前引入了一个模态预融合模块，将视觉信息提前融合到文本token中。下面分别介绍LLaVA-Mini的两个重要模块，视觉token压缩模块和模态预融合模块

**视觉token压缩模块**

LLaVA-Mini 引入了$C \times C$可学习的压缩查询 $Q_v$。这些查询通过交叉注意力与所有视觉token $H_v$进行交互，选择性地提取重要的视觉信息，生成$C \times C$压缩的视觉token $\hat{H}_v \in \mathbb{R}^{C_2 \times d_h}$。为了在压缩过程中保留图像的空间信息，我们对可学习查询和原始视觉token引入了2D正弦位置编码。

**模态预融合模块**

模态预融合模块$f(\cdot)$由 $N_{\text{fusion}}$ 个Transformer块组成，每个Transformer块与LLM骨干网络共享相同的结构和超参数。视觉token $H_v$和文本token $H_q$被连接并输入到预融合模块中，然后提取与文本相关的视觉信息作为融合token，表示为：

$$
\hat{H}_q = f(\text{Concat}(H_v, H_q))[-l_q:]
$$

其中$\hat{H}_q \in \mathbb{R}^{l_q \times d_h}$是包含相关视觉信息的文本表示的融合token。

最终，压缩后的视觉token  $\hat{H}_v$和融合token $\hat{H}_q$（共$C_2 + l_q$个token）一起输入到LLM中，以生成响应。

## 实验结果



本文在图像和视频理解任务上评估LLaVA-Mini，为了公平比较，采用与LLaVA-v1.5相同的配置。分为两个配置LLaVA-Mini-HD-压缩至64个token，LLaVA-Mini-压缩为一个token。实验在11个图像基准和7个视频基准上进行，实验结果分别如下：

![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2025/01/11/1736578368821-315bbd93-5f8e-4376-b356-b21ba71f83da.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img13@main/2025/01/11/1736578387466-c9f48c50-7ced-401d-af96-c15d2dd5c76d.png)


综上，本文推出了LLaVA-Mini，结合模态预融合模块高效压缩视觉token。LLaVA-Mini在图像和视频理解方面表现出色，同时在计算效率、推理延迟和内存使用方面具有优势。


---

  - 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/abs/2410.10630v1]
  - **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
    https://github.com/ZJU-LLMs/Foundations-of-LLMs
  - 本文编辑：胡中豪，毛玉仁