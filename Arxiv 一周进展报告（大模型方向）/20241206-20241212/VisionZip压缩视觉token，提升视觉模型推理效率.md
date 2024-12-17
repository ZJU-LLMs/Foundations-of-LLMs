# VisionZip: Longer is Better but Not Necessary in Vision Language Models

_Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia_

_CUHK HKUST HITSZ_

现有的视觉大模型如LLaVA，依赖大量的视觉 token从图像中提取信息。然而，随着视觉 token 数量的增加，模型的性能并不是线性提升，反而有可能导致冗余和效率低下的问题。因此本文提出了 **VisionZip** 方法。该方法是一种简单有效的视觉token压缩方法，通过选择高信息量的视觉token并减少冗余，提高视觉语言模型的推理效率，在不显著牺牲性能的情况下显著降低计算成本。

## 研究内容

本文研究如何压缩视觉token以减少冗余，实现视觉语言模型的效率提升，同时保持模型性能。

## 研究动机

现有视觉语言模型中视觉token的数量远超文本令牌，存在大量冗余，导致计算成本高，限制了其在实际应用中的发展。

## 技术动机

观察到视觉编码器生成的视觉token中仅有少部分包含高信息量，大部分token关注度低且贡献有限，因此需要通过选择和融合高信息量token来提高模型效率。

## 解决方案

![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/12/13/1734054696996-c87a2141-6599-4461-91e6-d7ad857c5256.png)

**重要token选择**
 分析视觉编码器生成的视觉token，基于注意力权重计算每个token的重要性。通过以下公式计算注意力分数：
$$
S_h = \text{Softmax} \left( \frac{Q_h K_h^\top}{\sqrt{D_h}} \right)
$$
其中，$Q_h$ 和 $K_h$ 分别是查询和键向量，$D_h$ 是注意力头的维度。综合所有头的注意力分数：
$$
S_{\text{avg}} = \frac{1}{H} \sum_{h=1}^H S_h
$$
**剩余token合并**
对剩余token进行相似性计算，使用点积公式评估token间的语义相似性，将相似的token合并，通过加权平均生成上下文token：

**调优**
采用少量数据对跨模态投影器进行微调，冻结其他模型组件，以适应压缩后的视觉token。



## 实验结果

![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2024/12/13/1734054805390-669eb4be-12db-4154-bab0-9116c7fe2648.png)

在实验过程中，VisionZip 被广泛应用于诸多基准测试，比如 LLaVA 和 Video-LLaVA 等。实验得出的结果表明，运用 VisionZip 的模型在多项任务中都有优异表现。具体来说，在 LLaVA-1.5 模型中，即使仅使用 64 个视觉 token，VisionZip 也能取得与使用 576 个 token 相近的性能。而且，VisionZip 还明显提升了推理速度，预填充时间大幅缩短，达到原来的八分之一，充分展现出其在实际应用中的巨大潜力。

---

- 查看 Arxiv 原文请点击"**阅读原文**"[https://arxiv.org/pdf/2412.04467]
- **更多**大模型学习资料，详见浙江大学LLMs Github仓库: 
  https://github.com/ZJU-LLMs/Foundations-of-LLMs
- 本文编辑：徐文溢，毛玉仁