<h1 align="center">大模型基础</h1>


<div align="center"> 
  <img src=".\figure\cover.png" style="width: 50%">
</div>

<p align="center">
  <img src="https://img.shields.io/github/stars/ZJU-LLMs/Foundations-of-LLMs?style=social">
  <img src="https://img.shields.io/github/forks/ZJU-LLMs/Foundations-of-LLMs?style=social">
<!--   <img src="https://img.shields.io/github/license/ZJU-LLMs/Foundations-of-LLMs"> -->
</p>

本书旨在为对大语言模型感兴趣的读者系统地讲解相关基础知识、介绍前沿技术。作者团队将认真听取开源社区以及广大专家学者的建议，持续进行**月度更新**，致力打造**易读、严谨、有深度**的大模型教材。并且，本书还将针对每章内容配备相关的**Paper List**，以跟踪相关技术的**最新进展**。

本书第一版包括**传统语言模型**、**大语言模型架构演化**、**Prompt工程**、**参数高效微调**、**模型编辑**、**检索增强生成**等六章内容。为增加本书的易读性，每章分别以**一种动物**为背景，对具体技术进行举例说明，故此本书以六种动物作为封面。当前版本所含内容均来源于作者团队对相关方向的探索与理解，如有谬误，恳请大家多提issue，多多赐教。后续，作者团队还将继续探索大模型推理加速、大模型智能体等方向。相关内容也将陆续补充到本书的后续版本中，期待封面上的动物越来越多。

当前完整的本书PDF版本路径为<a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大模型基础.pdf">大模型基础.pdf</a>。另外，我妈还提供了两个文件夹，<a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容">大语言模型分章节内容</a>文件夹中包含了各章节的PDF版本。而<a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型相关论文">大语言模型相关论文</a>文件夹中包含了各章节的相关论文，当前正处于不断更新中。

其中每个章节的内容目录如下表所示。

## 本书目录

<table>
    <tr>
        <th style="text-align:center; width: 25%;">章节</th>
        <th style="text-align:center; width: 75%;" colspan="3">所含内容</th>
    </tr>
    <tr>
        <td rowspan="2"><b><a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容/第1章%20语言模型基础.pdf">第 1 章：语言模型基础</a></b></td>
        <td style="width: 25%;">1.1 基于统计方法的语言模型</td>
        <td style="width: 25%;">1.2 基于 RNN 的语言模型</td>
        <td style="width: 25%;">1.3 基于 Transformer 的语言模型</td>
    </tr>
    <tr>
        <td>1.4 语言模型的采样方法</td>
        <td>1.5 语言模型的评测</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><b><a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容/第2章%20大语言模型.pdf">第 2 章：大语言模型</a></b></td>
        <td>2.1 大数据 + 大模型 → 新智能</td>
        <td>2.2 大语言模型架构概览</td>
        <td>2.3 基于 Encoder-only 架构的大语言模型</td>
    </tr>
    <tr>
        <td>2.4 基于 Encoder-Decoder 架构的大语言模型</td>
        <td>2.5 基于 Decoder-only 架构的大语言模型</td>
        <td>2.6 非 Transformer 架构</td>
    </tr>
    <tr>
        <td rowspan="2"><b><a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容/第3章%20Prompt工程.pdf">第 3 章：Prompt 工程</a></b></td>
        <td>3.1 Prompt 工程简介</td>
        <td>3.2 上下文学习</td>
        <td>3.3 思维链</td>
    </tr>
    <tr>
        <td>3.4 Prompt 技巧</td>
        <td>3.5 相关应用</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><b><a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容/第4章%20参数高效微调.pdf">第 4 章：参数高效微调</a></b></td>
        <td>4.1 参数高效微调简介</td>
        <td>4.2 参数附加方法</td>
        <td>4.3 参数选择方法</td>
    </tr>
    <tr>
        <td>4.4 低秩适配方法</td>
        <td>4.5 实践与应用</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><b><a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容/第5章%20模型编辑.pdf">第 5 章：模型编辑</a></b></td>
        <td>5.1 模型编辑简介</td>
        <td>5.2 模型编辑经典方法</td>
        <td>5.3 附加参数法：T-Patcher</td>
    </tr>
    <tr>
        <td>5.4 定位编辑法：ROME</td>
        <td>5.5 模型编辑应用</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><b><a href="https://github.com/ZJU-LLMs/Foundations-of-LLMs/tree/main/大语言模型分章节内容/第6章%20检索增强生成.pdf">第 6 章：检索增强生成</a></b></td>
        <td>6.1 检索增强生成简介</td>
        <td>6.2 检索增强生成架构</td>
        <td>6.3 知识检索</td>
    </tr>
    <tr>
        <td>6.4 生成增强</td>
        <td>6.5 实践与应用</td>
        <td></td>
    </tr>
</table>





## 致谢

本书的不断优化，将仰仗各位读者的帮助与支持。您的建议将成为我们持续向前的动力！

所有提出issue的人，我们都列举在此，以表达我们深深的谢意。




如果有此书相关的其他问题，请随时联系我们，可发送邮件至：xuwenyi@zju.edu.cn。
