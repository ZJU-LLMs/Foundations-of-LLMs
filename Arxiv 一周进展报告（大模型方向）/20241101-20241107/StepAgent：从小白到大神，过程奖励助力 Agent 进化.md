

#  From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning

**作者**：*Zhirui Deng, Zhicheng Dou* 等

**单位**：*Gaoling School of Artificial Intelligence, Renmin University of China* 等


![](https://fastly.jsdelivr.net/gh/bucketio/img8@main/2024/11/08/1731080788479-be08c132-b269-43a5-a03d-7638d450ebcb.png)



本文研究的是**通过逐步强化学习优化 Agent 系统的策略模型**。现有的 LLM as Agent 方法主要依赖于LLM的固有知识，或者使用强化学习策略来增强 Agent 解决复杂交互任务的能力。然而，这些方法受到稀疏奖励问题的限制，即现有数据集仅提供每个多步推理链的最终标量奖励，可能导致策略学习的效果和效率低下。为了解决这一问题，本文提出了 **StepAgent** 优化框架，利用逐步奖励来优化 Agent 的强化学习过程。

StepAgent 的核心思想是模拟新手到专家的学习过程，通过自动构建监督信号来优化 Agent 策略。具体而言，该框架分为两个阶段：**检查（Inspection）** 和 **反思（Reflection）**。在检查阶段， Agent 观察专家行为并进行模仿练习，以识别自身与专家之间的能力差距；在反思阶段， Agent 根据练习结果，通过两种策略（隐式奖励强化学习和逆强化学习）生成步骤奖励，再使用 PPO 算法改进策略。

#### 1. 检查阶段：生成步骤级对比数据

在 LLM  Agent 任务中， Agent 需与环境交互并多次试错才能得出最终推理结果，传统监督微调方法观察和模仿专家的完整轨迹，并根据最终的环境奖励或人工标注信号进行优化。一方面，模拟完整轨迹需不断与环境交互，该过程顺序进行且无法并行化，需要大量**计算时间和资源**；另一方面，Agent 同时理解所有专家行为易造成**信息过载**，难以消化和掌握每个行为的细节，导致学习过程低效。

为克服这些问题，本文让 Agent 逐步骤观察专家行为并进行练习，根据轨迹片段生成步骤级的对比数据。

对于包含 $n$ 步的专家轨迹 $t_{e}=(\hat{o}_{1}, \hat{a}_{1}, \cdots, \hat{o}_{n}, \hat{a}_{n})$ ，在每个动作之后**分割轨迹**，产生**专家轨迹片段**，将每个动作视为 Agent 学习目标，即 

$$
(\hat{o}_{1}, \hat{a}_{1}, \cdots, \hat{o}_{i}, \hat{a}_{i}) \in \mathcal{T}_{sample }, i = 1,2, \cdots, n
$$
Agent 确定学习目标后进入练习阶段，基于专家轨迹片段中的状态生成动作。具体来说，对于 $\mathcal{T}_{sample}$ 中的每个轨迹片段，将当前动作之前的序列作为状态 $\hat{s}_{i}=(\hat{o}_{1}, \hat{a}_{1}, \cdots, \hat{o}_{i})$ ， 组成 Prompt 让 Agent $\pi_{\theta}$ 生成相应动作 $a_{i}^{\theta} \sim \pi_{\theta}(a | s)$ ，得到 Agent 在每个轨迹片段的决策动作，构成 Agent 的轨迹片段数据：

$$
(\hat{o}_{1}, \hat{a}_{1}, \cdots, \hat{o}_{i}, a_{i}^{\theta}) \in T_{sample}^{\theta}, i = 1,2, \cdots, n
$$
#### 2. 反思阶段：生成步骤级奖励信号

反思阶段利用**专家轨迹片段数据**和 **Agent 轨迹片段数据**，自动生成步骤级的奖励信号，用来直到强化学习算法更新策略模型。本文提出了两种方法来产生步骤奖励，分别为**隐式奖励** 和**逆强化学习**。

**2. 1 隐式奖励**

利用 DPO 损失 $L_{implicit}(\pi_{\theta}, \pi_{e})$ 优化 Agent 策略： 

$$
 L_{implicit}(\pi_{\theta}, \pi_{e}) = -\mathbb{E}\left[log \sigma\left(\beta log \frac{\pi_{\theta}\left(\hat{a}_{i} | \hat{s}_{i}\right)}{\pi_{e}\left(\hat{a}_{i} | \hat{s}_{i}\right)}-\beta log \frac{\pi_{\theta}\left(a_{i}^{\theta} | \hat{s}_{i}\right)}{\pi_{e}\left(a_{i}^{\theta} | \hat{s}_{i}\right)}\right)\right]
$$
通过优化该损失， Agent 策略可逐渐接近专家策略。

**2.2 逆强化学习**

训练**判别器网络**区分专家和 Agent 策略与环境交互产生的状态-行动对的**数据分布差异**，以此作为奖励信号优化 Agent 策略。

1.定义策略 $\pi$ 的**占用度量** $\rho_{\pi}$ ，表示 Agent 采用策略 $\pi$ 与环境交互过程中产生的**状态-行动对**的归一化分布： 

$$
\rho_{\pi}(s, a)=(1 - \gamma) \sum_{t = 0}^{\infty} \gamma^{t} P_{\pi}\left(s_{t}=s\right) \pi(a | s)
$$
其中 $1 - \gamma$ 是归一化因子， $P_{\pi}(s_{t}=s)$ 表示 Agent 在时间 $t$ 处于状态 $s$ 的概率。

2.为准确模仿专家策略，让 Agent 的占用度量 $\rho_{\pi_{\theta}}$ 接近专家的 $\rho_{\pi_{e}}$ ，采用 **Jensen-Shannon散度（JS）** 衡量两分布距离，优化目标为： 

$$
\min _{\pi} JS\left(\rho_{\pi_{\theta}}, \rho_{\pi_{e}}\right)-\lambda H\left(\pi_{\theta}\right)
$$
   其中 $\lambda$ 是超参数， $H(\pi_{\theta}) \triangleq \mathbb{E}_{\pi_{\theta}}[-log \pi_{\theta}(a | s)]$ 是 Agent 策略的 $\gamma$  - 折扣因果熵。 

3.根据 GAIL，Jensen - Shannon散度 $JS(\rho_{\pi_{\theta}}, \rho_{\pi_{e}})$ 可由凸成本函数正则化项 $\omega(\rho_{\pi_{\theta}}-\rho_{\pi_{e}})$ 表示（在常数偏移和缩放范围内），凸成本函数正则化项 $\omega: \mathbb{R}^{S \times A} \to \mathbb{R} \cup{\{\infty\}}$ 定义为：

$$
\omega(c) \triangleq \begin{cases}\mathbb{E}_{\pi_{c}}\left[-c(s, a)-log \left(1 - e^{c(s, a)}\right)\right] & c < 0 \\ +\infty & c \geq 0\end{cases}
$$

上述正则化项 $\omega(\rho_{\pi_{\theta}}-\rho_{\pi_{e}})$ 的最优解表示为：  

$$
 \sup _{D \in(0,1)^{S \times A}} \mathbb{E}_{\pi_{\theta}}[log (D(s, a))]+\mathbb{E}_{\pi_{e}}[log (1 - D(s, a))]
$$

因此，优化问题可转化为寻找下式的鞍点 $(\pi, D)$  :

$$
\mathbb{E}_{\pi_{\theta}}[log (D(s, a))]+\mathbb{E}_{\pi_{e}}[log (1 - D(s, a))]-\lambda H\left(\pi_{\theta}\right)
$$

4.使用从专家和 Agent 轨迹中采样的数据训练**判别器网络** $D: S \times A \to (0,1)$ ，其主要目标是**区分 Agent 策略 $\pi_{\theta}$ 和专家策略 $\pi_{e}$ 生成的数据分布**。当判别器无法区分时， Agent 的占用度量成功匹配专家。**判别器网络 $D$ 可作为隐式奖励模型为 Agent 策略提供逐步的奖励信号。**

最后，使用生成的过程奖励，指导 PPO 算法对 Agent 策略进行更新。

#### 3. 实验

在 Web 任务、Agent 任务以及 Question - Answering 任务中，StepAgent 的两种变体（Implicit 和 Inverse）在各项评估指标上均表现出色，超越了所有基线方法。


![](https://fastly.jsdelivr.net/gh/bucketio/img6@main/2024/11/08/1731080466207-212c74de-7a7c-4089-a75a-35795be2774d.png)

综上所述，StepAgent框架通过观察阶段和反思阶段的逐步监督学习，利用步骤过程奖励有效地改进了LLM代理的策略训练过程。

---

- 查看 Arxiv 原文请点击"**阅读原文**" [https://arxiv.org/abs/2411.03817]
- **更多**文章请详见 Github 仓库: 
  **https://github.com/ZJU-LLMs/Foundations-of-LLMs**
- 本文编辑：樊怡江，毛玉仁