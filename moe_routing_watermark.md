# MoE路由水印的信息论形式化定义

## 1. 基础符号与系统定义

### 1.1 MoE路由机制基础

设MoE模型的$l$层为：
$$\text{MoE}_l(x) = \sum_{i=1}^{n} g_i(x) \cdot E_i(x)$$

其中：
- $x \in \mathbb{R}^d$：输入向量
- $g_i(x)$：第$i$个专家的路由权重（由路由器Router产生）
- $E_i(x)$：第$i$个专家的输出
- $n$：专家总数

**路由器输出**：
$$\mathbf{r} = \text{Router}(x) = \sigma(\mathbf{w}^T x + b) \in \mathbb{R}^n$$

其中$\sigma$为softmax函数，得到路由分布$\mathbf{r} = [r_1, r_2, \ldots, r_n]$。

### 1.2 水印系统定义

水印系统$\mathcal{W} = (\mathcal{E}, \mathcal{V})$包括：
- $\mathcal{E}$：嵌入过程（Embedding）
- $\mathcal{V}$：验证过程（Verification）

## 2. 水印信息的编码空间

### 2.1 路由状态空间

对于输入$x$，在第$l$层的路由状态表示为：
$$\mathcal{S}(x) = (\mathbf{r}, \sigma, \tau) \in \mathbb{R}^n \times \mathbb{N} \times \{0,1\}^*$$

其中：
- $\mathbf{r}$：路由分布向量
- $\sigma$：激活的专家集合$\sigma = \{i: r_i > \epsilon_{\text{th}}\}$
- $\tau$：激活序列（按权重排序）$\tau = \text{argsort}(\mathbf{r})$

### 2.2 可用信息维度

**维度1：激活模式** 
$$I_{\text{pattern}} = \log_2 \binom{n}{k}$$
其中$k$为激活专家数量。对$n=128, k=2$：
$$I_{\text{pattern}} \approx 13 \text{ bits}$$

**维度2：排列顺序**
$$I_{\text{order}} = \log_2(k!)$$
对前$k$个专家的排列。对$k=4$：
$$I_{\text{order}} = \log_2(24) \approx 4.6 \text{ bits}$$

**维度3：权重量化**
假设每个路由权重$r_i$量化为$b$-bit精度：
$$I_{\text{weight}} = n \cdot b \text{ bits}$$
对$n=128, b=4$（16级）：
$$I_{\text{weight}} = 512 \text{ bits}$$

**总容量上界**：
$$C_{\max} = I_{\text{pattern}} + I_{\text{order}} + I_{\text{weight}}$$

## 3. 信息容量的量化

### 3.1 可实现容量（Achievable Capacity）

在约束模型性能的条件下，考虑可实现的容量：

$$C_{\text{achievable}} = \max_{\delta \in (0, \epsilon)} I(\mathbf{m}; \mathcal{S}(x) | x)$$

其中：
- $\mathbf{m}$：待嵌入的水印信息
- $\delta$：嵌入强度参数，满足模型性能下降$\leq \epsilon$
- $I(\cdot; \cdot | \cdot)$：条件互信息

**具体形式**：
$$C_{\text{achievable}} = \int_{\mathcal{X}} p(x) \max_{\delta} I(\mathbf{m}; \mathbf{r}_{\delta}(x)) \, dx$$

其中$\mathbf{r}_{\delta}(x)$为被修改的路由分布。

### 3.2 信息-性能权衡曲线

定义权衡函数$\mathcal{T}$：
$$\mathcal{T}(\epsilon) = \{C: \exists \delta \text{ s.t. } \Delta_{\text{perf}} \leq \epsilon \text{ and } I(\mathbf{m}; \mathcal{S}(x)) = C\}$$

其中$\Delta_{\text{perf}} = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{watermarked}}$。

## 4. 鲁棒性的形式化定义

### 4.1 对抗性攻击下的鲁棒性

设攻击为随机变换$\mathcal{A}: \mathcal{S} \to \mathcal{S}'$，如路由器微调或输入扰动。

**鲁棒性定义**：
$$\text{Robustness} = \frac{\sum_{\mathcal{A} \in \mathcal{A}_{\text{adv}}} \mathbb{1}[\mathcal{V}(f_{\text{watermarked}}, \mathcal{A})]}{|\mathcal{A}_{\text{adv}}|}$$

其中$\mathcal{V}(f, \mathcal{A}) = 1$表示在攻击$\mathcal{A}$后仍能成功验证。

### 4.2 鲁棒性的下界（Robustness Bound）

对于独立同分布的攻击，水印的鲁棒性下界由Fano不等式给出：

$$P_{\text{error}} \geq 1 - \frac{I(\mathbf{m}; \mathcal{S}'(x)) + 1}{\log_2 |\mathcal{M}|}$$

其中：
- $\mathcal{S}'(x) = \mathcal{A}(\mathcal{S}(x))$：攻击后的路由状态
- $|\mathcal{M}|$：水印消息空间大小

### 4.3 对特定攻击的鲁棒性量化

**攻击1：路由器重训练**

设路由器参数的扰动为$\Delta \mathbf{w}, \Delta b$，考虑Kullback-Leibler散度：

$$D_{\text{KL}}(\mathbf{r} || \mathbf{r}') = \sum_i r_i \log \frac{r_i}{r_i'} \leq \beta$$

鲁棒性与$\beta$的关系：
$$\text{Rob}_{\text{retrain}} = 1 - \exp(-\lambda D_{\text{KL}})$$

其中$\lambda$为编码的纠错码强度参数。

**攻击2：模型蒸馏**

设蒸馏温度为$T$，学生模型的路由分布为$\mathbf{r}_s$：

$$\text{Rob}_{\text{distill}} = \exp\left(-\frac{D_{\text{KL}}(\mathbf{r}^T || \mathbf{r}_s^T)}{H(\mathbf{r})}\right)$$

其中$H(\mathbf{r})$为路由分布的熵。

**攻击3：量化与剪枝**

对于$q$-bit量化：
$$\text{Rob}_{\text{quant}} = \left(1 - \frac{2^{-q}}{2}\right)^{n \cdot I_{\text{weight}}}$$

## 5. 验证过程与检测能力

### 5.1 假设检验框架

验证问题可设为二元假设检验：
- $H_0$：模型未被水印化（原始模型）
- $H_1$：模型被正确的水印化

给定测试集$\mathcal{T} = \{x_1, \ldots, x_N\}$，提取特征向量：
$$\mathbf{f} = [\mathcal{F}(x_1), \ldots, \mathcal{F}(x_N)]$$

其中$\mathcal{F}$为特征提取函数（如路由激活模式）。

### 5.2 检测功率（Detection Power）

采用Neyman-Pearson引理，最优检测器为：
$$\Lambda(\mathbf{f}) = \frac{p(\mathbf{f}|H_1)}{p(\mathbf{f}|H_0))} \gtrless \tau$$

检测功率定义为：
$$\beta = P(\text{detect} | H_1 \text{ true}) = P(\Lambda > \tau | H_1)$$

**假阳性率**（False Positive Rate）：
$$\alpha = P(\text{detect} | H_0 \text{ true})$$

### 5.3 ROC曲线与AUC

检测质量由接收者操作特征曲线量化：
$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

对于信息论上界，Bhattacharyya距离给出：
$$\text{AUC} \geq \Phi\left(\frac{D_B}{2}\right)$$

其中$\Phi$为标准正态CDF，$D_B$为Bhattacharyya距离：
$$D_B = -\ln \int \sqrt{p(\mathbf{f}|H_0) p(\mathbf{f}|H_1)} \, d\mathbf{f}$$

## 6. 综合性能指标

### 6.1 水印质量函数

综合定义水印系统的质量为三维指标：

$$\mathcal{Q} = (C_{\text{achievable}}, \text{Robustness}, \text{AUC})$$

归一化形式：
$$Q_{\text{normalized}} = \alpha C_n + \beta R_n + \gamma A_n$$

其中$\alpha + \beta + \gamma = 1$为权重，$n$表示归一化到$[0,1]$。

### 6.2 帕累托前沿

考虑多目标优化：
$$\max \{C_{\text{achievable}}, \text{Robustness}, \text{AUC}\}$$
$$\text{s.t.} \quad \Delta_{\text{perf}} \leq \epsilon, \quad \alpha \leq \alpha_{\max}$$

帕累托前沿定义为非被支配的解集合。

## 7. 数学推导示例

### 示例：基于激活模式的容量计算

假设设计仅基于激活模式$\sigma$进行水印：

**信息来源**：从$n$个专家中选择$k$个激活
$$I = \log_2 \binom{n}{k} = \log_2 \frac{n!}{k!(n-k)!}$$

**编码方案**：使用修改路由权重使特定专家组合优先激活
$$\tilde{\mathbf{w}} = \mathbf{w} + \delta \mathbf{b}_{\text{target}}$$

其中$\mathbf{b}_{\text{target}}$为目标激活模式的偏置向量。

**性能约束**：KL散度限制
$$D_{\text{KL}}(\text{Router}_{\text{clean}} || \text{Router}_{\text{watermarked}}) \leq \delta_{\max}$$

**鲁棒性下界**：对路由器重训练
$$P_{\text{detect}} \geq 1 - \exp\left(-\frac{2I}{n}\right)$$

## 8. 开放的理论问题

1. **信息论上界**：MoE路由水印的香农容量是否达到？
2. **鲁棒性-容量权衡**：能否导出率失真理论（Rate-Distortion）框架下的界？
3. **隐蔽性分析**：路由分布的统计隐蔽性如何量化？
4. **联合攻击**：多种攻击同时进行时的鲁棒性？