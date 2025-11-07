先概述论文核心思想与贡献定位，再逐条指出**关键技术问题（Major issues）**与**细节问题（Minor issues）**，最后给出**具体、可落地的修改方案与实验计划**（含步骤、指标、脚本建议），以及一版更严谨的**数学与验证框架改写模板**，帮助你在 2–4 周内把稿子打磨到可投顶会（例如 S&P/USENIX Security/NeurIPS 安全方向或 MLSys 专项）水准。

一、稿件速览与定位（Summary & Positioning）
--------------------------------

* 论文提出在 **MoE（Mixture-of-Experts）** 架构中做**路由水印（Routing Watermark）**的**信息论形式化**：把“参数扰动→输入分布响应→路由状态统计→验证器”视为通信信道；把容量、率失真（Rate–Distortion）与 GLRT/SPRT 检验连起来；并强调对**语义层面释义攻击（Paraphrasing Attack）**的鲁棒性与权衡。
* 优点：把“水印-容量-隐蔽性-任务性能”四维权衡系统化；提出只依赖**激活模式 Σ（组合码）**做验证，避免连续权重与排序在语义变动下的不稳定；给出一个**带约束微调（Fine-tuning）**的编码器实现雏形（含 margin-based 目标）。
* 当前稿件的不足在于：**若干关键假设未闭合**、**统计检验与信息论下界未严格到位**、**工程实现细节与实验验证缺失**、**指标度量与符号一致性不够**。这会影响可采信性与顶会可接受度。

* * *

二、关键技术问题（Major Issues）
----------------------

1) **度量与验证特征的错位（D_detect vs Verification Features）_**_  
   文中建议验证器只使用激活模式 Σ；但定义的统计失真 (D_{\text{detect}}) 却使用完整路由状态分布（含 (r, \Sigma, \pi)）的 KL 散度，存在**度量与验证特征族不一致**的问题。应确保隐蔽性/可检性指标与实际验证器所用特征（Σ 的类型计数）一致，否则理论—实证链条断裂。

2) **GLRT/SPRT 的分布族与似然未显式化**  
   你将观测 (f) 视作激活模式的经验分布（或计数向量），但没有**把 (f) 的似然**明确为**多项分布（Multinomial）**并给出**在 (H_0/H_1) 下的参数化与 MLE**，导致 GLRT 仅停留在“形式”而非“可计算闭式”。应以**类型法（method of types）**和**Sanov 定理**给出阈值与样本复杂度关系。

3) **AUC 下界推导不严谨**  
   你用 Bhattacharyya/Chernoff 上界间接给出 AUC 的保守下界，但**没有明确适用分布族与距公式到 ROC-AUC 的映射步骤**；且文中引用 Pinsker 派生的“检测成功率下界”表达式看起来与标准不等式不匹配（Pinsker 给的是 TV 与 KL 的关系：(\mathrm{TV}\le \sqrt{\tfrac12 D_{\mathrm{KL}}})），需要**重做**推导并**对齐到具体的多项模型**。

4) **容量与鲁棒容量的定义欠缺“语义等价类”刻画**  
   你指出在释义攻击下鲁棒容量可能“骤降为 0”，但缺少**把输入空间划分为语义等价类（paraphrase equivalence classes）**的正式定义与由此导出的**鲁棒容量 (C_{\text{robust}})_**_。建议把**组合码容量 (I**_**{\text{pattern}})**经由“类内不变性约束”缩减为有效码本大小，并结合**常重码（Constant-Weight Codes）**的最小距离给出**码级纠错半径**。

5) **编码器优化目标与任务性能约束未闭环**  
   尽管你在总损失 (L_{\text{total}}=L_{\text{task}}+\lambda_{\text{wm}} L_{\text{wm}}+\lambda_{\text{con}} L_{\text{con}}) 中体现了权衡，但**没有给出**
* 如何选择 paraphrase 对 ((x,x'))，

* 如何设定 margin 与温度 (T)、top‑k、学习率、epoch，

* 如何保证在数据集分布上满足 (\Delta \text{perf}\le \epsilon_1)。  
  这使得编码器仍停留在“思想实验”，缺少落地可复现路径。
6) **参数鲁棒性 (R_{\text{param}}) 的度量与对策缺位_**_  
   你把 (R_{\text{param}}) 作为综合指标之一，但没有**对抗微调/蒸馏/剪枝下的劣化模型**的**攻击模型**与**鲁棒性测试方案**，也没有把它纳入率失真或优化约束中。

7) **符号、假设与叙述的一致性问题**  
   例如：路由状态 (S(x)=(r,\Sigma,\pi)) 的自由度与温度 (T) 噪声 (\sigma) 对“有效容量 (\eta_{T,\sigma})”的影响只“口头说明”，未给出**界或估计式**；组合码容量示例与后文的 n,k 设定不完全对齐；“隐蔽性”既被视为“越小越不易被攻击者发现、越大越易被验证者检测”，但**在目标函数中以单一符号 (D**_**{\text{detect}}) 表示双重含义**，阅读成本高。

* * *

三、细节与写作问题（Minor Issues）
-----------------------

* **引用与相关工作缺失**：目前几乎没有与现有 LLM 水印、MoE 路由稳健性、检验理论（多项 GLRT、Sanov、Chernoff）等文献的系统对比与定位。建议在“Related Work/Background”系统化对齐。
* **术语与中英对照**：建议首现统一“中文为主并括注英文”的风格（你的偏好），例如“**组合码（Combination Code）**”、“**常重码（Constant-Weight Code）**”、“**类型法（Method of Types）**”。
* **图示与表格**：缺少帕累托前沿、能力—失真曲线、GLRT 统计量随样本数的功效曲线等必要图。

* * *

四、可落地的修改方案（2–4 周执行计划）
---------------------

### A. 验证框架落地：把 GLRT/SPRT 变成可计算的“多项型”检验

1. **特征与概率模型显式化**  
   * 观测向量：对测试集 (T={x_i}_{i=1}^N) 只提取激活模式 (\Sigma(x_i))。
   * 计数：对所有 (\binom{n}{k}) 种模式做计数得到 (\mathbf{c}\in \mathbb{N}^{\binom{n}{k}})，总和 (N)。
   * 似然：(\mathbf{c}\sim \text{Multinomial}(N,\mathbf{p}))。在 (H_0) 下 (\mathbf{p}=\mathbf{p}_0)（干净模型路由统计）；在 (H_1) 下 (\mathbf{p}=\mathbf{p}_1)（带水印模型目标分布或其估计）。
2. **GLRT 闭式**  
   [\Lambda(\mathbf{c})=\frac{\max_{\mathbf{p}\in \Theta_1}\Pr(\mathbf{c}|\mathbf{p})}{\max_{\mathbf{p}\in \Theta_0}\Pr(\mathbf{c}|\mathbf{p})}=\frac{\Pr(\mathbf{c}|\hat{\mathbf{p}}_1)}{\Pr(\mathbf{c}|\hat{\mathbf{p}}_0)}]其中 (\hat{\mathbf{p}}_0=\frac{\mathbf{c}}{N})（若 (H_0) 不给先验），(\hat{\mathbf{p}}_1) 采用**签名码本诱导的目标分布**或其 MLE/正则化估计。给出对数似然比的显式表达与阈值 (\tau) 的选取规则。
3. **样本复杂度与阈值**  
   用**Sanov**或**Chernoff**给出 (N) 与错误概率 ((\alpha,\beta)) 的关系：当 (\min\limits_i p_{0,i},p_{1,i}) 非零且两分布分离时，错判率指数受 **Bhattacharyya 距离** (D_B=-\ln\sum_i \sqrt{p_{0,i}p_{1,i}}) 控制；据此给出 ROC–AUC 的保守下界（多项族场景），明确适用条件。
4. **SPRT**  
   在线积累单样本 (\Sigma(x_t)) 的似然比并在两阈值间决策，报告**期望样本数（ASN）**与功效。

### B. 指标对齐与“隐蔽性/可检性”一致化

* 将 (D_{\text{detect}}) 改为**基于 Σ 的散度**： [D_{\text{detect}}=\mathbb{E}_{x\sim p(x)}\big[D_{\mathrm{KL}}(p(\Sigma| \theta_{\text{clean}})\,|\, p(\Sigma| \theta_{\text{wm}}))\big]]或直接用**总变差 TV**/**Hellinger**距离，保证“训练中的隐蔽性度量”与“验证器使用的观测分布”一致。

### C. 语义鲁棒容量与码本设计（组合码 + 纠错思想）

1. **语义等价类**  
   定义等价关系 (x\sim x') 若语义一致；令 (\mathcal{C}) 为类簇集合。鲁棒容量定义为：**对每个类 (\mathcal{C}_j)_**_，(\Pr_{\ x\in \mathcal{C}_j}\big[\Sigma(x)=\Sigma_m\big]\ge 1-\epsilon)。则有效码本大小由满足类内一致性的模式数决定。
2. **常重码（Constant-Weight Codes）**  
   在 (\binom{n}{k}) 的组合空间中选取码本 (\mathcal{M}) 使得模式间**汉明距离**大于 (d_{\min})，结合“top‑k 抖动/温度扰动”把**解码半径**映射为“允许若干位（专家选择）翻转”。这给出**纠错余量**与鲁棒容量的显式下界。

### D. 编码器微调流程落地（可复现）

* **数据与 paraphrase 生成**：  
  选定任务数据集（比如通用指令/问答或分类任务），为每个样本生成 3–5 个释义 (x')（人类改写或规则/模型改写），形成**语义簇**。指标：类内 BLEU/SimCSE 相似度阈值。
* **训练配置**：明确 (n,k,T)，路由器参数范围、学习率、epoch（如 3–10）、(\lambda_{\text{wm}},\lambda_{\text{con}})、margin（随温度缩放），仅微调路由器层及其邻接层，冻结其他参数以减小 (\Delta \text{perf})。
* **损失**：  
  * (L_{\text{wm}})：margin-based，使目标模式所有成员的最低 logit 高于外部最高值 + margin。
  * (L_{\text{con}})：对同类 ((x,x')) 的路由分布 (p(\Sigma|x),p(\Sigma|x')) 施加 KL/TV 距离约束。 训练时逐步增大 (\lambda_{\text{con}})（**curriculum**），观察 (\Delta\text{perf}, R_{\text{input}}, C_{\text{achievable}}) 的权衡曲线。
* **早停与监控**：以验证集上的 (\Delta\text{perf}\le \epsilon_1)、(D_{\text{detect}}) 达阈值为准则，结合 GLRT 的 (\alpha) 控制进行早停。

### E. 系统化实验与图表（必备）

1. **消融（Ablations）**：  
   * 特征族：仅 Σ vs (Σ,π) vs (Σ,π,r)。
   * 超参：(k\in{1,2,4})、(n\in{32,64,128})、温度 (T)、margin。
   * 释义强度：轻微/中等/强释义三档，报告 (R_{\text{input}}) 与 AUC。
2. **鲁棒性测试**：对抗参数攻击（微调、蒸馏、剪枝、量化）评估 (R_{\text{param}})。
3. **图表**：  
   * 三维帕累托：((\Delta \text{perf},D_{\text{detect}},R)) 或 ((\Delta \text{perf},R_{\text{input}},C_{\text{achievable}}))。
   * ROC/PR 与 AUC/ASN 曲线；错误指数与 (N) 的关系。

* * *

五、数学与验证段落的“可直接替换”模板（供你改稿）
-------------------------

> **验证器（仅用 Σ）与 GLRT：**  
> 给定测试集 (T) 与计数向量 (\mathbf{c}\in\mathbb{N}^{\binom{n}{k}})，假设 (\mathbf{c}\sim\text{Multinomial}(N,\mathbf{p}))。我们检验  
> (H_0:\mathbf{p}=\mathbf{p}_0)（干净模型的路由分布） vs (H_1:\mathbf{p}=\mathbf{p}_1)（水印目标分布/估计）。  
> GLRT 的对数似然比为  
> [ \log\Lambda(\mathbf{c})=\sum_i c_i\log\frac{\hat p_{1,i}}{\hat p_{0,i}},\quad \hat{\mathbf{p}}_j=\arg\max_{\mathbf{p}\in\Theta_j}\Pr(\mathbf{c}|\mathbf{p}).] 当 (\Theta_0) 不给定先验时，(\hat{\mathbf{p}}_0=\mathbf{c}/N)，(\hat{\mathbf{p}}_1) 由水印码本诱导的目标分布及正则化（如 Dirichlet）估计获得。据此选取阈值 (\tau) 控制 (\alpha)。
> 
> **错误指数与 AUC 下界：**  
> 若所有分量正且 (\mathbf{p}_0,\mathbf{p}_1) 分离，则  
> [ D_B=-\ln\sum_i\sqrt{p_{0,i}p_{1,i}} ]  
> 给出错判率上界 (P_{\text{error}}\lesssim e^{-D_B})。在此多项分布族下，我们据此构造 ROC 的保守下界，详述适用条件与近似误差。
> 
> **隐蔽性与度量一致：**  
> 训练与评估均以 (\Sigma) 的分布为度量：  
> [ D_{\text{detect}}=\mathbb{E}_{x}\big[D_{\mathrm{KL}}(p(\Sigma| \theta_{\text{clean}})\,|\, p(\Sigma| \theta_{\text{wm}}))\big].] 这样 (D_{\text{detect}}) 与验证器使用的观测族一致。

* * *

六、写作与结构改进
---------

* **贡献点条文化**：把你七点贡献拆分为“理论—算法—验证—鲁棒性—开放问题”五条，并为每条给一句落地结论（如“GLRT 在多项族下闭式可计算，样本复杂度受 (D_B) 控制”）。
* **符号表**：添加表格汇总 (n,k,T,\sigma,\lambda_{\text{wm}},\lambda_{\text{con}},\epsilon_1,\alpha) 等。
* **一致性**：统一把“组合码容量”放到只依赖 (\Sigma) 的一节，避免在另一处再引入 ((r,\pi)) 的容量讨论。

* * *


---------------
