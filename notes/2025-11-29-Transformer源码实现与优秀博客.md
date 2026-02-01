# Transformer源码实现与优秀博客


## 📘 优秀博客

* [Harvard NLP Annotated Transformer（超经典图文讲解 超级推荐）](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  
* [The Illustrated Transformer（可视化讲解，入门必读）](https://jalammar.github.io/illustrated-transformer/)

## 🧩 源码实现

* [Annotated Transformer（Harvard NLP 官方代码）](https://github.com/harvardnlp/annotated-transformer)

* [Tensor2Tensor（Google 早期官方实现）](https://github.com/tensorflow/tensor2tensor)
  
* [Attention Is All You Need - PyTorch 复现](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
  
* [HuggingFace Transformers（主流库，工业级实现）](https://github.com/huggingface/transformers)
  
---

# Transformer 吃透计划


## 1) 全网高质量资料索引（按“吃透顺序”）

1. **原论文：Transformer从0到1**
   Vaswani et al., *Attention Is All You Need*（架构、Scaled Dot-Product Attention、多头、位置编码、mask等都在这里）([arXiv][1])
2. **逐行实现版：把论文变成可运行代码**（强烈推荐）
   Harvard NLP *The Annotated Transformer*：把关键模块拆开、配代码与解释([nlp.seas.harvard.edu][2])
3. **最强可视化直觉：把Q/K/V和多头“看见”**
   Jay Alammar *The Illustrated Transformer*([jalammar.github.io][3])
4. **注意力机制的“前传”：为什么需要attention**

   * Bahdanau additive attention：缓解“固定长度向量瓶颈”、做软对齐([arXiv][4])
   * Luong attention：global/local、不同打分函数体系化总结([arXiv][5])
5. **Transformer里两个“训练稳定性神器”的来源**

   * LayerNorm：不依赖batch统计、训练/推理一致([arXiv][6])
   * Residual（残差思想）：深网络更易优化([arXiv][7])
6. **应用侧里程碑：Encoder-only的代表（理解mask/双向的意义）**
   BERT：深度双向Transformer表示学习([arXiv][8])

---

## 2) 注意力机制：把它“吃透”需要抓住的3个本质

### 本质A：Attention = **内容寻址（content-based addressing）**

给定一个“我现在要找什么”的**Query**，去一堆“可被匹配的索引”**Key**里算相似度，再对对应的“信息内容”**Value**做加权平均。

> 直觉：像在记忆库里用Query去检索Key，检索到的权重再用来汇总Value。

---

### 本质B：Scaled Dot-Product Attention 的数学骨架

Transformer用的是（缩放）点积注意力：
[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\right)V
]
这是论文的核心公式之一([arXiv][1])

**逐项吃透：**

* (QK^\top)：每个query对所有key打分（相似度矩阵，形状通常是 `T_q × T_k`）
* `softmax`：把相似度变成概率分布（注意力权重）
* `mask`：把“不允许看的位置”加到 (-\infty)，softmax后权重≈0（后面细讲）
* 乘 (V)：对value做加权求和，得到“融合上下文”的输出

**为什么要除 (\sqrt{d_k})**：点积随维度增大方差变大，softmax更容易饱和（梯度小），缩放能稳定训练；这也是论文明确写出的设计点([arXiv][1])

---

### 本质C：Additive vs Dot-Product（你会经常在面试/论文里看到）

* **Bahdanau（additive）**：用一个小MLP算相似度，更“表达力强”，早期NMT常用([arXiv][4])
* **Dot-product（multiplicative）**：矩阵乘法更高效、适合并行；Transformer选择它并加了缩放([arXiv][1])
  Harvard那篇也把这两类放在一起对比过([nlp.seas.harvard.edu][2])

---

## 3) Transformer：你要把它当成“可重复堆叠的积木”

### 3.1 一层Encoder长什么样

一层Encoder基本是两块：

1. **Multi-Head Self-Attention**
2. **Position-wise FFN（逐位置前馈网络）**
   每块外面都有 **Residual + LayerNorm（Add & Norm）**([arXiv][1])

**关键点：Self-Attention里 Q、K、V 都来自同一份输入 (X)**（只是乘不同线性变换得到）：
[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
]
（Harvard实现版把这些写得非常清楚）([nlp.seas.harvard.edu][2])

---

### 3.2 Decoder为什么多一块“Cross-Attention”

Decoder每层通常是三块：

1. **Masked Multi-Head Self-Attention**（不能看未来）
2. **Cross-Attention**：**Q来自decoder当前隐状态**，**K/V来自encoder输出**（把源序列信息“取”过来）([arXiv][1])
3. **FFN**

这就是经典encoder-decoder Transformer翻译架构([arXiv][1])

---

## 4) Multi-Head Attention：不只是“多做几次attention”

多头注意力做的事是：把表示维度切成多个子空间，在不同子空间里分别做attention，再拼起来：

[
\text{head}_i=\mathrm{Attention}(XW_Q^i, XW_K^i, XW_V^i)
]
[
\mathrm{MultiHead}(X)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
]
这是Transformer表达力的关键之一([arXiv][1])

**你真正要吃透的直觉：**

* 单头：只能学到一种“相关性度量/聚合方式”
* 多头：能并行学多种关系（语法依赖、指代、主题一致性…），最后融合

---

## 5) Mask：Transformer里最容易“看懂但写错”的地方

你至少要区分两种mask：

1. **Padding mask（填充mask）**
   把padding位置屏蔽掉，否则模型会把“PAD”也当成信息。Harvard实现里有很清晰的mask构造方式([nlp.seas.harvard.edu][2])

2. **Causal / Subsequent mask（因果mask）**
   用在Decoder self-attention：位置t只能看 (\le t) 的token，保证自回归生成的因果性([arXiv][1])

---

## 6) 位置编码：没有RNN以后，“顺序”从哪来？

Transformer没有循环结构，所以必须显式注入位置信息。原论文用的是**正弦/余弦位置编码**（也可学习位置embedding）([arXiv][1])

你要抓住的要点：

* **Self-attention本身是置换不变的**（打乱token顺序，注意力计算形式不变），所以必须加位置
* 位置编码等价于告诉模型“第几个token”，让注意力能表达相对/绝对顺序关系

---

## 7) 你想“完全吃透”：给你一套最有效的训练方式（不靠死记）

### 7.1 6个必须能手写/口述的“检查点”

1. 写出并解释 attention 公式（含mask）([arXiv][1])
2. 说清 self-attn vs cross-attn 的Q/K/V来源([nlp.seas.harvard.edu][2])
3. 说清 multi-head 的“为什么不是多此一举”([NeurIPS Proceedings][9])
4. 说清两种mask各自解决什么问题([nlp.seas.harvard.edu][2])
5. 说清 residual + layernorm 为什么能稳定训练([nlp.seas.harvard.edu][2])
6. 说清为什么需要位置编码、原论文怎么做([arXiv][1])

### 7.2 3个“做完就会了”的实战任务

* **任务1：从零实现Scaled Dot-Product Attention（带mask）**
  输入随机Q/K/V，检查：mask位置权重≈0；softmax每行和为1。参考Harvard逐行实现([nlp.seas.harvard.edu][2])
* **任务2：实现Multi-Head Attention并做形状自检**
  强制自己写出每一步张量形状（`B,T,d_model` → `B,h,T,d_k` 等），最能治“看懂但写不出来”。([nlp.seas.harvard.edu][2])
* **任务3：跑一个最小Transformer**
  用极小数据（copy task / reverse task）训练到过拟合；这会让你真正理解mask、位置、decoder输入右移等细节。Harvard那篇就有训练脚手架思路([nlp.seas.harvard.edu][2])

---

## 8) Transformer家族一眼看懂：你在学的到底是哪一种？

* **Encoder-Decoder**：机器翻译/Seq2Seq（原论文）([arXiv][1])
* **Encoder-only**：理解型任务（BERT是代表）([arXiv][8])
* **Decoder-only**：自回归生成（很多LLM属于这类；核心是causal mask）([arXiv][1])

---
[1]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"
[2]: https://nlp.seas.harvard.edu/annotated-transformer/?utm_source=chatgpt.com "The Annotated Transformer - Harvard University"
[3]: https://jalammar.github.io/illustrated-transformer/?utm_source=chatgpt.com "The Illustrated Transformer – Jay Alammar – Visualizing machine ..."
[4]: https://arxiv.org/abs/1409.0473?utm_source=chatgpt.com "Neural Machine Translation by Jointly Learning to Align and Translate"
[5]: https://arxiv.org/abs/1508.04025?utm_source=chatgpt.com "Effective Approaches to Attention-based Neural Machine Translation"
[6]: https://arxiv.org/abs/1607.06450?utm_source=chatgpt.com "Layer Normalization"
[7]: https://arxiv.org/abs/1512.03385?utm_source=chatgpt.com "Deep Residual Learning for Image Recognition"
[8]: https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
[9]: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf?utm_source=chatgpt.com "Attention Is All You Need - NeurIPS"


---

# Transformer 相关优秀视频讲解

下面是我帮你“全网深度搜”出来、**真讲透注意力机制 + Transformer** 的高质量视频清单（中英都有），并给你一条**最省时间的观看顺序**。每个条目后面都带可点的来源。

---

## 最推荐的观看顺序（照这个看，最容易“吃透”）

1. **先建立直觉（看懂 Q/K/V、softmax、multi-head 到底在干嘛）**

* 3Blue1Brown：*Attention in transformers, step-by-step*（把注意力矩阵怎么来的讲得极清楚，强烈推荐先看） ([YouTube][1])
* MIT 6.S191（2025版）：*RNNs, Transformers, and Attention*（从序列建模痛点 → attention → transformer，体系完整） ([YouTube][2])

2. **再把“标准Transformer”系统化（结构、mask、位置编码、训练细节）**

* Stanford CS224N（Lecture 8）：*Self-Attention and Transformers*（经典课，推导+结构讲得很正） ([YouTube][3])
* Stanford CS224N（Lecture 7）：*Attention*（专门讲attention，适合补齐基础与细节） ([YouTube][4])

3. **最后用“论文带读 + 代码实战”完成闭环（真正吃透的关键）**

* 李沐（B站）：*Transformer论文逐段精读*（按论文逐段讲，信息密度极高） ([哔哩哔哩][5])
* Andrej Karpathy：*Let’s build GPT from scratch*（从零写GPT，mask、embedding、训练循环全打通） ([YouTube][6])

---

## 中文向：非常适合系统吃透（入门→进阶）

* 李宏毅（B站相关合集）：*自注意力机制与Transformer详解*（更友好、更适合入门建立框架） ([哔哩哔哩][7])
* 动手学深度学习v2（李沐）：*68 Transformer*（配合实现与工程细节，适合“会写出来”） ([YouTube][8])
* 李沐（B站）：*Transformer论文逐段精读*（进阶必看，吃透“为什么这么设计”） ([哔哩哔哩][5])

---

## 英文向：大学课程级别（深度 + 视野）

* Stanford CS25（Transformers United，整套playlist）：覆盖NLP/CV/RL/对齐/检索增强/专家混合等（看完视野会非常开） ([YouTube][9])
* Stanford CS25（含Karpathy的Intro to Transformers）：讲Transformer直觉与演化脉络很强 ([YouTube][9])
* Stanford CME295（2025）：*Transformers & LLMs*（更贴近“LLM时代”的课堂结构，含attention与transformer总览） ([YouTube][10])
* fast.ai（Lesson 24：Attention & Transformers）：偏“代码直觉+应用”，适合补不同视角 ([Practical Deep Learning for Coders][11])

---

## 如果你只想挑“3套”就吃透（最省时间版）

1. 3Blue1Brown：把注意力机制真正看懂 ([YouTube][1])
2. CS224N Lecture 8：系统掌握Transformer标准结构与要点 ([YouTube][3])
3. Karpathy：从零写GPT，把mask/训练/推理细节全部落地 ([YouTube][6])

---

## 你“吃透”的自测标准（看完要能做到这些）

* 能手写/口述：Scaled Dot-Product Attention、为什么要除 √d、为什么multi-head有效
* 能说清：**padding mask vs causal mask** 各解决什么问题
* 能画出一层：Encoder层 / Decoder层，并说明 self-attn / cross-attn 的 Q/K/V 分别来自哪里
* 能自己实现一个最小Transformer/GPT（哪怕小数据过拟合）

---

[1]: https://www.youtube.com/watch?v=eMlx5fFNoYc&utm_source=chatgpt.com "Attention in transformers, step-by-step | Deep Learning Chapter 6"
[2]: https://www.youtube.com/watch?v=GvezxUdLrEk&utm_source=chatgpt.com "MIT 6.S191: Recurrent Neural Networks, Transformers, and Attention"
[3]: https://www.youtube.com/watch?v=LWMzyfvuehA&utm_source=chatgpt.com "Stanford CS224N NLP with Deep Learning | 2023 | Lecture 8 - YouTube"
[4]: https://www.youtube.com/watch?v=J7ruSOIzhrE&utm_source=chatgpt.com "Stanford CS224N: NLP w/ DL | Spring 2024 | Lecture 7 - YouTube"
[5]: https://www.bilibili.com/opus/586487984822265072?utm_source=chatgpt.com "Transformer论文逐段精读【论文精读】 - 哔哩哔哩"
[6]: https://www.youtube.com/watch?v=kCc8FmEb1nY&utm_source=chatgpt.com "Let's build GPT: from scratch, in code, spelled out. - YouTube"
[7]: https://www.bilibili.com/list/3537113337301588?bvid=BV1r8nMz4EAj&oid=115269171155173&utm_source=chatgpt.com "李宏毅 | 自注意力机制和Transformer详细解析，目前最合适 ..."
[8]: https://www.youtube.com/watch?v=oIH17WP8Wko&utm_source=chatgpt.com "68 Transformer【动手学深度学习v2】 - YouTube"
[9]: https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&utm_source=chatgpt.com "Stanford CS25 - Transformers United - YouTube"
[10]: https://www.youtube.com/watch?v=Ub3GoFaUcds&utm_source=chatgpt.com "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 1 ... - YouTube"
[11]: https://course.fast.ai/Lessons/lesson24.html?utm_source=chatgpt.com "24: Attention & transformers - Practical Deep Learning for Coders"


---

# Transformer 相关综述


下面按“**最全总览** → **高效/长上下文** → **视觉/多模态** → **压缩与部署**”给你一组**口碑+覆盖面+信息密度**都很强的 Transformer/变种综述（基本都能当文献入口用）。

## 必读：Transformer 变种全景总览（最接近你要的“最全”）

1. **A Survey of Transformers (arXiv:2106.04554)**
   给了比较系统的 **X-formers 分类法**，从**结构改造、预训练、应用**三个视角把大量变种串起来，适合作为“总目录”。 ([arXiv][1])

2. **Efficient Transformers: A Survey (arXiv:2009.06732)**
   专攻“**为什么/怎么把 Transformer 做高效**”：稀疏、低秩、线性注意力、长序列等经典路线都有梳理，适合你要吃透各种结构 trick 的“脉络图”。 ([arXiv][2])

3. **A Historical Survey of Advances in Transformer Architectures (Applied Sciences, 2024)**
   偏“**历史演进视角**”：从早期 Transformer 到 LLM/ViT 时代的关键分叉与代表作，适合把发展时间线捋顺。 ([MDPI][3])

---

## 注意力机制 & 长序列：把“attention 这坨”吃透的综述入口

4. **Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention (PDF)**
   很硬核的“**注意力优化大全**”，把方法按 **硬件友好 / 稀疏 / KV 压缩 / 线性注意力**做统一 taxonomy，还配统一分析框架（想把 attention 研究线索一次性掌握，这篇很顶）。 

5. **Advancing Transformer Architecture in Long-Context LLMs: A Comprehensive Survey (arXiv:2311.12351)**
   专门讲 **长上下文**：从预训练到推理阶段的架构升级、评测数据集/指标、工具链等，适合“长上下文能力”这条主线深挖。 ([arXiv][4])

---

## 视觉 Transformer 变种：CV 方向最权威的入口之一

6. **Transformers in Vision: A Survey (arXiv:2101.01169)**
   覆盖很广：分类/检测/分割、生成、多模态、视频、低层视觉、3D 等，适合把 ViT 系列与 CV 任务脉络一次连起来。 ([arXiv][5])

---

## 压缩、推理与落地：想把“变种”理解到工程层必读

7. **A Survey on Transformer Compression (arXiv:2402.05964)**
   Transformer 专属的压缩视角：剪枝、量化、蒸馏、结构高效化等，并且覆盖 LLM/LVM 场景。 ([arXiv][6])

8. **A Survey of Techniques for Optimizing Transformer Inference (arXiv:2307.07982)**
   更偏“推理阶段全栈优化”：算法层（蒸馏/剪枝/量化/NAS…）到硬件与加速器设计都有整理，适合工程落地。 ([arXiv][7])

9. **From Transformers to LLMs: A Systematic Survey of Efficiency Considerations in NLP (arXiv:2406.16893, v2 2025)**
   NLP/LLM 侧的“效率系统综述”，覆盖到 2011–2025 的大量工作，还把 **微调、提示工程、RAG** 的效率问题也纳入讨论。 ([arXiv][8])

10. **A Survey on LLM Acceleration based on KV Cache Management (arXiv:2412.19442, TMLR 2025)**
    长上下文推理的核心瓶颈之一：**KV cache**。这篇把 token/model/system 三层策略做了清晰分类，还给了基准与论文清单仓库。 ([arXiv][9])

---

## 如果你想“最快吃透”，建议阅读顺序

* **先读总览**：#1（全景 taxonomy）→ #2（高效 X-formers）
* **再把注意力搞透**：#4（attention 优化大全）
* **按你方向选分支**：CV 就 #6；长上下文就 #5 + #10；工程部署就 #7 + #8；NLP 效率与RAG就 #9

如果你告诉我你更关注：**LLM（长上下文/推理）** 还是 **CV（ViT 体系）**，我可以把上面这几篇再细化成“每篇该看哪些章节、配套该补哪些经典原论文”的精读路线。

[1]: https://arxiv.org/abs/2106.04554 "[2106.04554] A Survey of Transformers"
[2]: https://arxiv.org/abs/2009.06732 "[2009.06732] Efficient Transformers: A Survey"
[3]: https://www.mdpi.com/2076-3417/14/10/4316 "A Historical Survey of Advances in Transformer Architectures | MDPI"
[4]: https://arxiv.org/abs/2311.12351 "[2311.12351] Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey"
[5]: https://arxiv.org/abs/2101.01169 "[2101.01169] Transformers in Vision: A Survey"
[6]: https://arxiv.org/abs/2402.05964 "[2402.05964] A Survey on Transformer Compression"
[7]: https://arxiv.org/abs/2307.07982 "[2307.07982] A Survey of Techniques for Optimizing Transformer Inference"
[8]: https://arxiv.org/abs/2406.16893 "[2406.16893] From Transformers to LLMs: A Systematic Survey of Efficiency Considerations in NLP"
[9]: https://arxiv.org/abs/2412.19442 "[2412.19442] A Survey on Large Language Model Acceleration based on KV Cache Management"

