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
