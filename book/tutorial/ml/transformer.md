![Transformer图像](https://pic2.zhimg.com/v2-6a13fc0ffc0b1e48c60782ec43ff4d18_1440w.jpg?source=172ae18b)

<center><h1>Transformer</h1></center>

#### 常见问题
1. 为什么使用“多头”注意力机制？
1. 为什么进行缩放$$\sqrt{d_k}$$?
1. 官方Transformer的multi-head attention有多少个“头”？q、k、v各有多少维？
1. Transformer的layer是由什么组成的？

---

Transformer完全没有使用循环网络结构，只使用了self-attention模块完成了seq2seq任务，这使得整个模型可以并行计算，不再受限于LSTM/GRU等模型的时序性质。

## Multi-Head Attention
> Scaled Dot Product Attention ，将Query和Key-Value映射到输出

<center><img src="https://pic1.zhimg.com/80/v2-e551f16cc7511f55151d152d28e2aab8_720w.jpg" width = "25%" height = "30%"/></center>

$$
Attention \left ( Q,K,V\right )=Softmax(\frac{QK^T}{\sqrt{d_k}})V \\
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V) \\
MultiHead(Q,K,V)=Concat(head_1,..,head_h)W^O \\
W_i^Q\in \mathbb{R}^{d_{model} \times d_k},W_i^K\in \mathbb{R}^{d_{model} \times d_k},W_i^V\in \mathbb{R}^{d_{model} \times d_v},W_i^O\in \mathbb{R}^{h\cdot d_v \times d_{model}}
$$

    
```python
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    
def attention(query, key, value, mask=None, dropout=None):
    """
    输入形状: B*h*N*d_k | B*h*N*d_k | B*h*N*d_v | 1*1*N*N
    Batch、head是独立的，运算时为了方便思考可以不考虑
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # B*h*N*N
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # B*h*N*N 在 0/1轴广播mask
    p_attn = F.softmax(scores, dim = -1) # Softmax不改变形状  B*h*N*N
    if dropout is not None:
        p_attn = dropout(p_attn) # dropout的概率结果归零，既不关注某个位置的单词  B*h*N*N
    return torch.matmul(p_attn, value), p_attn # B*h*N*N x B*h*N*d_v 每个单词与其他位置的单词V累加（矩阵乘） --> B*h*N*d_v
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # 此处的输入已经假设是 Multi-Head 形式
        # B*N*d_model | B*N*d_model |  B*N*d_model |  B*N*N
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # 1*1*N*N
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # B*N*d_model x d_model*d_model => B*N*d_model==> B*N*h*d_k ==> B*h*N*d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                                    for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        # B*h*N*d_k --> B*h*N*d_v , B*h*N*N
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        # B*h*N*d_v ==> B*N*h*d_v ==> B*N*d_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # B*N*d_model x d_model*d_model ==> B*N*d_model
        return self.linears[-1](x)
```

* 为什么进行缩放$$\sqrt{d_k}$$?
    * 对于较大的$$d_k$$，点积大幅增大, 将Softmax函数值推向具有极小梯度的区域
    * 为了阐明点积变大的原因，假设$$q$$和$$k$$是独立的随机变量, 平均值为$$0$$，方差为$$1$$，这样他们的点积为$$q\cdot k=\sum_i q_ik_i$$，同样是均值$$0$$为方差为$$d_k$$。
    * 为了抵消这种影响，我们用$$\frac{1}{\sqrt{d_k}}$$来缩放点积。
    * 可以从余弦相似度的角度解释 $$cos(q,k) = \frac{q\cdot k}{\parallel q\parallel \times \parallel k\parallel }$$

* 为什么使用“多头”注意力机制？
    * 把输入映射到不同的子空间，可以理解为从不同的角度观察，类似卷积核的思想。

* Transformer的输入dim和$$Q$$/$$K$$/$$V$$及Head数$$h$$的关系？
    * $$d_q=d_k$$
    * $$d_v$$和$$d_{model}$$可以任意维度
    
* 官方Transformer的multi-head attention有多少个“头”？q、k、v各有多少维？
    * 在Transformer中$$h=8$$
    * 对每个Head有$$d_k=d_v=d_{model}/ 8=64$$
    
* Transformer中哪些地方应用了 Multi-Head Attention机制？   
    * Encoder中的Multi-Head Attention层。Key、Value和Query都来自前一层的输出，Encoder中当前层的每个位置都能Attend到前一层的所有位置。
    * Decoder中的Multi-Head Attention层（*）。Query来自先前的解码器层，Key和Value来自Encoder的输出，Decoder中的每个位置Attend输入序列中的所有位置。
    * Decoder中的Masted Multi-Head Attention层。Key、Value和Query都来自前一层的输出，Decoder中的每个位置Attend当前解码位置和它前面的所有位置。在缩放后的点积Attention logit中，屏蔽（设为负无穷）Softmax的输入中所有对应着非法连接的Value。
    
## Position-Wise前馈网络
$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$

在Transformer中的Position-Wise前馈网络Layer中，包含两次线性变换，中间使用Relu激活函数，参数的维度为 $$W_1\in \mathbb{R}^{512 \times 2048} , W_2\in \mathbb{R}^{2048 \times 512}$$
每个P-W前馈网络对应一个位置（单词），不同的位置（单词）共享全连接参数。计算后得到的格式为 $$B\times N\times d_{model}$$，与输入格式一致。  

```python
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B*N*d_model x d_model*d_ff x d_ff*d_model ==> B*N*d_model  
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

## Add & Norm

$$
LN(x)=\alpha\times \frac{x-\mu}{\sigma + \varepsilon}+\beta
$$

以上提到的Multi-Head Attention层和Position-Wise前馈网络都属于SubLayer，结合Add&Norm组成一个完整的Layer。需要注意的是，在该版本中Norm出现在每次进入sublayer时最开始的第一个操作。

```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # B*N*d_model ==> B*N*1
        mean = x.mean(-1, keepdim=True)
        # B*N*d_model ==> B*N*1
        std = x.std(-1, keepdim=True)
        # d_model *  B*N*d_model + d_model ==> B*N*d_model
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # B*N*d_model + B*N*d_model 
        return x + self.dropout(sublayer(self.norm(x)))
```

* Layer Norm的作用是什么？
    * 随着网络深度的增加，数据的分布会不断发生变化,为了保证数据特征分布的稳定性，加入LN这样可以**加速模型的收敛速度**。
    * 防止输入数据落在激活函数的饱和区，发生梯度消失的问题，使得模型训练变得困难。
    
* Layer Norm 与 Batch Norm的区别？
    * BN的主要思想是: 在每一层的每一批数据(一个batch里的同一通道)上进行归一化。
    * LN的主要思想是:是在每一个样本(一个样本里的不同通道)上计算均值和方差，而不是 BN 那种在批方向计算均值和方差。

## Encoder

#### EncoderLayer
一个Layer包含两个Sublayer，分别是Attention Layer和 FFN Layer。
在Transformer中，Encoder包含6个Layer。

```python
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

## Decoder

除了拥有每个Encoder Layer中的两个Sublayer之外，Decoder还插入了第三种类型的Sublayer对Encoder的输出实行“多头”的Attention。
Mask确保了生成位置$$i$$的预测时，仅依赖小于$$i$$的位置处的已知输出，相当于把后面不该看到的信息屏蔽掉。

###### Mask Map
<center><img src="https://pic3.zhimg.com/80/v2-3c315c4be3eac87af1ed64eead1024ca_720w.jpg" width="30%" height="30%" /></center>

```python
def subsequent_mask(size):
    # 1*N*N
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

src-attn模块的输入来源于Encoder的Key、Value，Decoder上一层的self-attention的输出作为Query。

```python
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

## Positional Encoder Embedding

$$
PE_{pos,2i}=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE_{pos,2i+1}=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) 
$$

其中$$pos$$是位置, $$i$$是维度。也就是说，位置编码的每个维度都对应于一个正弦曲线,。
在Transformer模型中，dropout=0.1。
在Encoder与Decoder的输入部分都加入了位置编码。

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # 生成空位置向量   max_len*d_model
        position = torch.arange(0, max_len).unsqueeze(1) # max_len*1
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 1*max_len*d_model
        self.register_buffer('pe', pe) # 设置为buffer（不需要求梯度的参数）
        
    def forward(self, x):
        # B*N*d_model + 1*N*d_model --> B*N*d_model
        x = x + torch.tensor(self.pe[:, :x.size(1)])
        return self.dropout(x)
```

## 分类器 
分类器使用一个Linear层+Softmax进行分类，置于Decoder之后。

```python
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # B*N*d_model x d_model*voc ==> B*N*voc 
        return F.log_softmax(self.proj(x), dim=-1)
        
```

## Encoder-Decoder
通用的组织架构，同样适用于Transformer。

```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# Small example model.
tmp_model = make_model(10, 10, 2)
```

## 训练

#### 构造Batch和Mask
```python
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        # B*N 
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # B*1*N
        if trg is not None:
            self.trg = trg[:, :-1] # B*(N-1) Decoder输入
            self.trg_y = trg[:, 1:] # B*(N-1) Decoder输出
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2) 
        # B*1*N & 1*N*N ==> B*N*N
        tgt_mask = tgt_mask & torch.tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

