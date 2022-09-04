<center><h1>多标签CrossEntropy</h1></center>

对于多标签分类问题，一个简单且直观的想法是，将其转换成多个二分类问题。具体来说，就是把“n个类别选k个目标类别”的问题转化为“n个二分类”问题，然后将n个二分类交叉熵之和作为损失。这种做法会遇到的问题是，一般尤其在序列标注任务中$$n \gg k$$，会面临严重的类别不均衡问题，那么我们就需要一些后续的平衡策略，比如，手动调整正负样本的权重；使用focal loss等等。

其实，n选k应该是n选1的自然延伸，却需要多做那么多工作，是不科学的事情。

首先考虑多分类单标签问题，即从n个候选类别中选1个目标类别。假设各个类的得分为$$s_1, s_2, ... s_n$$，目标类为$$t \in {1, 2, ..., n}$$，那么损失为：
$$
-log\frac {e^{s_t}}{\sum _{i=1}^ne^{s_i}} = -s_t+log\sum _{i=1}^ne^{s_i}
$$
换一种方式看单标签分类交叉熵：
$$
\begin{aligned}
-log\frac {e^{s_t}}{\sum _{i=1}^ne^{s_i}} &= -log\frac {1}{\sum _{i=1}^n e^{s_i-s_t}} \\
&= log \sum _{i=1}^n e^{s_i-s_t} \\
&= log(1+\sum _{i=1,i\not=t}^ne^{s_i-s_t})
\end{aligned}
$$
因为logsumexp是max的光滑近似，因此：
$$
log(1+\sum _{i=1,i\not=t}^ne^{s_i-s_t}) \approx max
\left[
  \begin{matrix}
   0 \\
   s_1-s_t \\
   s_2-s_t \\
   ... \\
   s_{t-1}-s_t \\
   s_{t+1}-s_t \\
   ... \\
   s_n-s_t
  \end{matrix}
  \right]
$$
该损失的特点为，所有的非目标类得分$$\{s_1, s_2, ..., s_{t-1}, s_{t+1},...,s_n\}$$和目标类得分$$\{s_t\}$$两两做差比较，它们差的最大值都要尽可能小于零，因此实现了“目标类得分都大于每个非目标类的得分”的效果。

如果有多个目标类的多标签分类场景，我们也希望“每个目标类得分都不小于每个非目标类的得分”，因此loss有如下形式：
$$
log(1 + \sum _{i\in \Omega_{neg},j\in \Omega_{pos}}e^{s_i-s_j}) = log(1 + \sum _{i\in \Omega_{neg}}e^{s_i}\cdot \sum _{j\in \Omega_{pos}}e^{-s_j}) \tag{1}
$$
即我们希望$$s_i<s_j$$。

n类选k类，如果k是固定的，那么直接用式(1)即可，预测时直接输出得分最大的k个类别。

对于k 不固定的多标签分类，需要一个阈值来确定输出哪些类，因此引入一个0类，希望目标类的分数都大于$$s_0$$，非目标类分数都小于$$s_0$$。从(1)式我们知道，“希望$$s_i<s_j$$就在$$log$$里边加入$$e^{s_i-s_j}$$”，因此(1)式变为：
$$
log(1 + \sum _{i\in \Omega_{neg},j\in \Omega_{pos}}e^{s_i-s_j} + \sum _{i\in \Omega_{neg}}e^{s_i-s_0} + \sum _{j\in \Omega_{pos}}e^{s_0-s_j})  \\
\\
\begin{aligned}
&= log(e^{s_0-s_0} + \sum _{i\in \Omega_{neg},j\in \Omega_{pos}}e^{s_i-s_j} + \sum _{i\in \Omega_{neg}}e^{s_i-s_0} + \sum _{j\in \Omega_{pos}}e^{s_0-s_j}) \\
\\
&= log[e^{s_0}(e^{-s_0} + \sum _{j\in \Omega_{pos}}e^{-s_j}) + \sum _{i\in \Omega_{neg}}e^{s_i}(\sum _{j\in \Omega_{pos}}e^{-s_j}+e^{-s_0})] \\
\\
&= log[(\sum _{j\in \Omega_{pos}}e^{-s_j}+e^{-s_0})(\sum _{i\in \Omega_{neg}}e^{s_i}+e^{s_0})] \\
\\
&= log(\sum _{j\in \Omega_{pos}}e^{-s_j}+e^{-s_0}) + log(\sum _{i\in \Omega_{neg}}e^{s_i}+e^{s_0})
\end{aligned} \tag{2}
$$
指定$$s_0=0$$，那么(2)式为：
$$
log(1 + \sum _{j\in \Omega_{pos}}e^{-s_j}) + log(1 + \sum _{i\in \Omega_{neg}}e^{s_i}) \tag{3}
$$
预测时取$$s_k>0$$为正例。
代码实现完全就是公式(3):
```python
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素非0即1，
           1表示对应的类为目标类，0表示对应的类为非目标类；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
    """
    y_pred = (1 - 2 * y_true) * y_pred  #  把正标签的分数变负号
    y_neg = y_pred - y_true * K.infinity()  #  把负标签取出来，正标签位置对应负无穷，在logsumexp计算时相当于没有，即logsumexp([a, b, c]) == logsumexp([a, b, c, -无穷])
    y_pos = y_pred - (1 - y_true) * K.infinity() # 与上一行同理
    zeros = K.zeros_like(y_pred[..., :1])
    y_neg = K.concatenate([y_neg, zeros], axis=-1) # e的0次方等于1
    y_pos = K.concatenate([y_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_neg, axis=-1)  #  axis == -1 表示在最后一个维度上进行计算
    pos_loss = K.logsumexp(y_pos, axis=-1)
    return neg_loss + pos_loss
```

$$
s_\alpha(i, j)=(\pmb{W}_q\pmb{h}_i)^T(\pmb{W}_k\pmb{h}_j)+\pmb{w}_\alpha^T[\pmb{h}_i;\pmb{h}_j]
$$