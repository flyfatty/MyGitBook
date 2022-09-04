<div align="center"/>![](https://pic1.zhimg.com/80/v2-87e80c796751415635f84bae864bd154_720w.jpg)

# <center>损失函数</center>

### facal loss

* $$\gamma$$ 一般设置 2:负责降低简单样本的损失值, 以解决加和后负样本loss值很大 
* $$\alpha_t$$ 一般设置 0.25:调和正负样本的不平均，如果设置0.25, 那么就表示负样本为0.75, 对应公式 $$1-\alpha$$

$$
FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t)
$$
