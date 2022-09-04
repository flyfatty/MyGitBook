
# <center>评价指标</center>

### 分类

###### 混淆矩阵

-|预测值=1|预测值=0
:-:|:-:|:-:
真实值=1|TP 真阳性|FN 假阴性
真实值=0|FP 假阳性|TN 真阴性

###### 准确率 Accuracy

$$
Accuracy =  \frac{TP + TN}{TP+FP+TN+FN} 
$$

###### 精确率 、查准率 Precision

$$
Precision = \frac{TP}{TP+FP}
$$
  
###### 召回率、查全率 Recall

$$
Recall = \frac{TP}{TP+FN}
$$

###### F1 Score

>调和平均数

$$
\begin{align}
F1&=\frac{1}{\frac{1}{2}\cdot \frac{1}{P}+\frac{1}{2}\cdot \frac{1}{R}} \\
  &=\frac{2\cdot P\cdot R}{P+R}
\end{align}
$$

###### Fβ Score

$$
\begin{align}
F\beta&=\frac{1}{\frac{1}{\beta+1}\cdot \frac{1}{P}+\frac{\beta}{\beta+1}\cdot \frac{1}{R}} \\
&=\frac{(1+\beta)\cdot P\cdot R}{\beta P+R}
\end{align}
$$

###### P-R曲线

###### ROC曲线、AUC