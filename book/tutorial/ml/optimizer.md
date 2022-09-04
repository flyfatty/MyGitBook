<div align="center"/>![](https://pic4.zhimg.com/v2-ed8f70ed5bb8e8a5ba4dd0cf99c0f557_b.webp)

# <center>Optimizer</center>

[参考文章](https://blog.csdn.net/google19890102/article/details/69942970)


> finding the parameters $$\theta $$ of a neural network that significantly reduce a cost function $$J(\theta) $$ , which typically includes a performance measure evaluated on the entire training set as well as additional regularization terms

### 1、 梯度下降法 Gradient Descent

$$
\theta_{t+1}=\theta_t - \alpha \cdot \triangledown J(\theta)
$$

###### 特点

- 学习率不好选择，过低收敛缓慢，过高波动太大
- 所有参数使用同样的学习率
- 容易收敛到局部最优，可能会被困在鞍点（梯度为0）

##### Batch Gradient Descent (BGD)

- n表示所有样本数量

$$
\theta_{t+1}=\theta_t - \alpha \cdot \frac{1}{n}\sum_{i=1}^{n} \triangledown J_i(\theta,x^i,y^i)
$$

##### Stochastic Gradient Descent (SGD)

- 每一个step仅一个样本，现实中一般指mini-batch

$$
\theta_{t+1}=\theta_t - \alpha \cdot \triangledown J_i(\theta,x^i,y^i)
$$

##### Mini-Batch Gradient Descent

- m表示一个batch size

$$
\theta_{t+1}=\theta_t - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\triangledown J_i(\theta,x^i,y^i)
$$

### 2、动量法

- $$\mu$$ 动量因子（衰减系数），表示保持的动量比重，通常取0.9
- $$m_t$$ 指t时刻的动量

###### 特点

- 加快收敛并且减少动荡

##### Momentum

$$
\begin{align} 
m_{t+1} &= \mu \cdot m_t + \alpha \cdot \triangledown J(\theta) \\ 
\theta_{t+1} &= \theta_t - m_{t+1}
\end{align} 
$$

##### NAG（Nesterov accelerated gradient）

- 在原始形式中，NAG相对于Momentum的改进在于，以“向前看”看到的梯度而不是当前位置梯度去更新。经过变换之后的等效形式中，NAG算法相对于Momentum多了一个本次梯度相对上次梯度的变化量，这个变化量本质上是对目标函数二阶导的近似。由于利用了二阶导的信息，NAG算法才会比Momentum具有更快的收敛速度
$$
\begin{align} 
m_{t+1} &= \mu \cdot m_t + \alpha \cdot \triangledown J(\theta - \mu \cdot m_t) \\ 
\theta_{t+1} &= \theta_t - m_{t+1} 
\end{align} 
$$

### 3、自适应学习率算法

- $$n_t$$ 梯度累计变量
- $$\varepsilon$$ 极小常量值防止分母为0，一般设置 10e-8

##### Adagrad

- $$\delta$$ 全局学习率，一般设置 0.01

###### 特点

- 作为约束项$$\frac{1}{\sqrt{\sum_{i=0}^{t}g_i^2 +\varepsilon }}$$，随着梯度累计越来越多，学习率被衰减的越多，中后期参数更新量可能趋近于0，无法学习。
- 仍然需要设置全局学习率$$\delta$$，如果设置过大会使约束项过于敏感，对梯度调节太大。

$$
\begin{align} g &= \triangledown_{\theta} J(\theta) \\ 
n_t &= n_{t-1} + g^2 \\ 
\triangle \theta &= \frac{\delta}{\sqrt{n_t + \varepsilon}}\cdot g \\ 
\theta &= \theta - \triangle \theta 
\end{align} 
$$

##### RMSprop

- $$\delta$$ 全局学习率，一般设置 0.001
- $$\nu $$ 衰减系数，一般设置 0.9

###### 特点

- 修改了AdaGrad的梯度平方和累加为指数加权的移动平均，使得其在非凸设定下效果更好。
- RMSprop依然依赖于全局学习率$$\rho$$
- 适合处理非平稳目标(包括季节性和周期性)——对于RNN效果很好

$$
\begin{align} g &= \triangledown_{\theta} J(\theta) \\ 
n_t &= \nu \cdot n_{t-1} + (1-\nu) \cdot g^2 \\ 
\triangle\theta &= \frac{\delta}{\sqrt{n_t + \varepsilon}}\cdot g \\ 
\theta &= \theta - \triangle \theta \end{align} 
$$

##### Adam (AdamW , HuggingfaceAdamW)

- $$\delta$$ 全局学习率，一般设置 0.001
- $$\mu $$ 控制1阶动量，一般设置 0.9
- $$\nu $$ 控制2阶动量，一般设置 0.999
- $$w $$ AdamW参数 - weight decay ，一般设置 0.01

###### 特点

- 结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点
- 自动调整参数的学习率
- 也适用于大多非凸优化 - 适用于大数据集和高维空间
- AdamW相当于在原Loss的基础上添加了L2 Norm
- HuggingFace-AdamW有一点区别，$$w$$系数后的参数使用的未添加L2 Norm的$$\theta_{t+1}$$ 
  
$$
\begin{align} 
g &= \triangledown_{\theta} J(\theta) \\ 
m_t &= \mu \cdot m_{t-1} + (1-\mu) \cdot g \\ 
n_t &= \nu \cdot n_{t-1} + (1-\nu) \cdot g^2 \\ 
\hat{m_t} &= \frac{m_t}{1-\mu^t} \\ 
\hat{n_t} &= \frac{n_t}{1-\nu^t} \\ 
\triangle \theta &= \frac{\delta}{\sqrt{\hat{n_t} +\varepsilon}}\cdot \hat{m_t} \\ 
\triangle \theta &= \delta \cdot (\frac{1}{\sqrt{\hat{n_t} + \varepsilon}}\cdot \hat{m_t} + w\theta_t )【AdamW】\\ 
\theta_{t+1} &= \theta_t - \triangle \theta 
\end{align} 
$$

### 更多方法
##### [lookahead](https://mp.weixin.qq.com/s/3J-28xd0pyToSy8zzKs1RA)
##### [训练过程中改变学习率的方法 noam](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer)

<div align="center"/>![](http://nlp.seas.harvard.edu/images/the-annotated-transformer_69_0.png)

$$
lr(s)=\frac{\lambda}{\sqrt{d_m}}\cdot min(\frac{1}{s^{0.5}},\frac{s}{s_{warmup}^{1.5}})
$$

##### [Optuna](https://github.com/optuna/optuna) : A hyperparameter optimization framework

* 概念
    * Trial  一次针对目标函数的尝试
    * Study 包含多次尝试的一次会话 ，指定`study_name` 优化方向`direction` ; 指定sampler/pruner ; 设置db级别的storage用于并行(同时启动多个任务即可)
    * Parameter 需要优化的参数
    
* 常用API
    * 执行study `study.optimize(objective, n_trials=100)`
    * 获得study后的最优parameter  `study.best_params`
    * 获得study后的最小目标函数值  `study.best_value`
    * 获得study后的最佳trial的信息  `study.best_trial`
    * 获得study后的所有trials信息  `study.trials`

* 定义参数空间
  * 分支 if / 循环 for
  * categorical / int / float

```python
optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
num_channels = trial.suggest_int("num_channels", 32, 512, log=True)
drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
```


* 优化算法
  * sampling Algorithms (缩减参数空间规模）
    * Tree-structured Parzen Estimator algorithm `optuna.samplers.TPESampler` (默认)
    * CMA-ES based algorithm `optuna.samplers.CmaEsSampler`
    * Grid Search `optuna.samplers.GridSampler`
    * Random Search  `optuna.samplers.RandomSampler`

  * pruning Algorithms (自动化的 early-stopping))
    * Asynchronous Successive Halving algorithm `optuna.pruners.SuccessiveHalvingPruner`
    * Hyperband algorithm `optuna.pruners.HyperbandPruner`
    * Median pruning algorithm `optuna.pruners.MedianPruner` (默认)
      * Prune if the trial’s best intermediate result is worse than median of intermediate results of previous trials at the same step.
    * Threshold pruning algorithm `optuna.pruners.ThresholdPruner`
      * 指定固定的 max/min threashold 作为判断标准 
    * 应用指定方法进行裁剪 Trial.report / Trial.should_prune

  * 搭配建议
    * RandomSampler + MedianPruner / TPESampler + Hyperband ( for not deep-learning)
    * 通过 callback 的方式与其他框架整合 `optuna.integration.*`


* sklearn + optuna + pruner

```python
import optuna
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=0
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step) # report 汇报每个step的验证集评价

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return 1.0 - clf.score(valid_x, valid_y) # 返回最终的验证集评价

study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
```

* allennlp + optuna + pruner

```jsonnet
// alennlp  hparams.json
[
  {
    "type": "int",
    "attributes": {
      "name": "embedding_dim",
      "low": 64,
      "high": 128
    }
  },
  {
    "type": "float",
    "attributes": {
      "name": "dropout",
      "low": 0.0,
      "high": 0.5
    }
  },
  {
  "type": "categorical",
  "attributes": {
    "name": "kernel",
    "choices": ["linear", "poly", "rbf"]
    }
  }
]


// optuna.json
{
  "pruner": {
    "type": "HyperbandPruner",
    "attributes": {
      "min_resource": 1,
      "reduction_factor": 5
    }
  },
  "sampler": {
    "type": "TPESampler",
    "attributes": {
      "n_startup_trials": 5
    }
  }
}

// *.jsonnet
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local kernel = std.parseJson(std.extVar('kernel'));

trainer: {
  // NOTE add `optuna_pruner` here!
  epoch_callbacks: [
    {
      type: 'optuna_pruner',
    }
  ],
  num_epochs: num_epochs,
  optimizer: {
    lr: lr,
    type: 'sgd',
  },
  validation_metric: '+accuracy',
}
```

```shell
# 训练
allennlp tune \
    imdb_optuna.jsonnet \
    hparams.json \
    --optuna-param-path optuna.json \
    --serialization-dir result \
    --study-name demo \
    --direction maximize \
    --storage mysql://<user_name>:<passwd>@<db_host>/<db_name> \
    --skip-if-exists

# 获得参数结果
allennlp best-params \
    --study-name demo \
    --storage mysql://<user_name>:<passwd>@<db_host>/<db_name>
    
# 使用最佳参数 Retrain 
allennlp retrain \
    imdb_optuna.jsonnet \
    --serialization-dir retrain_result \
    --study-name demo \
    --storage mysql://<user_name>:<passwd>@<db_host>/<db_name> 
```
  