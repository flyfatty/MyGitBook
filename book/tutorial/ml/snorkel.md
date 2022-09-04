<div align="center"/>![](https://www.snorkel.org/doks-theme/assets/images/layout/Overview.png)

# <center>Snorkel</center>

> Snorkel is a system for programmatically building and managing training datasets without manual labeling.

### 弱监督学习

类型|策略
:-:|:-:
仅部分labeled data | active learning 主动学习、semi-supervised learning 半监督学习
label粒度较粗 | multi-instance learning 多示例学习
存在错误或模棱两可的label | learning with label noise 带噪学习

### snorkel一般步骤

1. **标注函数** (LFs)
    * 方法
      * ① 模式匹配 （关键字、正则匹配）
      * ② 启发式规则 （句子长度）
      * ③ 第三方模型 （词性、其它近似任务）
      * ④ 远程监督
      * ⑤ 众包标注
    * 步骤
        * 观测样本或利用先验知识总结模式
        * 写一个LF初始版本
        * 检查在采样的训练集（或验证集）上的效果
        * 循环步骤不断改进LFs，提升覆盖率/准确率

 ```python
 from snorkel.labeling import labeling_function
 @labeling_function()
 def lf_keyword_my(x):
     """Many spam comments talk about 'my channel', 'my video', etc."""
     return SPAM if "my" in x.text.lower() else ABSTAIN
 ```

2. 建模 & 融合 LFs
   * **label matrix**, L_train ==> 行表示data point，列表示LFs
   * evaluate metrics
      * Polarity  标签集合
      * Coverage 覆盖率
      * Overlaps 重叠率 至少存在一个其它LF支持该LF的选择
      * Conflicts 冲突率 至少存在一个其它LF不同于该LF的选择
      * Correct 正确数 (存在gold labels)
      * Incorrect 错误数 (存在gold labels)
      * Accuracy 准确率 (存在gold labels)

    ```python
    # evaluate LFs
    LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    
    # 方便过滤指定label进行分析
    from snorkel.analysis import get_label_buckets
    
    buckets = get_label_buckets(L_train[:, 0], L_train[:, 1])
    df_train.iloc[buckets[(ABSTAIN, SPAM)]].sample(10, random_state=1)
    ```

   * **LabelModel** ==>  modeling LFs , 类似 stacking 的思路

    ```python
    from snorkel.labeling.model import LabelModel
    from snorkel.labeling import PandasLFApplier , LFAnalysis
    
    # Define the set of labeling functions (LFs)
    lfs = [lf_keyword_my, lf_regex_check_out, lf_short_comment, lf_textblob_polarity]
    
    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    ```

   * 删除 `ABSTAIN` 数据形成 Train Data
   
    ```python
       df_train = df_train[df_train.label != ABSTAIN]
    ```

3. **变换函数** (TFs) for Data Augmentation
   * 策略: text随机替换同义词 
   
4. **切片函数** (SFs) for Data Subset Selection

5. 训练最终模型,提升泛化能力