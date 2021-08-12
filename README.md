# Label-Distangled-Analysis_code-demo
The code derives from the paper entitled "Label Disentangled Analysis for unsupervised visual domain adaptation". In this code, the dataset is substituded to the gas dataset. 
* demo.m is needed for initialization
* onehot.m is a function for label-to-one-hot switchment.
* A.mat is the original dataset of Benchmark sensor drift dataset containing 10 batches. The 1st batch is recognized to be the no-drift one, while the extent of drift in batches are strengthen one by one till Batch 10. 
* Gas_data.mat is the dataset containing 10 batches, but normalized by L2 normalization. It have the ordinary label set.

The specific analysis of **Label Distangled Analysis(LDA)** is stated in my personal website: https://adonijahh.github.io/2021/08/04/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%EF%BC%9ALabel%20Disentangled%20Analysis%20for%20unsupervised%20visual%20domain%20adaptation/

该代码源自题为“无监督视觉域适应的标签解开分析”的论文。 在这段代码中，数据集采用的是来自UCSD的气体数据集。
* demo.m用于初始化和运行
* onehot.m为标签由one-hot形式和常规标签数据集的转换子函数。
* A.mat是气体数据集的原始数据，其中有10个批次的子数据集。第一个数据集无漂移，其他数据集的漂移程度依次加深。
* Gas_data.mat为A.mat的2-范数归一化版本，包含标签信息。

LDA算法的具体分析陈述在我的个人网站，具体参见网站：
https://adonijahh.github.io/2021/08/04/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%EF%BC%9ALabel%20Disentangled%20Analysis%20for%20unsupervised%20visual%20domain%20adaptation/
