# 实验 3 报告

PB21000030 马天开

## 实验流程

- 按照任务要求完成、调试 `submisson.py`
- `python train.py --use_pca`
- `python grade.py --sample_index 1 --result_path ../results/{datetime}`
- `python `visualization.py --results_path "../results/{datetime}" --cluster_label`
- `python visualization.py --results_path "../results/{datetime}"`

## 超参数修改

未修改任何参数。

## 实验结果

### 降维可视化

|             | True Label                                        | Cluster Label                                        |
| ----------- | ------------------------------------------------- | ---------------------------------------------------- |
| PCA         | ![](../results/2024-12-15_14-49-00/true_pca.png)  | ![](../results/2024-12-15_14-49-00/cluster_pca.png)  |
| tSNE        | ![](../results/2024-12-15_14-49-00/true_tsne.png) | ![](../results/2024-12-15_14-49-00/cluster_tsne.png) |
| AutoEncoder | ![](../results/2024-12-15_14-49-00/true_ae.png)   | ![](../results/2024-12-15_14-49-00/cluster_ae.png)   |


### 评分

```bash
python grade.py --sample_index 1 --result_path ../results/2024-12-15_14-49-00
```

输出：

```bash
Your model got a Davies Bouldin score of 3.52
The sklearn model got a Davies Bouldin score of 2.98
You got a score of 25.34/30 in total.
```

### 生成结果

DDPM 生成的图片：

![](../results/2024-12-15_14-49-00/ddpm_sample.png)

GMM 生成的图片：

![](../results/2024-12-15_14-49-00/gmm_sample.png)

## 回答问题

- 从**训练速度**,**降维效率**,**灵活性**(eg.是否适用于各种类型的数据),**对数据分布的保持程度**,**可视化效果**这几个方面比较 PCA,tSNE,AutoEncoder 这三种降维方法的优劣 (你可以列一个表格)(10 points)

    |                      | PCA  | tSNE | AutoEncoder |
    | -------------------- | ---- | ---- | ----------- |
    | 训练速度             | 最快 | 最慢 | 中等        |
    | 降维效率             | 高   | 低   | 低          |
    | 灵活性               | 低   | 高   | 高          |
    | 对数据分布的保持程度 | 低   | 高   | 高          |
    | ｜ 可视化效果        | 高   | 低   | 高          |


- 从**生成效率**,**生成质量**,**灵活性**(eg.是否适用于各种类型的数据),**是否可控**(eg.生成指定类别的样本) 这几个方面比较 GMM 和 DDPM 的优劣，(DDPM 的原理参考书面作业第四题)(10 points)

    |          | GMM | DDPM |
    | -------- | --- | ---- |
    | 生成效率 | 高  | 低   |
    | 生成质量 | 低  | 高   |
    | 灵活性   | 低  | 高   |
    | 是否可控 | 是  | 是   |

## 反馈

无，这次框架 Debug 起来合理很多了。