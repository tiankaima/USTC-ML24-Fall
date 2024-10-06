#import "@preview/colorful-boxes:1.3.1": *
#set text(font: ("linux libertine", "Source Han Serif SC", "Source Han Serif"))


#align(right)[
  #highlight[= Lab 1 实验报告]

  PB21000030 马天开
]

== 实验流程 & 调试超参数过程

=== 环境搭建

- `pipenv --python3.9`

- `pipenv shell`

- `pip install -r requirements.txt`

=== 实现过程

- 完成 `src/submission.py` 要求的内容


#outline-colorbox(color: "green", title: "Note")[
  调试中注意到: BCE Loss 使用如下方法计算:

  $
    l(x, y) = -(y dot log (x) + (1 - y) dot log(1 - x))
  $

  其中若 $x in {0,1}$ 时会给出一个 $oo$ 的梯度, 在参考了 PyTorch 的文档对这一问题的处理 (#link("https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html")[Link here]) 后, 决定对 `np.log` 的结果 Clamp 到 $[-100, +oo)$, 实现如下:

  ```py
  @np.vectorize
  def new_log(x):
      if x == 0:
          return -100  # to avoid log(0), as pytorch does
      return np.log(x)
  ```

  对应 BCE Loss 的计算:

  ```py
  -np.mean(y_true * new_log(y_pred) + (1 - y_true) * new_log(1 - y_pred))
  ```
]

- Regression 任务的调参过程

  (是 `history | grep train` 复原的过程, 可能有遗漏)

  - #strike[`python trainR.py --results_path "..\results\train\"` 怎么报错目录不存在, 哦原来是反斜杠]

  - 默认参数跑一轮. 好慢

  - 开始提 `lr`, 从 `1e-5`, `1e-4`, 最后定在 `5e-5`. 大概是能收敛就继续加.

  - 对速度还是不满意, 于是开始降 `batch_size`, 从 `64`, `32`, 最后定在 `16`. 速度快了很多.

  - 最后发现到收敛附近速度实在是难以接受, 于是 `lr_decay` 直接设置成 `0`, 放弃学习率衰减, 同时大概 `epochs=10` 就足够收敛了.

  #align(center)[
    #image("regression.png", height: 200pt)

    #strike[草, 好完美的 loss 曲线]
  ]

- Classification 任务的调参过程

  - 在 Regression 中尝到了放弃学习率衰减的甜头, 于是这里同样试图调高 `lr`, 降低 `lr_decay=1`. `steps=100` 效果已经很好

  #align(center)[
    #image("classification.png", height: 200pt)
  ]

== 最好结果

- Regression

```txt
MSE Error: 0.6968
R^2: 0.0459
Average prediction: 4.277902644548472
Relative error: 0.0172238318634599
```

- Classification

```txt
Accuracy: 0.9996379622446341
```

== 问题

- 在对数据进行预处理时, 你做了那些额外的操作, 为什么?

  首先是显而易见的对 `Run_time` 列取对数, 显然随着其他指标的增加, 这应该是个指数级的变化, 处理成对数就得到了一个线性的关系.

  尝试对其他列进行归一化/标准化, 但是并未得到调参上的便利, 于是放弃了.

- 在处理分类问题时, 使用交叉熵损失的优势是什么?

  在理论上有概率的解释, 同时也等价于 NLL Loss.

  在实践中与 Softmax 结合, 一个处理成概率分布, 一个衡量差异值, 结合在一起梯度也比较容易计算.

- 本次实验中回归问题参数并不好调, 在实验过程中你总结了哪些调参经验?

  #strike[说起来这个场景里完全没用上 GPU 嘛] 数据线性关系很强的时候可以大胆一点, 降低 `batch_size=16` 也不会过拟合, $lr$ 在收敛的情况下拉大就完了.

- 你是否做了正则化, 效果如何? 为什么?

  并没出现过拟合的情况, 所以并没有做正则化.

== 反馈

用时方面还好, 大概 DDL 前两个晚上赶出来了. #strike[书面作业有点头疼]

#strike[提醒助教记得格式化代码, `*.py` 和 `.md` format 一下真的干净不少]