#import "@preview/colorful-boxes:1.3.1": *
#set text(font: ("Libertinus Serif", "Source Han Serif SC", "Source Han Serif"))

#align(right)[
  #highlight[= Lab 2 实验报告]

  PB21000030 马天开
]

== 实验流程

#outline-colorbox(color: "green", title: "数据集划分")[
  ```py
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
  ```
]

#outline-colorbox(color: "blue", title: "sklearn 调用")[
  ```py
  mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
  mlp.fit(standard_train, Y_train)
  result = mlp.predict(standard_test)
  ```
]

#outline-colorbox(color: "red", title: "Sigmoid 函数")[
  ```py
  def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
      return x * (1 - x)
  ```
]

#outline-colorbox(color: "green", title: "正向 & 反向传播")[
  ```py
  def forward(self, X):
      self.hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
      self.hidden_output = self.sigmoid(self.hidden)

      self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
      self.output_output = self.sigmoid(self.output)

      return self.output_output

  def backward(self, X, y, output, learning_rate):
      self.output_error = y - output
      self.output_delta = self.output_error * self.sigmoid_derivative(output)

      self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
      self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)

      self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta) * learning_rate
      self.weights_input_hidden += X.T.dot(self.hidden_delta) * learning_rate
      self.bias_output += np.sum(self.output_delta) * learning_rate
      self.bias_hidden += np.sum(self.hidden_delta) * learning_rate
  ```
]

== 调整超参数

```py
hidden_size = 2
nn.train(standard_train, Y_train_encoded, epochs=1000, learning_rate=0.1)
```

预测的准确率为： 0.5666666666666667

```py
hidden_size = 10
nn.train(standard_train, Y_train_encoded, epochs=1000, learning_rate=0.1)
```

预测的准确率为： 1.0

== 实验总结

花费时间 <=10 分钟，有点简单了……