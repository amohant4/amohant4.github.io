---
layout: post
comments: true
title:  "Learning Rate in Deep Learnaing"
excerpt: "An introduction to learning rate hyper parameter used in deep neural network training. "
date:   2019-01-29 05:04:00
mathjax: true
---

Learning rate is one of the most critical hyper-parameter and decides the fate of your training process. If you mess up the learning rate, then the optimizer might not be able to converge at all. Learning rate controls how much we updating the parameters in our network with respect to the gradient of loss. 


$$mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n}$$

The gradient is given by: 

$$g = \frac{1}{m^{'}} \nabla_{\theta}\sum_{i=1}^{m^{'}}L(x^{(i)}, y^{(i)}, \theta)}$$

Using this gradient from the minibatch, stochastic gradient descent follows the estimated downhill: 

$$\theta \leftarrow \theta - \epsilon g $$  

