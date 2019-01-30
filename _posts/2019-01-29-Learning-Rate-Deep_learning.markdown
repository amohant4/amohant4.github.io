---
layout: post
comments: true
title:  "Learning Rate in Deep Learnaing"
excerpt: "An introduction to learning rate hyper parameter used in deep neural network training. "
date:   2019-01-29 05:04:00
mathjax: true
---

Learning rate is one of the most critical hyper-parameter and decides the fate of your training process. If you mess up the learning rate, then the optimizer might not be able to converge at all. Learning rate controls how much we updating the parameters in our network with respect to the gradient of loss. 

The gradient is given by: 

$$g = \frac{1}{m^{'}}\nabla_{\theta}\sum_{i=1}^{m^{'}}L(x^{(i)},y^{(i)},\theta)$$

Using this gradient from the minibatch, stochastic gradient descent follows the estimated downhill: 

$$\theta \leftarrow \theta - \epsilon g $$  

where $$ \epsilon $$ is the learning rate. 

```
new_weight = existing_weight — learning_rate * gradient
```

The following figure explains the effects of learning rate on gradient descent. 

<div class="imgcap">
<img src="/assets/Learning-Rate_Selection/effect_of_lr.png" width="50%">
<div class="thecap">Effect of various learning rates on convergence (Image credit: <a href="http://cs231n.github.io/neural-networks-3/">cs231n)</a>.</div>
</div>