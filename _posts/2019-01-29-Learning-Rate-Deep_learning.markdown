---
layout: post
comments: true
title:  "Learning Rate in Deep Learning"
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

The following figure explains the effects of learning rate on gradient descent. A very small learning rate will make gradient descent take small steps even if the gradient is big, thus slowing the process of learning. If the learning rate is high, then if becomes impossible to learn very small changes in the parameters needed to fine tune the model towards the end of the training process, so the error flattens out very early. If the learning rate is very high, then gradient descent takes big steps and jumps around. This can lead to divergence and thus increase the error. 

<div class="imgcap">
<img src="/assets/Learning-Rate-Selection/effect_of_lr.png" width="35%">
<div class="thecap">Effect of various learning rates on convergence (Image credit: <a href="http://cs231n.github.io/neural-networks-3/">cs231n</a>).</div>
</div>

Now the question aries, what is the best value of the learning rate and how to decide it ? A systematic way to estimate a good learning rate is by training the model initially with a very low learning rate and increasing it (either linearly or exponentially) at each iteration (illustrated below). We keep doing it to the point where the loss stops decreasing and starts to increase. That means that the learning rate is too high for the application and so gradient descent is diverging. For practical applications our learning rate should ideally be 1 or 2 step smaller than this value. 

<div class="imgcap">
<img src="/assets/Learning-Rate-Selection/schedule_lr_1.png" width="35%">
<div class="thecap">Image credit: <a href="https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10"></a>Hafidz's Blog.</div>
</div>

If we keep track of the learning rate and plot log of the learning rate and the error we will see a plot as shown below. A good learning rate somewhere to the left to the lowest point of the graph (as demonstrated in below graph). In this case, its 0.001 to 0.01. 

<div class="imgcap">
<img src="/assets/Learning-Rate-Selection/select_lr.png" width="35%">
<div class="thecap">Image credit: <a href="https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10"></a>Hafidz's Blog.</div>
</div>

In general no fixed learning rate works best for the entire training process. Typically we start with a learning rate found using the method described above. During the training process we change learning rate to best facilitate learning. There are many different ways to accomplish this. In this blog, we will go through a few popular learning rate scheduler. 


#### Step Decay

Step decay schedule drops the learning rate by a factor every few epochs. The mathematical form of step decay is:

$$ \epsilon_{k} = \epsilon_{0} \times \alpha^{\lfloork/N\rfloor }$$

where, $$\epsilon_{k}$$ is the learning rate for $$k_{th}$$ epoch, $$\epsilon_{0}$$ is the initial learning rate, $$\alpha$$ is the fraction by which learning rate is reduced, $$\lfloor . \rfloor$$ is floor operation and N is the number of epochs after which learning rate is dropped.  





#### Linear or Exponential Time-Based Decay

This technique is also known as learning rate annealing. We start with a relatively high learning rate and then gradually lower it during training. The intuition behind this approach is that we'd like to traverse quickly from the initial parameters to a range of "good" parameter values but then we'd like a learning rate small enough that we can explore the "deeper, but narrower parts of the loss function" (fine tuning the parameters to get best results). 

In practice, it is common to decay the learning rate  until iteration $$\tau$$. In case of linear decay, the learning rate is modified in the following manner:

$$ \epsilon_{k} = (1-\alpha)\epsilon_{0} + \alpha \epsilon_{\tau} $$

with $$ \alpha = \frac{k}{\tau}$$. After iteration $$ \tau$$, it is common to leave $$\epsilon$$  constant.   

#### 




#### Cyclic learning rates