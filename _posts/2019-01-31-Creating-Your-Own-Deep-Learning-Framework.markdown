---
layout: post
comments: true
title:  "Creating Your Very Own Deep Learning Framework"
excerpt: "Designing a framework to do deep learning inference using only python."
date:   2019-01-31 02:04:00
mathjax: true
---

<div class="imgcap">
<img src="/assets/Creating-your-own-dl-framework/framework.png" width="100%">
<!--div class="thecap">(Image credit: <a href="http://cs231n.github.io/neural-networks-3/">cs231n</a>).</div-->
</div>

You may be the brain but software frameworks have always been the brawn behind development and sucess of deep learning algorithms. Frameworks tike Tensorflow, Caffe, Torch, Theano, etc. do a lot of the heavy lifting by abstracting out the time consuming and difficult task of coding the routines necessary for development. In this post, as a tribute to all the awesome engineers working towards the development of these frameworks, we shall implement a very small scale deep learning inference framework in python. I came across something like this while I was taking the Python for Machine Learning course in Udacity. I added more functionality to make it look like a complete framework very similar to tensorflow.