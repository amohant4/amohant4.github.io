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

You may be the brain but software frameworks have always been the brawn behind development and sucess of deep learning algorithms. Frameworks tike Tensorflow, Caffe, Torch, Theano, etc. do a lot of the heavy lifting by abstracting out the time consuming and difficult task of coding the routines necessary for development. In this post, as a tribute to all the awesome engineers working towards the development of these frameworks, we shall implement a very small scale deep learning inference framework in python. I came across something like this while I was taking the Python for Machine Learning course in Udacity. I added more functionality to make it look like a complete framework very similar to tensorflow :-)

Lets start with a very simple tensorflow example.

```
import tensorflow as tf

x = tf.placeholder(tf.float32)  
y = tf.placeholder(tf.float32)
z = tf.add(x,y)

with tf.Session() as sess:
    feed_dict = {x: 1.0, y: 2.0}
    result = sess.run(z, feed_dict=feed_dict)
    print result
    feed_dict = {x: 1.5, y: 2.5}
    result = sess.run(z, feed_dict=feed_dict)
    print result
```

Looking at this program, we observe that we need the following:
1. "graph" which defines what needs to be executed.
2. "placeholders" to accept new inputs.  
3. "session" in which the graph is executed.
4. "operations" to operate on data in the graph.
5. "variables" to store intermediate tensors.

Lets tackle each of these separately.

#### GRAPH
Graph is the superset that contains all the operations, placeholders and variables. To make things simple, we will store all operations, placeholders and variables as separate lists in the graph object. The python code for this is very simple and straight forward.

```
class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
```

Check out my <a href="https://github.com/amohant4/myFramework">github repo</a> for complete implementation.