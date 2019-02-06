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

#### PLACEHOLDERS
Placeholders are for getting in new data. So at the start all we need to know is the shape of the placeholder and the actual data in it will be filled at the time of execution. So we create a class in which the constructor needs only the shape of it (shapes are critical when we can to determine sizes of intermediate variables based on other variables). Note that everytime we create a placeholder we append it to the list of placeholders in the graph object (em._default_graph). 
```
import emulator as em

class Placeholder():
    def __init__(self, shape):
        self.output_nodes = []
        self._shape = shape
        em._default_graph.placeholders.append(self)

    @property
    def shape(self):
        return self._shape  
```

#### VARIABLES
Variables are objects whose value can be altered during execution. We need to provide them with an initial value aswell. As as we did before, anytime we instantiate a variable object we need to append it to the list containing all variables in the graph. 

```
import emulator as em

class Variable():
    def __init__(self, shape, initial_value = None):
        """
        Constructor
        """
        self._shape = shape
        self.value = initial_value
        self.output_nodes = []
        em._default_graph.variables.append(self)

    @property
    def shape(self):
        """
        API to get the shape of a variable
        return type is python list
        """
        return self._shape

    def load(self, val):
        """
        API to load values to a variable.
        shape of the new value should be same as
        the original shape of the variable.
        Only supports numpy arrays as val.
        """
        assert list(val.shape) == self._shape
        self.value = val
```

Check out my <a href="https://github.com/amohant4/myFramework">github repo</a> for complete implementation.