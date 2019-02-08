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

1. `graph` which defines what needs to be executed.
2. `placeholders` to accept new inputs.  
3. `session` in which the graph is executed.
4. `operations` to operate on data in the graph.
5. `variables` to store intermediate tensors.

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
Placeholders are for getting in new data. So at the start all we need to know is the shape of the placeholder and the actual data in it will be filled at the time of execution. So we create a class in which the constructor needs only the shape of it (shapes are critical when we can to determine sizes of intermediate variables based on other variables). Since its a node in the graph it will have an attribute called output_nodes to keep all the nodes to which this placeholder provides data to. Note that everytime we create a placeholder we append it to the list of placeholders in the graph object (em._default_graph).
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
Variables are objects whose value can be altered during execution. We need to provide them with an initial value aswell. As as we did before, anytime we instantiate a variable object we need to append it to the list containing all variables in the graph. Similarliy it will have an attribute called output_nodes to keep all the nodes to which this placeholder provides data to. We will provide an additional method to allow loading values into the variables.

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

Before implementing operators and session, lets take a look at an simple example. 
<div class="imgcap">
<img src="/assets/Creating-your-own-dl-framework/graph.png" width="60%">
</div>

#### OPERATIONS
Operations are responsible for modifying values in variables and placeholders to produce outputs. Here we will implement the base class for all operators. So methods like `shape` and `compute` which depend on the actual operation are to be implemented in the inherited subclass.

```
import emulator as em

class Operation(object):
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []

        # For every node in the input, we append this operation (self) to the list of
        # the consumers of the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)

        # There will be a global default graph (TensorFlow works this way)
        # We will then append this particular operation
        # Append this operation to the list of operations in the currently active default graph
        em._default_graph.operations.append(self)

    @property
    def shape(self):
        raise NotImplementedError('Must be implemented in the subclass')

    def compute(self):
        """
        Must be implemented in the sub class.
        """
        raise NotImplementedError('Must be implemented in the subclass')
```

#### SESSION

```
import numpy as np
from .operation import Operation
from .placeholder import Placeholder
from .variable import Variable

class Session:
    def traverse_postorder(self,operation):
        nodes_postorder = []
        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)
        recurse(operation)
        return nodes_postorder

    def run(self, operation, feed_dict = {}):
        nodes_postorder = self.traverse_postorder(operation)
        for node in nodes_postorder:
            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output
```
Check out my <a href="https://github.com/amohant4/myFramework">github repo</a> for complete implementation.