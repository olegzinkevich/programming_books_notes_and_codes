Neural Language Models
In Chapter 7, we introduced the notion that a language model could be learned from
a sufficiently large and domain-specific corpus using a probabilistic model. This is
known as a symbolic model of language. In this chapter we consider a different
approach: the neural or connectionist language model.

The connectionist model of language argues that the units of language interact with
each other in meaningful ways that are not necessarily encoded by sequential context.
For instance, the contextual relationship of words may be sequential, but may also be
separated by other phrases,

Since many interactions are not directly interpretable, some intermediary representation
must be used to describe the connections, and connectionist models tend to use
artificial neural networks (ANNs) or Bayesian networks to learn the underlying relationships.

Neural networks comprise a very broad and variegated family of models, but are
more or less all evolved from the perceptron, a linear classification machine developed
in the late 1950s by Frank Rosenblatt at Cornell and modeled on the learning
behavior of the human brain.

At the core of the neural network model family are several components, as shown in
Figure 12-2—an input layer, a vectorized first representation of the data, a hidden
layer consisting of neurons and synapses, and an output layer containing the predicted
values. Within the hidden layer, synapses are responsible for transmitting signals
between the neurons, which rely on a nonlinear activation function to buffer those
incoming signals. The synapses apply weights to incoming values, and the activation
function determines if the weighted inputs are sufficiently high to activate the neuron
and pass the values on to the next layer of the network

In a feedforward network, signals travel from the input to the output layer in a single
direction. In more complex architectures like recurrent and recursive networks, signal
buffering can combine or recur between the nodes within a layer.

Backpropagation is the process by which error, computed at the final layer of the network,
is communicated back through the layers to incrementally adjust the synapse
weights and improve accuracy in the next training iteration. After each iteration, the
model calculates the gradient of the loss function to determine the direction in which
to adjust the weights