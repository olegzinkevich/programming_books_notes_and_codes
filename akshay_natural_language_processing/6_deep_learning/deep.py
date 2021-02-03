# Each neuron takes input, does
# some kind of manipulation within the neuron, and produces an output
# that is closer to the expected output (in the case of labeled data). What happens within the neuron is what we are interested in: to get to
# the most accurate results. In very simple words, it’s giving weight to every
# input and generating a function to accumulate all these weights and pass it
# onto the next layer, which can be the output layer eventually.
#
# The network has 3 components:
# • Input layer
# • Hidden layer/layers
# • Output layer
#
# The functions can be of different types based on the problem or the
# data. These are also called activation functions. Below are the types.
# • Linear Activation functions: A linear neuron takes a
# linear combination of the weighted inputs; and the
# output can take any value between -infinity to infinity.
# • Nonlinear Activation function: These are the most used
# ones, and they make the output restricted between
# some range:
# • Sigmoid or Logit Activation Function: Basically,
# it scales down the output between 0 and 1
# by applying a log function, which makes the
# classification problems easier.
# • Softmax function: Softmax is almost similar to
# sigmoid, but it calculates the probabilities of the
# event over ‘n’ different classes, which will be useful
# to determine the target in multiclass classification
# problems.
# • Tanh Function: The range of the tanh function is
# from (-1 to 1), and the rest remains the same as
# sigmoid.
# • Rectified Linear Unit Activation function: ReLU
# converts anything that is less than zero to zero. So,
# the range becomes 0 to infinity.
#
# Convolutional Neural Networks
#
# Convolutional Neural Networks (CNN) are similar to ordinary neural
# networks but have multiple hidden layers and a filter called the
# convolution layer. CNN is successful in identifying faces, objects, and traffic signs and also used in self-driving cars
#
# CNN is a special case of a neural network with an input layer, output layer,
# and multiple hidden layers. The hidden layers have 4 different procedures
# to complete the network.
#
# The Convolution (сверточный ) layer is the heart of a Convolutional Neural Network,
# which does most of the computational operations. The name comes from
# the “convolution” operator that extracts features from the input image.
# These are also called filters (Orange color 3*3 matrix). The matrix formed
# by sliding the filter over the full image and calculating the dot product
# between these 2 matrices is called the ‘Convolved Feature’ or ‘Activation
# Map’ or the ‘Feature Map’. Suppose that in table data, different types of
# features are calculated like “age” from “date of birth.” The same way here
# also, straight edges, simple colors, and curves are some of the features that
# the filter will extract from the image.
#
# During the training of the CNN, it learns the numbers or values present
# inside the filter and uses them on testing data. The greater the number
# of features, the more the image features get extracted and recognize all
# patterns in unseen images.
#
# Backpropagation (обратное развертывание): Training the Neural Network
# In normal neural networks, you basically do Forward Propagation to get
# the output and check if this output is correct and calculate the error. In
# Backward Propagation, we are going backward through your network that
# finds the partial derivatives of the error with respect to each weight
#
# As per the feed forward rule, weights are
# randomly assigned and complete the first iteration of training and also
# output random probabilities. After the end of the first step, the network
# calculates the error at the output layer using  Now, your backpropagation starts to calculate the gradients of the
# error with respect to all weights in the network and use gradient descent
# to update all filter values and weights, which will eventually minimize
# the output error. The filter matrix and connection weights will get updated for each run. The
# whole process is repeated for the complete training set until the error is
# minimized
#
#
#
#
# Recurrent Neural Networks
#
# CNNs are basically used for computer vision problems but fail to solve
# sequence models. Sequence models are those where even a sequence
# of the entity also matters. For example, in the text, the order of the words
# matters to create meaningful sentences. This is where RNNs come into the
# picture and are useful with sequential data because each neuron can use
# its memory to remember information about the previous step.
#
# It is quite complex to understand how exactly RNN is working. If you
# see the above figure, the recurrent neural network is taking the output from
# the hidden layer and sending it back to the same layer before giving the
# prediction. If we just discuss the hidden layer, it’s not only taking input from
# the hidden layer, but we can also add another input to the same hidden
# layer. Now the backpropagation happens like any other previous training
# we have seen; it’s just that now it is dependent on time.
#
# Long Short-Term Memory (LSTM)
# LSTMs are a kind of RNNs with betterment in equation and
# backpropagation, which makes it perform better. LSTMs work almost
# similarly to RNN, but these units can learn things with very long time gaps,
# and they can store information just like computers.
#
# The algorithm learns the importance of the word or character through
# weighing methodology and decides whether to store it or not. For this, it
# uses regulated structures called gates that have the ability to remove or add
# information to the cell. These cells have a sigmoid layer that decides how
# much information should be passed. It has three layers, namely “input,”
# “forget,” and “output” to carry out this process.


