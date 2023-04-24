# Neural-network-builder-in-python-from-scratch
Implemented a neural network class which includes , prediction , front prop , back prop , training the network etc

Method: build()
The build() method initializes the model architecture. It takes two parameters, x and activation. x is a list of integers representing the number of nodes in each layer of the neural network. activation is a list of integers representing the activation function to be used in each layer. If no activation function is specified, the method sets the activation to be a list of zeros.

The method returns the model object, which is then used by other methods in the class.

Method: activate()
The activate() method computes the activation of a layer of the neural network. It takes two parameters, current and i. current is the input to the layer, and i is the index of the layer whose activation function is to be computed.

The method returns the computed activation function.

Method: predict()
The predict() method takes two parameters, input and pass_network. input is the input data to the neural network, and pass_network is a flag indicating whether to return the entire network or just the output. The method returns the output of the neural network.

If pass_network is set to 1, the method returns a tuple containing the output of the network, the value of z, and the activation function of the network.

Method: train()
The train() method trains the neural network. It takes several parameters, including x, y, lr, epochs, batch_size, and op. x is the input data to the neural network, y is the target output, lr is the learning rate, epochs is the number of epochs to train for, batch_size is the size of the batches to be used during training, and op is the type of optimization to be used.

During training, the method computes the loss and backpropagates the error through the network to update the weights and biases. The method returns the loss.

Method: delta_activation()
The delta_activation() method computes the derivative of the activation function. It takes two parameters, dzi and i. dzi is the output of the layer, and i is the index of the layer.

The method returns the derivative of the activation function.

Method: back_prop()
The back_prop() method backpropagates the error through the neural network. It takes four parameters, x, y, lr, and op. x is the input data, y is the target output, lr is the learning rate, and op is the type of optimization to be used.

The method computes the error and backpropagates it through the network to update the weights and biases. The method returns the loss.

Method: mean_squared_error()
The mean_squared_error() method computes the mean squared error of the neural network. It takes two parameters, x and y. x is the input data, and y is the target output.

The method returns the mean squared error.

Method: get_bias()
The get_bias() method returns the bias of the neural network.

Method: get_weights()
The get_weights() method returns the weights of the neural network.

Method: sigmoid()
The sigmoid() method computes the sigmoid function of a given value. It takes one parameter, x.

The method returns the sigmoid value of x.

Note : The shape of first layer of the neural network that you are buiding should match the shape of the input.
       It doesn't have complex optimizers so its not suitable for complex problem statements.
