import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        # print(type(x))
        # x is also a nn.Constant
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        # use nn.as_scalar to convert a scalar Node into a Python floating-point number
        # print(self.run(x), nn.as_scalar(self.run(x)))
        val = nn.as_scalar(self.run(x))
        # == 0 is the positive class
        if val >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        # print(type(dataset))
        # <class 'backend.Dataset'>

        # repeatedly loop over the data set and make updates on examples that are misclassified
        while True:
            converge = True

            # When training a perceptron or neural network, you will be passed a dataset object.
            # You can retrieve batches of training examples by calling dataset.iterate_once(batch_size):
            batch_size = 1
            for x, y in dataset.iterate_once(batch_size):
                # print(x, y)
                label = nn.as_scalar(y)
                if self.get_prediction(x) != label:
                    # Use the update method of the nn.Parameter class to update the weights
                    # parameter.update(direction, multiplier)
                    self.get_weights().update(x, label)
                    converge = False
            
            if converge:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # make the nn has 1 hidden layer, with `neural_num` neurals
        neural_num = 64
        self.w1 = nn.Parameter(1, neural_num)
        self.b1 = nn.Parameter(1, neural_num)
        # choose ReLU to be the activate function 
        self.activate1 = nn.ReLU

        # the output layer
        self.w2 = nn.Parameter(neural_num, 1)
        self.b2 = nn.Parameter(1, 1)

        # choose L2 loss to be the loss function
        self.loss_function = nn.SquareLoss

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # print(x.data.shape)

        # hidden layer
        f_x = nn.Linear(x, self.w1)
        f_x = nn.AddBias(f_x, self.b1)
        f_x = self.activate1(f_x)

        # output layer
        f_x = nn.Linear(f_x, self.w2)
        f_x = nn.AddBias(f_x, self.b2)

        return f_x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return self.loss_function(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # sample_point = 0
        # for x, y in dataset.iterate_once(batch_size=1):
        #     sample_point += 1
        # print("sample_point = ", sample_point)

        # sample_point =  200
        # AssertionError: Dataset size 200 is not divisible by batch size 32
        batch_size = 20
        last_loss = float('inf')
        converge = False

        start_lr = 6e-2
        end_lr = 1e-2
        gamma = (end_lr / start_lr) ** (1 / 20)
        iter = 1

        lr = start_lr
        while converge == False:

            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                loss_num = nn.as_scalar(loss)
                # print(loss_num)
                # print('lr = ', lr)

                # gets a loss of 0.02 or better
                # print('loss change: ', abs(last_loss - loss_num))
                if loss_num < 0.005:
                    converge = True
                    break
                
                last_loss = loss_num

                # back propagation
                grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad_w1, -lr)
                self.b1.update(grad_b1, -lr)
                self.w2.update(grad_w2, -lr)
                self.b2.update(grad_b2, -lr)

                if iter <= 2:
                    lr *= gamma
            iter += 1

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # 2 hidden layer
        # 28 * 28 = 784 neurals
        neural_num = 256
        self.w1 = nn.Parameter(784, neural_num)
        self.b1 = nn.Parameter(1, neural_num)
        self.activate1 = nn.ReLU

        self.w2 = nn.Parameter(neural_num, neural_num // 2)
        self.b2 = nn.Parameter(1, neural_num // 2)
        self.activate2 = nn.ReLU

        # total 10 classes
        self.w3 = nn.Parameter(neural_num // 2, 10)
        self.b3 = nn.Parameter(1, 10)

        # Do not put a ReLU activation after the last layer of the network

        # use nn.SoftmaxLoss as classification loss
        self.loss_function = nn.SoftmaxLoss

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        
        f_x = nn.Linear(x, self.w1)
        f_x = nn.AddBias(f_x, self.b1)
        f_x = self.activate1(f_x)

        f_x = nn.Linear(f_x, self.w2)
        f_x = nn.AddBias(f_x, self.b2)
        f_x = self.activate2(f_x)

        # output layer
        f_x = nn.Linear(f_x, self.w3)
        f_x = nn.AddBias(f_x, self.b3)

        return f_x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return self.loss_function(self.run(x), y)
    
    # for highlight 
    import backend 
    def train(self, dataset : backend.DigitClassificationDataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        import math

        batch_size = 60
        converge = False
        start_lr = 8e-1
        end_lr = 1e-1
        gamma = (end_lr / start_lr) ** (1 / 2000)
        iter = 1

        lr = start_lr
        while converge == False:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)

                var = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(var[0], -lr)
                self.b1.update(var[1], -lr)
                self.w2.update(var[2], -lr)
                self.b2.update(var[3], -lr)
                self.w3.update(var[4], -lr)
                self.b3.update(var[5], -lr)

                # it may help to set a slightly higher stopping threshold on validation accuracy, such as 97.5% or 98%
                if dataset.get_validation_accuracy() > 0.975:
                    converge = True
                    break
                if iter <= 2:
                    lr *= gamma
            iter += 1

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # The hidden size should be sufficiently large
        neural_num = 512
        
        # W
        self.w1 = nn.Parameter(self.num_chars, neural_num)

        # W_hidden
        self.w2 = nn.Parameter(neural_num, neural_num)

        # there are total 5 languages
        self.w3 = nn.Parameter(neural_num, 5)

        # the classification task
        self.loss_function = nn.SoftmaxLoss

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # z0 = x0 * W
        z0 = nn.Linear(xs[0], self.w1)
        hi = z0

        for i in range(1, len(xs)):
            # zi = xi * W + hi * W_hidden
            # you should replace a computation of the form z = nn.Linear(x, W) with a computation of the form 
            # z = nn.Add(nn.Linear(x, W), nn.Linear(h, W_hidden))
            zi = nn.Add(nn.ReLU(nn.Linear(xs[i], self.w1)), nn.ReLU(nn.Linear(hi, self.w2)))
            hi = zi
        
        # output layer
        output = nn.Linear(hi, self.w3)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return self.loss_function(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # Dataset size 17500
        batch_size = 100
        
        start_lr = 8e-1
        end_lr = 1e-1
        gamma = (end_lr / start_lr) ** (1 / 350)
        lr = start_lr
        iter = 1
        converge = False
        
        lr = start_lr
        while converge == False:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)

                var = nn.gradients(loss, [self.w1, self.w2, self.w3])
                self.w1.update(var[0], -lr)
                self.w2.update(var[1], -lr)
                self.w3.update(var[2], -lr)

                # reference implementation can still correctly classify over 89% of the validation set
                if dataset.get_validation_accuracy() > 0.89:
                    converge = True
                    break
                
                if iter <= 2:
                    lr *= gamma
            iter += 1

