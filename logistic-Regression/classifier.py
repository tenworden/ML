import numpy as np


class LogisticRegression:
    """ A basic classfier """

    def __init__(self):
        self.W = None
        self.velocity = None

    def w_init(self, dim, num_classes):
        return np.random.randn(dim)
    
    def loss(self, X_batch, y_batch, reg):
        from utils import logistic_regression_loss
        
        return logistic_regression_loss(self.W, X_batch, y_batch, reg)
    
    def predict(self, X):

        y_pred = np.zeros(X.shape[0])

        from utils import sigmoid

        ########################################################################
        # Implement this method. Store the predicted labels in y_pred.         #
        ########################################################################
        ########################################################################
        #                     START OF YOUR CODE                               #
        ########################################################################
        prob = sigmoid(np.matmul(X, self.W))
        
        y_pred = np.array([1 if p > 0.5 else 0 for p in prob])
        # raise NotImplementedError
        ########################################################################
        #                    END OF YOUR CODE                                  #
        ########################################################################

        return y_pred

    def train(
        self, X, y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        optim="SGD",
        momentum=0.5,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent(SGD).
        Batch size is set to 200, learning rate to 0.001, regularization rate to 0.00001.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.
        - y: a numpy array of shape (N,) containing training labels; y[i] = c
            means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) L2 regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - optim: the optimization method, the default optimizer is 'SGD' and
            feel free to add other optimizers.
        - verbose: (boolean) if true, print progress during optimization.

        Returns:
        - loss_history: a list containing the value of the loss function of each iteration.
        """

        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is the number of classes
        num_classes = np.max(y) + 1

        # Initialize W and velocity(for SGD with momentum)
        if self.W is None:
            self.W = 0.001 * self.w_init(dim, num_classes)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.W)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            ########################################################################
            # TODO:                                                                #
            # Sample batch_size elements from the training data and their          #
            # corresponding labels to use in this round of gradient descent.       #
            # Store the data in X_batch and their corresponding labels in          #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)  #
            # and y_batch should have shape (batch_size,)                          #
            #                                                                      #                        
            ########################################################################
            ########################################################################
            #                     START OF YOUR CODE                               #
            ########################################################################
            for i in range((num_train-1)//batch_size + 1):
                
                start_i = i*batch_size
                end_i = start_i + batch_size
                X_batch = X[start_i:end_i]
                y_batch = y[start_i:end_i]
                
            # raise NotImplementedError
            ########################################################################
            #                       END OF YOUR CODE                               #
            ########################################################################

            ########################################################################
            # TODO:                                                                #
            # Update the weights using the gradient and the learning rate.         #
            #                                                                      #
            # Hint: use self.loss() to compute the loss and gradient               #
            ########################################################################
            ########################################################################
            #                     START OF YOUR CODE                               #
            ########################################################################
                loss, dw = self.loss(X_batch, y_batch, reg)
                self.W -= learning_rate*dw
                loss_history.append(loss)
            ########################################################################
            #                    END OF YOUR CODE                                  #
            ########################################################################
            
            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history
    