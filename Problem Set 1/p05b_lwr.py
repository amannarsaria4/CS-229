import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau = 0.5)
    model.fit(x_train,y_train)
    # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept = True)
    y_pred = model.predict(x_val)
    mse = np.mean((y_pred - y_val)**2)
    print (mse)
    # Plot validation predictions on top of training set

    # No need to save predictions

    # Plot data
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_val, y_pred, 'ro')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        #make a weight matrix
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        
        #*** START CODE HERE ***
        from numpy.linalg import inv, norm
        m,n = x.shape
        d = x.reshape(m,1,n) - self.x
        d_normed = np.linalg.norm(d, ord = 2, axis = 2)
        w = np.exp(-1/(2*self.tau**2)* d_normed)

        W = np.apply_along_axis(np.diag, axis=1, arr=w)     
        
        # compute theta with the formula from (a)ii.
        # theta has shape (m,n,1)
        theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y.reshape((-1,1))
        
        # return an array of shape (m,)
        return np.ravel(x.reshape((m,1,n)) @ theta )
      
        # *** END CODE HERE ***
            
