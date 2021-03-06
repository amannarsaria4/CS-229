import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    model = PoissonRegression(step_size = lr, max_iter = 1000000000)
    model.fit(x_train,y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(n)

        g = lambda x: np.exp(x)
        for i in range(self.max_iter):
            th = self.theta
            grad = (1/m)*(y - g(x.dot(th))).dot(x)
            self.theta = th + self.step_size*grad
            print(th)
            if(np.linalg.norm(self.theta - th, ord = 1) < self.eps):
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        g = lambda x: np.exp(x)
        return g(x.dot(self.theta))
        # *** END CODE HERE ***
