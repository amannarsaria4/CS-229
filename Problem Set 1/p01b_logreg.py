import numpy as np
import util

from linear_model import LinearModel
#from linalg import inv,norm

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    # *** START CODE HERE ***
    
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    y_pred = model.predict(x_eval)
    util.plot(x_eval, y_eval, model.theta, '{}.png'.format(pred_path))
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        #shape of x
        m,n = x.shape

        #initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)

        g = lambda x: 1/(1 + np.exp(-x))

        #update theta
        while True:
            theta = self.theta

            x_theta = theta.dot(x.T)
            
            J = (-1/m)*(y - g(x_theta)).dot(x)

            H = (1/m)*(g(self.theta)).dot(1-g(self.theta))*x.T.dot(x)
            H_inv = np.linalg.inv(H)

            #update rule
            self.theta = theta - H_inv.dot(J)

            #break if loss less than eps
            if np.linalg.norm(self.theta - theta, ord = 1) < self.eps:
                #print(self.theta)
                """*** Why two values of self.theta are printed??"""
                
                break
            
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        g = lambda x: 1/(1 + np.exp(-x))
        th = self.theta
        y_pred = g(th.dot(x.T))

        return y_pred
        # *** END CODE HERE ***
