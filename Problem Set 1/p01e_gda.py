import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = False)
    y_pred = model.predict(x_eval)
    # Use np.savetxt to save outputs from validation set to pred_path
    util.plot(x_eval, y_eval,model.theta, '{}.png'.format(pred_path))
    rms = (((y_eval - y_pred)**2).sum()).mean()
    print(rms)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        # Find phi, mu_0, mu_1, and sigma
        #phi
        val = 0
        for i in range(m):
            if y[i] == 1:
                val+=1
        phi = val/m
        #mu_0
        mu_0 = (x[y==0]).sum(axis = 0)/(m-val)
        #mu_1
        mu_1 = (x[y==1]).sum(axis = 0)/(val)
        #sigma
        diff = x.copy()
        diff[y==0] -= mu_0
        diff[y==1] -= mu_1
        sigma = (1/m)*(diff.T).dot(diff)
        """
        phi = (y == 1).sum() / m
        mu_0 = x[y == 0].sum(axis=0) / (y == 0).sum()
        mu_1 = x[y == 1].sum(axis=0) / (y == 1).sum()
        diff = x.copy()
        diff[y == 0] -= mu_0
        diff[y == 1] -= mu_1
        sigma = (1 / m) * diff.T.dot(diff)"""
        
        print("phi = {}, mu_0 = {}, mu_1 = {}, sigma = {}".format(phi,mu_0,mu_1,sigma))
        #print(phi,"/n" mu_0,"/n" mu_1,"/n" sigma)
        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        theta0 = 0.5 * (mu_0.T.dot(sigma_inv).dot(mu_0) - mu_1.T.dot(sigma_inv).dot(mu_1)) - np.log((1 - phi) / phi)
        theta = ((mu_1 - mu_0).T).dot((np.linalg.inv(sigma)))
        theta0 = np.array([theta0])
        theta = np.hstack([theta0, theta])
        self.theta = theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,)
        """
        # *** START CODE HERE ***
        x = util.add_intercept(x)
        m,n = x.shape
        g = lambda z: 1/(1 + np.exp(-z))
        test = g((x.dot(self.theta)))
        y_pred = test.copy()
        for i in range(m):
            if y_pred[i] >0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
        #y_pred = (prob >= 0.5).astype(np.int)
        #return y_pred
        # *** END CODE HERE


