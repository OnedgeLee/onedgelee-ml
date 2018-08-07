import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

""" 
BASIC FUNCTIONS
 """

def bias_rm(X):
    return (X - np.matmul(np.ones((X.shape[0], X.shape[0])), X) / X.shape[0])


def cov_mat(X):
    return np.matmul(bias_rm(X).T, bias_rm(X)) / X.shape[0]


def grad_desc(conv_func, var_iter, num_iter, var_min=-np.inf, var_max=np.inf, var_diff=1e-6, learning_rate=1e-3):
    if type(var_iter == np.ndarray):
        for i in range(num_iter):
            var_iter_next = np.empty_like(var_iter)
            for j in range(len(var_iter)):
                var_plus_diff = np.copy(var_iter)
                var_plus_diff[j] = var_iter[j] + var_diff
                slope = (conv_func(var_plus_diff) -
                         conv_func(var_iter)) / var_diff
                next_val = var_iter[j] + slope * learning_rate
                if (var_min < next_val) & (next_val < var_max):
                    var_iter_next[j] = next_val
            var_iter = var_iter_next
    return var_iter


def grad_asc(conv_func, var_iter, num_iter, var_min=-np.inf, var_max=np.inf, var_diff=1e-6, learning_rate=1e-3):
    if type(var_iter == np.ndarray):
        for i in range(num_iter):
            var_iter_next = np.empty_like(var_iter)
            for j in range(len(var_iter)):
                var_plus_diff = np.copy(var_iter)
                var_plus_diff[j] = var_iter[j] + var_diff
                slope = (conv_func(var_plus_diff) -
                         conv_func(var_iter)) / var_diff
                next_val = var_iter[j] + slope * learning_rate
                if (var_min < next_val) & (next_val < var_max):
                    var_iter_next[j] = next_val
            var_iter = var_iter_next
    return var_iter

""" 
PCA FUNCTIONS
 """

class PCA:

    def __init__(self, X):
        self.set_data(X)

    def set_data(self, X):
        self.X = X
        eig_val, eig_vec = np.linalg.eig(cov_mat(self.X))
        eig_val_desc = np.argsort(-eig_val)
        self.eig_val = eig_val[eig_val_desc]
        self.eig_vec = eig_vec[eig_val_desc]

    def pca_exec(self, reduce_dim):
        pcv = np.empty((0, self.X.shape[1]))
        for i in range(self.X.shape[1] - reduce_dim):
            pcv = np.vstack([pcv, self.eig_vec[i]])
        return np.matmul(self.X, pcv.T)

    def pca_auto_exec(self, target_ratio=0.9):
        eig_val = self.eig_val
        var_max = self.eig_val.sum()
        var_ratio = 1
        reduce_dim = -1
        while var_ratio > target_ratio:
            var_ratio -= eig_val[-1] / var_max
            eig_val = eig_val[:-1]
            reduce_dim += 1
        return self.pca_exec(reduce_dim)

""" 
SVM FUNCTIONS
 """

class SVM:

    def __init__(self, X, Y):
        self.set_data(X, Y)

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y.reshape((-1, 1))

    def linear_kernel(self, X):
        return np.matmul(X, X.T)

    def polynomial_kernel(self, X):
        return (np.matmul(X, X.T) * self.gamma + self.coef0) ** self.degree

    def sigmoid_kernel(self, X):
        return np.tanh((np.matmul(X, X.T) * self.gamma + self.coef0))

    def rbf_kernel(self, X):
        norm_square = np.linalg.norm(X, axis=1).reshape((-1, 1)) ** 2
        diff_norm = norm_square + norm_square.T - 2 * np.matmul(X, X.T)
        return np.exp(- diff_norm * self.gamma)

    def _lag_dual_func(self, lag_mult):
        return lag_mult.sum() - (self.linear_kernel(lag_mult) * self.linear_kernel(self.Y) * self.kernel(self.X)).sum() / 2

    def svm_exec(self, kernel="linear", coef0=0, gamma=1, degree=1, c_param=np.inf, learning_rate=1e-3):
        if kernel == "poly":
            self.kernel = self.polynomial_kernel
            print("polynomial kernel selected")
        elif kernel == "sigmoid":
            self.kernel = self.sigmoid_kernel
            print("sigmoid kernel selected")
        elif kernel == "rbf":
            self.kernel = self.rbf_kernel
            print("rbf (radial basis fucntion, gaussian) kernel selected")
        else:
            self.kernel = self.linear_kernel
            print("linear kernel selected")
        self.coef0 = coef0
        self.gamma = gamma
        self.degree = degree
        self.c_param = c_param
        lag_mult = np.ones((self.X.shape[0], 1))
        opt_lag_mult = grad_asc(self._lag_dual_func,
                                lag_mult, 10000, var_min=0, var_max=self.c_param, learning_rate=learning_rate)
        self.W = (opt_lag_mult * self.Y * self.X).sum(axis=0).reshape((-1, 1))
        self.b = self.Y - np.matmul(self.X, self.W)
        return self.model

    def model(self, X):
        model_result = np.matmul(X, self.W) + self.b
        for i in range(len(model_result)):
            if model_result[i] > 0:
                model_result[i] = 1
            else:
                model_result[i] = -1
        return model_result


""" 
PCA EXECUTE

data = np.genfromtxt('data-03-diabetes.csv', delimiter=',')
X = data[:, :8]
pca = PCA(X)
X_pca = pca.pca_auto_exec()
print(X_pca) 


SVM EXECUTE

data = np.genfromtxt('data-03-diabetes.csv', delimiter=',')
X = data[:, :8]
Y = data[:, -1]
svm = SVM(X, Y)
model = svm.svm_exec(learning_rate=0.01, kernel='sigmoid')
Y = Y.reshape((-1, 1))
print(np.hstack((model(X), Y)))

 """


# def logistic(X, W):
#     return 1 / (1 + np.exp(-np.matmul(X, W)))

# def softmax(X, W):
#     return np.exp(np.matmul(X, W)) / np.exp(np.matmul(X, W)).sum(axis = 0)

# def lda(X):

# def naive_bayes(X):

# def lstm(X):
