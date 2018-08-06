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

# 수렴시 어떤 차이가 있을까? 생각해 볼 것
# def grad_desc2(conv_func, var_iter, num_iter, var_diff=1e-6, learning_rate=1e-3):
#     if type(var_iter == np.ndarray):
#         for i in range(num_iter):
#             for i in range(len(var_iter)):
#                 var_plus_diff = np.copy(var_iter)
#                 var_plus_diff[i] = var_iter[i] + var_diff
#                 slope = (conv_func(var_plus_diff) -
#                     conv_func(var_iter)) / var_diff
#                 var_iter[i] = var_iter[i] - slope * learning_rate
#     return var_iter


""" 
SVM FUNCTIONS
 """


class SVM:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y.reshape((20, 1))

    def _lag_dual_func(self, lag_mult):
        return lag_mult.sum() - (np.matmul(lag_mult, lag_mult.T) * np.matmul(self.Y, self.Y.T) * np.matmul(self.X, self.X.T)).sum() / 2

    def svm_calc(self, learning_rate=1e-3):
        lag_mult = np.ones((self.X.shape[0], 1))
        opt_lag_mult = grad_asc(self._lag_dual_func,
                                lag_mult, 10000, var_min=0, learning_rate=learning_rate)
        print((opt_lag_mult * self.Y).sum(axis=0))
        self.W = (opt_lag_mult * self.Y * self.X).sum(axis=0).reshape((-1, 1))
        self.b = self.Y - np.matmul(self.X, self.W)
        return self.W, self.b


class kernel_SVM:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y.reshape((20, 1))

    # def linear_kernel(X):

    def _lag_dual_func(self, lag_mult):
        return lag_mult.sum() - (np.matmul(lag_mult, lag_mult.T) * np.matmul(self.Y, self.Y.T) * np.matmul(self.X, self.X.T)).sum() / 2

    def svm_calc(self, c_param=np.inf, learning_rate=1e-3):
        lag_mult = np.ones((self.X.shape[0], 1))
        opt_lag_mult = grad_asc(self._lag_dual_func,
                                lag_mult, 10000, var_min=0, var_max=c_param, learning_rate=learning_rate)
        print(opt_lag_mult)
        print((opt_lag_mult * self.Y).sum(axis=0))
        self.W = (opt_lag_mult * self.Y * self.X).sum(axis=0).reshape((-1, 1))
        self.b = self.Y - np.matmul(self.X, self.W)
        return self.W, self.b


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

    def pca_reduce(self, reduce_dim):
        pcv = np.empty((0, self.X.shape[1]))
        for i in range(self.X.shape[1] - reduce_dim):
            pcv = np.vstack([pcv, self.eig_vec[i]])
        return np.matmul(self.X, pcv.T)

    def pca_auto_reduce(self, target_ratio=0.9):
        eig_val = self.eig_val
        var_max = self.eig_val.sum()
        var_ratio = 1
        reduce_dim = -1
        while var_ratio > target_ratio:
            var_ratio -= eig_val[-1] / var_max
            eig_val = eig_val[:-1]
            reduce_dim += 1
        return self.pca_reduce(reduce_dim)


# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# b = np.array([[1, -0.5], [2, -0.7], [3, -1], [4, -1.3], [-1, -0.5],
#               [-2, 0.7], [-3, 1], [-4, 1.3], [1, 0.7], [1, 0.4], [-2.5, -0.3]])
# pp.pprint(pca_cal(b))
# X2 = b[:,[0]]
# Y2 = b[:,[1]]
# plt.scatter(X2, Y2)
# plt.show()


data = np.genfromtxt('data-03-diabetes.csv', delimiter=',')
X = data[:, :8]
Y = data[:, -1]
svm = kernel_SVM(X, Y)
a = svm.svm_calc(learning_rate=0.01)
print(a)


# def logistic(X, W):
#     return 1 / (1 + np.exp(-np.matmul(X, W)))

# def softmax(X, W):
#     return np.exp(np.matmul(X, W)) / np.exp(np.matmul(X, W)).sum(axis = 0)

# def lda(X):

# def naive_bayes(X):

# def lstm(X):
