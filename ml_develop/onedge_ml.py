import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

""" 
BASIC FUNCTIONS
 """


def grad_desc(conv_func, var_iter, num_iter, var_min=-np.inf, var_max=np.inf, var_diff=1e-6, learning_rate=1e-3):
    print("gradient descending...")
    var_iter = np.array(var_iter)
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


def one_hot(Y, depth):
    Y = np.array(Y)
    one_hot = np.eye(depth)[Y.reshape(-1)].reshape(Y.shape + (depth,))
    return one_hot
    # reshape : multi-dimension을 상정


def maha_dist(x, X):
    unbiased_x = x - np.mean(X, axis=0, keepdims=True)
    square = np.sum(np.matmul(unbiased_x, np.linalg.inv(
        np.cov(X, rowvar=False))) * unbiased_x, axis=1, keepdims=True)
    return np.sqrt(square)
    # covariance matrix가 singular matrix일 경우 구할 수 없다...covariance matrix가 singular matrix라는 의미는? 이럴때는 gaussian pdf로 가정 불가?


def log_gauss_dist(x, X):
    return - ((maha_dist(x, X) ** 2) - np.log(np.linalg.det(np.cov(X, rowvar=False))) - X.shape[1] * np.log(2 * np.pi)) / 2


""" 
PCA FUNCTIONS
 """


class PCA:

    def __init__(self, X):
        self.set_data(X)

    def set_data(self, X):
        self.X = np.array(X)
        eig_val, eig_vec = np.linalg.eig(np.cov(self.X, rowvar=False))
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
        self.X = np.array(X)
        self.Y = np.array(Y)

    def _linear_kernel(self, X):
        return np.matmul(X, X.T)

    def _polynomial_kernel(self, X):
        return (np.matmul(X, X.T) * self.gamma + self.coef0) ** self.degree

    def _sigmoid_kernel(self, X):
        return np.tanh((np.matmul(X, X.T) * self.gamma + self.coef0))

    def _rbf_kernel(self, X):
        norm_square = np.linalg.norm(X, axis=1).reshape((-1, 1)) ** 2
        diff_norm = norm_square + norm_square.T - 2 * np.matmul(X, X.T)
        return np.exp(- diff_norm * self.gamma)

    def _lag_dual_func(self, lag_mult):
        return -(lag_mult.sum() - (self._linear_kernel(lag_mult) * self._linear_kernel(self.Y) * self.kernel(self.X)).sum() / 2)

    def svm_exec(self, kernel="linear", coef0=0, gamma=1, degree=1, c_param=np.inf, learning_rate=1e-3, num_iter=1000):
        if kernel == "poly":
            self.kernel = self._polynomial_kernel
            print("polynomial kernel selected")
        elif kernel == "sigmoid":
            self.kernel = self._sigmoid_kernel
            print("sigmoid kernel selected")
        elif kernel == "rbf":
            self.kernel = self._rbf_kernel
            print("rbf (radial basis fucntion, gaussian) kernel selected")
        else:
            self.kernel = self._linear_kernel
            print("linear kernel selected")
        self.coef0 = coef0
        self.gamma = gamma
        self.degree = degree
        self.c_param = c_param
        lag_mult = np.ones((self.X.shape[0], 1))
        opt_lag_mult = grad_desc(self._lag_dual_func,
                                 lag_mult, num_iter, var_min=0, var_max=self.c_param, learning_rate=learning_rate)
        self.W = (opt_lag_mult * self.Y * self.X).sum(axis=0).reshape((-1, 1))
        self.b = self.Y - np.matmul(self.X, self.W)
        return self.model

    def model(self, X):
        print("Weight : ")
        pp.pprint(self.W)
        print("bias : ")
        pp.pprint(self.b)
        model_result = np.matmul(X, self.W) + self.b
        for i in range(len(model_result)):
            if model_result[i] > 0:
                model_result[i] = 1
            else:
                model_result[i] = -1
        return model_result

# gradient descent시 lagrange multiplier를 데이터 개수만큼 구해야 하는데, 각각이 독립이 아니므로 데이터 개수만큼의 차원에 대해 gradient descent를 하는 것과 같음
# 때문에 gradient descent가 데이터 개수에 비례하여 오래걸리게 됨

# multiclass에 대해서도 만들어보자 - one-hot encoding을 one versus all로 변형시키면 가능할 것

# class multiclass_SVM:


""" 
QDA, LDA FUNCTIONS
 """


class QDA:
    """
    x를 독립변수, p(x|y=c) 를 multivariable gaussian distribution으로 가정
    p(x|y=c) = multivar_gauss_dist
    p(y=c|x) = (p(x|y=c) * p(y=c)) / (p(x|y=k) * p(y=k)).sum(iter=k)
     """

    def __init__(self, X, Y):
        self.set_data(X, Y)

    def set_data(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.nb_classes = self.Y.max() + 1
        self.X_c = []
        for i in range(self.X.shape[0]):
            for j in range(self.nb_classes):
                self.X_c.append(list())
                if self.Y[i] == j:
                    self.X_c[j].append(self.X[i])

    def qda_exec(self, X):
        self.mean_c = []
        self.inv_cov_c = []
        self.log_det_cov_c = []
        self.log_prior_c = []
        for i in range(self.nb_classes):
            self.X_c[i] = np.array(self.X_c[i])
            self.mean_c.append(np.mean(self.X_c[i], axis=0, keepdims=True))
            self.inv_cov_c.append(np.linalg.inv(
                np.cov(self.X_c[i], rowvar=False)))
            sign, logdet = np.linalg.slogdet(np.cov(self.X_c[i], rowvar=False))
            self.log_det_cov_c.append(sign * logdet)
            self.log_prior_c.append(
                np.log(self.X_c[i].shape[0] / self.X.shape[0]))
        return self.model

    def model(self, X):
        log_posterior = []
        sq_maha_dist_c = []
        for j in range(self.nb_classes):
            unbiased_X = X - self.mean_c[j]
            sq_maha_dist = np.sum(
                np.matmul(unbiased_X, self.inv_cov_c[j]) * unbiased_X, axis=1, keepdims=True)

            sq_maha_dist_c.append(sq_maha_dist)

            log_posterior.append(
                self.log_prior_c[j] - (self.log_det_cov_c[j]/2) - (sq_maha_dist/2))
            # print( - (self.log_det_cov_c[j]/2))
        log_posterior = np.array(log_posterior)
        model_result = np.argmax(log_posterior, axis=0)

        sq_maha_dist_c = np.array(sq_maha_dist_c)
        print(-sq_maha_dist_c[:,0])
        # print(self.log_prior_c[j])
        # print(self.log_det_cov_c[j]/2)
        # print(posterior[:,0])
        print(unbiased_X)
        return model_result

        # log posterior이 튄다 : 공분산역행렬이 튀어, 마할라노비스 거리가 튀기 때문으로 보임. 어떻게 해결해야?

""" 문제를 해결하기 위해 sk-learn의 코드를 가져와 봄

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('The number of classes has to be greater than'
                             ' one; got %d class' % (n_classes))
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        cov = None
        store_covariance = self.store_covariance or self.store_covariances
        if self.store_covariances:
            warnings.warn("'store_covariances' was renamed to store_covariance"
                          " in version 0.19 and will be removed in 0.21.",
                          DeprecationWarning)
        if store_covariance:
            cov = []
        means = []
        scalings = []
        rotations = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance '
                                 'is ill defined.' % str(self.classes_[ind]))
            Xgc = Xg - meang
            # Xgc = U * S * V.T
            U, S, Vt = np.linalg.svd(Xgc, full_matrices=False)

# >> 평균값 제거한 X에 대하여 SVD 진행, S: diag(SD), S * A = np.matmul(A, SD) <SD: Singulars Diagonal>
# >> U: LSV, S: RSV <LSV: Left Singular Vectors> <RSV: Right Singular Vectors>

            rank = np.sum(S > self.tol)
            if rank < n_features:
                warnings.warn("Variables are collinear")
            S2 = (S ** 2) / (len(Xg) - 1)

# >> S2 = diag(SD)^2 / (n - 1)

            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append(np.dot(S2 * Vt.T, Vt))

# >> cov[i] = (RSV (matmul) SD^2 (matmul) RSV.T) / (n - 1)

            scalings.append(S2)

# >> scalings[i] = diag(SD)^2 / (n - 1) (diagonal matrix이기 때문에 scaling)

            rotations.append(Vt.T)

# >> rotations[i] = RSV (orthogonal matrix이기 때문에 rotation)
            
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]

# >> R = RSV

            S = self.scalings_[i]

# >> S = diag(SD)^2 / (n - 1)

            Xm = X - self.means_[i]

# >> Xm = unbiased_X_var

            X2 = np.dot(Xm, R * (S ** (-0.5)))

# >> X2 = sqrt(n - 1) * unbiased_X_var (matmul) RSV (matmul) pinv(SD)

            norm2.append(np.sum(X2 ** 2, 1))

# >> norm2[i] = squarenorm( sqrt(n - 1) * unbiased_X_var (matmul) RSV (matmul) pinv(SD) )

        norm2 = np.array(norm2).T   # shape = [len(X), n_classes]
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])

# >> u = sum( log( diag(SD)^2 / (n - 1) ) ) = log( det(unbiased_X_model) / (n - 1) )

        return (-0.5 * (norm2 + u) + np.log(self.priors_))

# >> 이 return 식을 만족하려면 norm2의 제곱근이 mahalanobis distance라는 의미가 된다...하지만 왜....??????
# >> 직관적으로는 rignt singular vector로 회전을 시켜 축을 정렬한 뒤, singular diagonal의 역수를 곱하여 노말라이즈하니, 표준편차로 나눈 거리가 되기는 한다.
# >> 그러나 이를 수식적으로 증명하려면 어떻게 해야?

 """
class LDA:

    def __init__(self, X, Y):
        self.set_data(X, Y)

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y.reshape((-1, 1))

        self.X_c1 = self.X[self.Y.flatten() >= 0]
        self.Y_c1 = self.Y[self.Y.flatten() >= 0]
        self.X_c2 = self.X[self.Y.flatten() < 0]
        self.Y_c2 = self.Y[self.Y.flatten() < 0]

        self.X_c1_mean = np.mean(self.X_c1, axis=0)
        self.X_c2_mean = np.mean(self.X_c2, axis=0)

        self.X_c1_cov = np.cov(self.X_c1, rowvar=False)
        self.X_c2_cov = np.cov(self.X_c2, rowvar=False)

        pp.pprint(self.X_c1_mean)
        pp.pprint(self.X_c1_cov)


# PCA EXECUTE

# data = np.genfromtxt('data-03-diabetes.csv', delimiter=',')
# X = data[:, :8]
# pca = PCA(X)
# X_pca = pca.pca_auto_exec()
# print(X_pca)


# SVM EXECUTE

# data = np.genfromtxt('data-03-diabetes.csv', delimiter=',')
# X = data[:, :8]
# Y = data[:, -1]
# svm = SVM(X, Y)
# model = svm.svm_exec(learning_rate=0.01, kernel='sigmoid')
# Y = Y.reshape((-1, 1))
# print(np.hstack((model(X), Y)))


# def logistic(X, W):
#     return 1 / (1 + np.exp(-np.matmul(X, W)))

# def softmax(X, W):
#     return np.exp(np.matmul(X, W)) / np.exp(np.matmul(X, W)).sum(axis = 0)

# def naive_bayes(X):

# def lstm(X):


data1 = np.genfromtxt('data-04-zoo.csv', delimiter=',')
X = data1[:, 0:-1]
Y = data1[:, [-1]].astype(int)
pca = PCA(X)
X = pca.pca_auto_exec()
qda = QDA(X, Y)
model = qda.qda_exec(X)
model_X = model(X)
pp.pprint(np.hstack([Y, model(X)]))
c1 = 0
for i in range(Y.shape[0]):
    if model_X[i] == Y[i]:
        c1 += 1
print(c1 / Y.shape[0])


# data2 = np.genfromtxt('data-03-diabetes.csv', delimiter=',')
# X = data2[:, 0:-1]
# Y = data2[:, [-1]].astype(int)
# pca = PCA(X)
# X = pca.pca_exec(1)
# qda = QDA(X, Y)
# model = qda.qda_exec(X)
# model_X = model(X)
# c2 = 0
# for i in range(Y.shape[0]):
#     if model(X)[i] == Y[i]:
#         c2 += 1
# print(c2 / Y.shape[0])
