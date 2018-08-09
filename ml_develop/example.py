import numpy as np
import pprint as pp
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# class PCA:

#     def __init__(self, X):
#         self.set_data(X)

#     def set_data(self, X):
#         self.X = np.array(X)
#         eig_val, eig_vec = np.linalg.eig(np.cov(self.X, rowvar=False))
#         eig_val_desc = np.argsort(-eig_val)
#         self.eig_val = eig_val[eig_val_desc]
#         self.eig_vec = eig_vec[eig_val_desc]

#     def pca_exec(self, reduce_dim):
#         pcv = np.empty((0, self.X.shape[1]))
#         for i in range(self.X.shape[1] - reduce_dim):
#             pcv = np.vstack([pcv, self.eig_vec[i]])
#         return np.matmul(self.X, pcv.T)

#     def pca_auto_exec(self, target_ratio=0.9):
#         eig_val = self.eig_val
#         var_max = self.eig_val.sum()
#         var_ratio = 1
#         reduce_dim = -1
#         while var_ratio > target_ratio:
#             var_ratio -= eig_val[-1] / var_max
#             eig_val = eig_val[:-1]
#             reduce_dim += 1
#         return self.pca_exec(reduce_dim)


# data1 = np.genfromtxt('data-04-zoo.csv', delimiter=',')
# X = data1[:, 0:-1]
# Y = data1[:, -1].astype(int)

# pca = PCA(X)
# X = pca.pca_auto_exec()

# qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, Y)

# c1 = 0
# for i in range(Y.shape[0]):
#     if qda.predict(X)[i] == Y[i]:
#         c1 += 1
# print(c1 / Y.shape[0])


# # A = np.array([[-0.00863154,  0.09811895,  0.48468216,  1.22885436,  1.62626122],
# #               [-0.04064425, -0.39362137,  0.38497973,  1.25044332,  1.42853462],
# #               [0.22625274,  0.63547907,  0.41217464,  1.20881142,  1.2957319],
# #               [-1.10987419, -0.19254664, -1.2036845,   0.35838094,  1.53165362],
# #               [-1.54506444,  0.21646707, -0.94342634,  1.65557584,  1.26049608]])

# # B = np.array([[-0.20891794,  0.70593924, -0.22439883,  1.23855613,  1.76874527],
# #               [-0.03243604,  0.28362034, -0.60160129,  1.0134977,   1.81612101],
# #               [-0.20891794,  0.70593924, -0.22439883,  1.23855613,  1.76874527],
# #               [-0.20891794,  0.70593924, -0.22439883,  1.23855613,  1.76874527],
# #               [-0.17229484,  0.52577734,  0.17282539,  1.01659132,  2.09951296],
# #               [-0.12357075,  0.23947093, -0.56423417,  0.83428358,  1.79247153],
# #               [-0.20891794,  0.70593924, -0.22439883,  1.23855613,  1.76874527],
# #               [-0.17229484,  0.52577734,  0.17282539,  1.01659132,  2.09951296],
# #               [-0.20891794,  0.70593924, -0.22439883,  1.23855613,  1.76874527],
# #               [-0.12357075,  0.23947093, -0.56423417,  0.83428358,  1.79247153],
# #               [-0.12357075,  0.23947093, -0.56423417,  0.83428358,  1.79247153],
# #               [-0.40717912, -0.01158278,  0.24533292,  1.03663426,  2.43004228],
# #               [-0.17229484,  0.52577734,  0.17282539,  1.01659132,  2.09951296]])

# # A1 = np.array([[-8.6315409,     98.11894763,   484.68216404,  1228.85435656,
# #                 1626.26121606],
# #                [-40.64425158,  -393.62137165,   384.97972728,  1250.44331903,
# #                 1428.53461624],
# #                [226.2527389,    635.47906756,   412.17464118,  1208.81142207,
# #                 1295.73189748],
# #                [-1109.87418864,  -192.54664117, -1203.68450003,   358.38094077,
# #                 1531.65362069],
# #                [-1545.06443854,   216.46706922,  -943.42634096,  1655.57583864,
# #                 1260.49607991]])

# # cov_A = np.cov(A, rowvar=False)
# # det_cov_A = np.linalg.det(cov_A)

# # cov_B = np.cov(B, rowvar=False)
# # det_cov_B = np.linalg.det(cov_B)

# # cov_A1 = np.cov(A1, rowvar=False)
# # det_cov_A1 = np.linalg.det(cov_A1)


# # # pp.pprint(cov_A)
# # # pp.pprint(cov_B)
# # # pp.pprint(cov_A1)
# # # pp.pprint(np.linalg.det(np.cov(A1, rowvar=False)))
# # # pp.pprint(np.log(det_cov_A))
# # # pp.pprint(det_cov_B)
# # # pp.pprint(np.linalg.slogdet(np.cov(A)))
# # pp.pprint(np.linalg.slogdet(cov_A))

# # cov_A2 = np.array([[610975.69402866,  54630.49602467, 609548.6015857,   20559.42375775,
# #                     28357.43418003],
# #                    [54630.49602467, 156461.68568915,  62116.86051484,  67731.65638221,
# #                     -30305.7414865],
# #                    [609548.6015857,   62116.86051484, 685545.95900642, 142179.82837919,
# #                     18876.65753368],
# #                    [20559.42375775,  67731.65638221, 142179.82837919, 225393.44519498,
# #                     -39701.46611704],
# #                    [28357.43418003, -30305.7414865,   18876.65753368, -39701.46611704,
# #                     23900.71228811]])

# # # pp.pprint(np.linalg.det(cov_A2))

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = a - np.mean(a, axis=0)
U, S, Vt = np.linalg.svd(a, full_matrices=False)
pp.pprint(U)
pp.pprint(S)
pp.pprint(Vt)
S2 = (S ** 2) / (len(a) -1)
cov = np.matmul(S2 * Vt.T, Vt)
pp.pprint(S2)
pp.pprint(Vt.T)
pp.pprint(S2 * Vt.T)
pp.pprint(cov)