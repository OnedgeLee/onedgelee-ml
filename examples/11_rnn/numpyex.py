import numpy as np
import pprint as pp

# a = np.array([1, 0, 3])
# b = np.zeros((3, 4))
# b[np.arange(3), a] = 1
# b
# array([[ 0.,  1.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.]])

a = np.array([[0.,  1.,  0.,  0.],
              [1.,  0.,  0.,  0.],
              [0.,  0.,  0.,  1.]])

pp.pprint(a[0, 1])
pp.pprint(a[0][1])
pp.pprint(a[[0,1,2],[1,0,3]])
pp.pprint(a[[0,1],[1,0],[2,3]])

b = [[0.,  1.,  0.,  0.],
     [1.,  0.,  0.,  0.],
     [0.,  0.,  0.,  1.]]

print(b[0][1])
