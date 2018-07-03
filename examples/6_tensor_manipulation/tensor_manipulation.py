import numpy as np
import tensorflow as tf
import pprint

pp = pprint.PrettyPrinter(indent=4)

t = np.array([[0., 1., 2.],
              [3., 4., 5.],
              [6., 7., 8.]])
pp.pprint(t)
print(t)
print(t.ndim)
print(t.shape)
