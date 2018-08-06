
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

ar = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(type(ar))

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

answer = []
for di in data:
    answer.append(2*di)
pp.pprint(answer)

x = np.array(data)

pp.pprint(x)

d = np.array([[[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]],
              [[11, 12, 13, 14],
               [15, 16, 17, 18],
               [19, 20, 21, 22]]])

pp.pprint(d.ndim)
pp.pprint(d.shape)

pp.pprint(np.array([[10, 20, 30, 40], [
    50, 60, 70, 80]]))

m = np.array([[0,  1,  2,  3,  4],
              [5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14]])

pp.pprint(m[1, 2])
pp.pprint(m[2, 4])
pp.pprint(m[1, 1:3])
pp.pprint(m[1:, 2])
pp.pprint(m[:2, 3:])


# 불리안 인덱싱 : 원 ndarray와 크기가 같아야 하며 True / False로 체에 거름
# 정수 인덱싱 : 배열 크기 달라도 상관없으며, 해당정수인덱스의 원소들로 이루어진 배열 생성

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

pp.pprint(x[x % 3 == 0])
pp.pprint(x[x % 4 == 1])
pp.pprint(x[(x % 3 == 0) & (x % 4 == 1)])

pp.pprint(x.dtype)

x = np.array([1, 2, 3], dtype='f4')
pp.pprint(x.dtype)

a = np.array([0, 1, -1, 0])
b = np.array([1, 0, 0, 0])
c = a/b
pp.pprint(a.dtype)
pp.pprint(b.dtype)
pp.pprint(c.dtype)

q = np.arange(4, 9, 2)
pp.pprint(q)

pp.pprint(np.linspace(10, 1000, 5))
pp.pprint(np.logspace(1, 2, 5))

x = np.array([[1, 2, 3, 4, 5]])
pp.pprint(x)
pp.pprint(x.T)

pp.pprint(np.ones((3, 4)))

pp.pprint(np.zeros_like(x))

a = np.arange(12)
b = a.reshape(-1, 4)
pp.pprint(a)
pp.pprint(b)
pp.pprint(b.flatten())

c = np.arange(8)
d = c.reshape(-1, 8)
pp.pprint(c)
pp.pprint(d)
# (8, 1)
pp.pprint(c[:, np.newaxis])
pp.pprint(c[np.newaxis, :])
# (1, 8)

print('-------------------')
a1 = np.zeros([3, 4])
a2 = np.ones_like(a1)
pp.pprint(np.vstack([a1, a2]))
pp.pprint(np.r_[a1, a2])
pp.pprint(np.hstack([a1, a2]))
pp.pprint(np.c_[a1, a2])
pp.pprint(np.dstack([a1, a2]))
pp.pprint(np.stack([a1, a2]))
pp.pprint(np.stack([a1, a2], axis=0))
pp.pprint(np.stack([a1, a2], axis=1))
pp.pprint(np.stack([a1, a2], axis=2))

pp.pprint(np.hstack([np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]))
pp.pprint(np.c_[np.array([[1, 2, 3]]), np.array([[4, 5, 6]])])

pp.pprint(np.tile(a1, (3, 2)))

zeros = np.zeros([3, 3])
ones = np.ones([3, 2])
top = np.hstack([zeros, ones])
arange = np.arange(1, 16)
mult = arange * 10
bottom = np.reshape(mult, [-1, 5])
answer = np.vstack([top, bottom])
pp.pprint(answer)

x = np.arange(3)
y = np.arange(5)
X, Y = np.meshgrid(x, y)

pp.pprint(X)
pp.pprint(Y)

pp.pprint([list(zip(x, y)) for x, y in zip(X, Y)])

# plt.scatter(X, Y)
# plt.show()


x = np.arange(1, 10001)
y = np.arange(10001, 20001)
pp.pprint(x+y)
pp.pprint(x == y)

pp.pprint(np.vstack([range(7)[i:i+3] for i in range(5)]))
pp.pprint(np.vstack([range(20)[i:i+3] for i in range(5)]))
pp.pprint(np.vstack([range(4)[0:3]]))
# pp.pprint(np.vstack([[i:i+3] for i in range(5)]))

x = np.array([0, 1, 2, 3, 4])
pp.pprint(x.sum())
pp.pprint(np.sum(x))
pp.pprint(x.max())
pp.pprint(x.argmax())
pp.pprint(x.mean())
pp.pprint(np.all(x))
pp.pprint(np.any(x))
pp.pprint(np.any(x == 4))
pp.pprint(np.any(x != 4))
pp.pprint(np.all(x == 4))
pp.pprint(np.all(x != 4))

a = np.array([1, 2, 3, 2])
b = np.array([2, 2, 3, 2])
c = np.array([6, 4, 4, 5])

pp.pprint(((a <= b) & (b <= c)).all())
# 전체대상 조건이 아니라 인덱스별 조건

x = np.array([[1, 1], [2, 2]])
pp.pprint(x.sum())
pp.pprint(x.sum(axis=0))
pp.pprint(x.sum(axis=1))

y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
pp.pprint(y)
pp.pprint(y.ndim)
pp.pprint(y.shape)
pp.pprint(y.sum(axis=0))
pp.pprint(y.sum(axis=1))
pp.pprint(y.sum(axis=2))

a = np.vstack([range(8)[i:i+4] for i in range(1, 5)])
pp.pprint(a)
pp.pprint(a.max())
pp.pprint(a.sum(axis=1))
pp.pprint(a.mean(axis=0))

a = np.array([[1, 5, 9], [6, 3, 5], [2, 8, 4]])
pp.pprint(a)
pp.pprint(np.sort(a))
pp.pprint(np.sort(a, axis=-1))
pp.pprint(np.sort(a, axis=0))
pp.pprint(a)
pp.pprint(np.argsort(a))

a = np.array([[1,    2,    3,    4],
              [46,   99,  100,   71],
              [81,   59,   90,  100]])
pp.pprint(a)
# pp.pprint(a[np.argsort(a)[1]])
pp.pprint(a[a[:,1].argsort()])
pp.pprint(np.argsort(a)[1])
pp.pprint(a[1,:].argsort())
pp.pprint(a)
pp.pprint(np.vstack([a[i, a[1,:].argsort()] for i in range(3)]))

pp.pprint(len(a))
pp.pprint(a.mean())
pp.pprint(a.var())
pp.pprint(a.var(ddof=1))
pp.pprint(a.std())
pp.pprint(a.max())
pp.pprint(a.min())
pp.pprint(np.median(a))
pp.pprint(np.percentile(a, 25))


np.random.seed(0)
pp.pprint(np.random.rand(5))
pp.pprint(np.random.rand(5))
np.random.seed(0)
pp.pprint(np.random.rand(5))
pp.pprint(np.random.rand(5))

a = np.array([1,2,3,4,5,6,7,8,9])
pp.pprint(a)
np.random.shuffle(a)
pp.pprint(a)
pp.pprint(np.random.choice(a))

pp.pprint(np.diag([1,2,3]))
pp.pprint(np.identity(3))
pp.pprint(np.eye(4))

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[10,20,30],[40,50,60],[70,80,90]])
pp.pprint(a*b)
pp.pprint(np.dot(a,b))

pp.pprint(np.dot(np.ones((5,1)),np.ones((1,5))))
one_vec = np.ones((3,3))
x = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
pp.pprint(x - np.dot(one_vec, x)/3)

from sklearn.datasets import load_digits
X = load_digits().data
pp.pprint(X)
d0 = digi