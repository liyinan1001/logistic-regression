import numpy as np
import math
import pylab
w = [0 for i in range(0, 64)]
w = np.array(w)
train3 = np.loadtxt('train3.txt')
label3 = np.array([[1 for i in range(0, train3.shape[0])]]).T
train3 = np.concatenate((train3, label3), axis=1)
train5 = np.loadtxt('train5.txt')
label5 = np.array([[0 for i in range(0, train5.shape[0])]]).T
train5 = np.concatenate((train5, label5), axis=1)
train = np.concatenate((train3, train5), axis=0)
t = train.shape[0]

test3 = np.loadtxt('test3.txt')
test5 = np.loadtxt('test5.txt')
l3 = np.array([[1 for i in range(0, test3.shape[0])]]).T
test3 = np.concatenate((test3, l3), axis=1)
l5 = np.array([[0 for i in range(0, test5.shape[0])]]).T
test5 = np.concatenate((test5, l5), axis=1)
test = np.concatenate((test3, test5), axis=0)


err = 1000
thre = 0.05
def sigma(x, w):
    return 1.0 / (1 + math.exp(-np.dot(x, w)))


def likelihood(row, w):
    return row[-1] * math.log(sigma(row[0:-1], w)) + (1 - row[-1]) * math.log(sigma(row[0:-1], -w))


def update(row, w):
    return (row[-1] - sigma(row[0:-1], w)) * row[0:-1]


def error(row, w):
    result = 0
    if sigma(row[0:-1], w) > 0.5:
        result = 1
    if row[-1] == result:
        return 0
    else:
        return 1

errArray = []
likeArray = []
count = 0
while count < 20000:
    w = w + 0.1 / t * np.sum(np.apply_along_axis(update, 1, train, w), axis=0)
    likeli = np.sum(np.apply_along_axis(likelihood, 1, train, w))
    err = np.sum(np.apply_along_axis(error, 1, train, w)) * 1.0 / t
    count += 1
    if count % 50 == 0:
        errArray.append(err)
        likeArray.append(likeli)
        print count, likeli, err

print 'weight vector:'
for i in range(0,8):
    print w[i * 8:i * 8 + 8]
pylab.plot(range(0, len(errArray)), errArray)
pylab.show()
pylab.plot(range(0, len(likeArray)), likeArray)
pylab.show()

testError = np.sum(np.apply_along_axis(error, 1, test, w)) * 1.0 / test.shape[0]
print 'test error: ', testError
