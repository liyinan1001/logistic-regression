import numpy as np
import pandas as pd

df = pd.read_table('nasdaq00.txt', header=None)
eq = [[0 for j in range(0, 4)] for i in range(0, 4)]
eq = np.array(eq)
b = [0 for i in range(0, 4)]
b = np.array(b)
for t in range(4, len(df.index)):
    for r in range(0, 4):
        for c in range(0, 4):
            eq[r][c] += df.iloc[t - c - 1] * df.iloc[t - r - 1]
        b[r] += df.iloc[t] * df.iloc[t - r - 1]


x = np.linalg.solve(eq, b)
print 'coefficitents:', x

def computeError(df):
    mError = 0
    for t in range(4, len(df.index)):
        err = df.iloc[t]
        for i in range(0, 4):
            err -= df.iloc[t - i - 1] * x[i]
        mError += err ** 2
    mError = mError * 1.0 / (len(df.index) - 4)
    return mError


df1 = pd.read_table('nasdaq01.txt', header=None)
mError00 = computeError(df)
mError01 = computeError(df1)
print 'mean sqr error: ', '00: ', mError00, '01:', mError01, 'ratio: ', mError01 / mError00