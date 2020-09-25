import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

f = open("studentspen-train.csv", "r")
f.readline()
f1 = f.readlines()
data = np.empty([len(f1), 9], dtype=int)
d1 = 0
d7 = 0
d9 = 0
D = 8
eps = 1e-2
for i in range(len(f1)):
    tmp = f1[i].split(",")
    data[i] = [tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8]]
    if int(tmp[8]) == 1:
        d1 += 1
    if int(tmp[8]) == 7:
        d7 += 1
    if int(tmp[8]) == 9:
        d9 += 1

data1 = np.empty([d1, 8], dtype=int)
data2 = np.empty([d7, 8], dtype=int)
c1 = 0
c7 = 0
for i in data:
    if i[8] == 1:
        data1[c1] = i[0:8]
        c1 += 1
    if i[8] == 7:
        data2[c7] = i[0:8]
        c7 += 1

X = np.concatenate((data1, data2), axis=0)
y = np.concatenate((np.ones(d1), - np.ones(d7)))

svm = SVC(kernel='linear')
clf = make_pipeline(StandardScaler(), svm)
clf.fit(X, y)

d11 = 0
d99 = 0
for i in svm.support_:
    if i > 390:
        d11 += 1
    else:
        d99 += 1
array1 = np.empty([d1 + d11, 8], dtype=int)
array9 = np.empty([d9 + d99, 8], dtype=int)
c1 = 0
c9 = 0
for i in data:
    if i[8] == 1:
        array1[c1] = i[0:8]
        c1 += 1
    if i[8] == 9:
        array9[c9] = i[0:8]
        c9 += 1
for i in svm.support_:
    if i > 390:
        array1[c1] = X[i]
        c1 += 1
    else:
        array9[c9] = X[i]
        c9 += 1

X9 = np.concatenate((array1, array9), axis=0)
y9 = np.concatenate((np.ones(d1+d11), - np.ones(d9+d99)))

svm9 = SVC(kernel='linear')
clf9 = make_pipeline(StandardScaler(), svm9)
clf9.fit(X9, y9)

pred1 = clf9.predict(array1[:d1])
pred9 = clf9.predict(array9[:d9])

tmpc = 0
tmpi = 0
for i in pred1:
    if i == 1:
        tmpc += 1
    else:
        tmpi += 1
for i in pred9:
    if i == -1:
        tmpc += 1
    else:
        tmpi += 1

print(tmpi/(tmpc+tmpi))
f.close()