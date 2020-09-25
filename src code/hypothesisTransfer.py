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
data3 = np.empty([d9, 8], dtype=int)
c1 = 0
c7 = 0
c9 = 0
for i in data:
    if i[8] == 1:
        data1[c1] = i[0:8]
        c1 += 1
    if i[8] == 7:
        data2[c7] = i[0:8]
        c7 += 1
    if i[8] == 9:
        data3[c9] = i[0:8]
        c9 += 1

X = np.concatenate((data1, data2), axis=0)
y = np.concatenate((np.ones(d1), - np.ones(d7)))

svm7 = SVC(kernel='linear')
clf7 = make_pipeline(StandardScaler(), svm7)
clf7.fit(X, y)

X1 = np.concatenate((data1, data3), axis=0)
y1 = np.concatenate((np.ones(d1), - np.ones(d9)))

svm9 = SVC(kernel='linear')
clf9 = make_pipeline(StandardScaler(), svm9)
clf9.fit(X1, y1)

X2 = np.concatenate((data1, data1, data2, data3), axis=0)
y2 = np.concatenate((np.ones(d1+d1), - np.ones(d7+d9)))

svm = SVC(kernel='linear')
clf = make_pipeline(StandardScaler(), svm)
clf.fit(X2, y2)

array1 = np.empty([d1, 8], dtype=int)
array9 = np.empty([d9, 8], dtype=int)
c1 = 0
c9 = 0
for i in data:
    if i[8] == 1:
        array1[c1] = i[0:8]
        c1 += 1
    if i[8] == 9:
        array9[c9] = i[0:8]
        c9 += 1
pred1 = clf.predict(array1)
pred9 = clf.predict(array9)

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