import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# mappiung function - SVC primal form
# loss function = hinge loss

write = False       # boolean to generate predictions text file

f = open("studentspen-train.csv", "r")
f.readline()
f1 = f.readlines()
data = np.empty([len(f1), 9], dtype=int)
for i in range(len(f1)):
    tmp = f1[i].split(",")
    data[i] = [tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8]]

X = np.empty([len(f1), 8])
x = 0
c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
for i in data:
    if i[8] == 0:
        c0 += 1
    if i[8] == 1:
        c1 += 1
    if i[8] == 2:
        c2 += 1
    if i[8] == 3:
        c3 += 1
    if i[8] == 4:
        c4 += 1
    if i[8] == 5:
        c5 += 1
    if i[8] == 6:
        c6 += 1
    if i[8] == 7:
        c7 += 1
    if i[8] == 8:
        c8 += 1
    if i[8] == 9:
        c9 += 1
    X[x] = i[0:8]
    x += 1

digits = np.zeros(len(f1))
digits[c0:c0+c1] = 1
digits[c0+c1:c0+c1+c2] = 2
digits[c0+c1+c2:c0+c1+c2+c3] = 3
digits[c0+c1+c2+c3:c0+c1+c2+c3+c4] = 4
digits[c0+c1+c2+c3+c4:c0+c1+c2+c3+c4+c5] = 5
digits[c0+c1+c2+c3+c4+c5:c0+c1+c2+c3+c4+c5+c6] = 6
digits[c0+c1+c2+c3+c4+c5+c6:c0+c1+c2+c3+c4+c5+c6+c7] = 7
digits[c0+c1+c2+c3+c4+c5+c6+c7:c0+c1+c2+c3+c4+c5+c6+c7+c8] = 8
digits[c0+c1+c2+c3+c4+c5+c6+c7+c8:c0+c1+c2+c3+c4+c5+c6+c7+c8+c9] = 9

t = open("studentsdigits-test.csv", "r")
t.readline()
t1 = t.readlines()
testArray = np.empty([len(t1), 8], dtype=int)
for i in range(len(t1)):
    tmp = t1[i].split(",")
    testArray[i] = [tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]]

svm = SVC(kernel='linear')
clf = make_pipeline(StandardScaler(), svm)
clf.fit(X, digits)
pred = clf.predict(testArray)


if write:
    w = open("steven-guan-prediction-file.txt", "w+")
    for i in pred:
        w.write("%d\n" % (int(i)))
    w.close()
f.close()
t.close()

# Test code
correct = 0
testPrediction = clf.predict(X)
for i in range(len(digits)):
    if testPrediction[i] == digits[i]:
        correct += 1

n = len(digits)
training_error = 1 - correct/len(digits)
print(training_error)
delta = 0.05
VC = 81     # dimension(X) + 1 = 8 features * 10 digits + 1
epsilon_max = (1/n) * (training_error + 4*np.log(4/delta) + 4*VC*np.log((2*np.e*n)/VC))
epsilon_min = max((VC-1)/(32*n), (1/n)*np.log(1/delta))

# Generalization error max and min
print(epsilon_max)
print(epsilon_min)

