import numpy as np
from matplotlib import pyplot as plt

data = np.array([
    [1, 83.5],
    [2, 96.1],
    [8, 84.2],
    [12, 80.6],
    [20, 77.3],
    [32, 74.5],
    [40, 68.6],
    [1, 84.3],
    [2, 97.0],
    [4, 87.6],
    [6, 82.3],
    [8, 78.9],
    [10, 74.2]
])
xCoods=[]
yCoods=[]
for i in data:
    xCoods.append(i[0])
    yCoods.append(i[1])
plt.title("Features vs Accuracy for Forward Selection")
plt.plot(xCoods[0:6], yCoods[0:6], color="r", label="Large Dataset")
plt.plot(xCoods[7:13], yCoods[7:13], color="g", label="Small Dataset")
plt.xlabel("X = No. of Features")
plt.ylabel("Y =  Accuracy (Percentage)")
plt.legend()
plt.show()


data = np.array([
    [-1, 74.7],
    [-2, 77.0],
    [-4, 79.0],
    [-6, 79.0],
    [-8, 77.0],
    [-10, 74.7],
    [-1, 83.5],
    [-2, 86.1],
    [-8, 81.1],
    [-12, 79.8],
    [-20, 79.1],
    [-32, 74.8],
    [-40, 68.6]
])
xCoods=[]
yCoods=[]
for i in data:
    xCoods.append(i[0])
    yCoods.append(i[1])
xCoods=xCoods[::-1]
yCoods=yCoods[::-1]
plt.title("Features vs Accuracy for Backward Elimination")
plt.plot(xCoods[0:6], yCoods[0:6], color="r", label="Large Dataset")
plt.plot(xCoods[7:13], yCoods[7:13], color="g", label="Small Dataset")
plt.xlabel("X = No. of Features")
plt.ylabel("Y = Accuracy (Percentage)")
plt.legend()
plt.show()