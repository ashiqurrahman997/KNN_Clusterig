import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint

df = pd.read_csv('data.txt', sep=" ", header=None, dtype='float')

data = df.values

data_lenght = len(data)

plt.subplot(2, 1, 1)
plt.scatter(data[:, 0], data[:, 1])


rn1 = randint(0, data_lenght - 1)
rn2 = randint(0, data_lenght - 1)

print("Ashiqur Rahman, ID : 150204057")
print(rn1, ' ', rn2)


centorid_1 = np.zeros(2)
centorid_2 = np.zeros(2)
centorid_1 = data[rn1,:]
centorid_2 = data[rn2,:]

plt.scatter(centorid_1[0], centorid_1[1], color='red')
plt.scatter(centorid_2[0], centorid_2[1], color='yellow')

k = 2

classifier = np.zeros(data_lenght)

for j in range(0, 1000):
    cls1, cls2 = 0, 0
    cls1_nmbr, cls2_nmbr = 0, 0
    for i in range(0, data_lenght):
        dis1 = np.sqrt(np.power((data[i][0] - centorid_1[0]), 2) + np.power((data[i][1] - centorid_1[1]), 2))
        dis2 = np.sqrt(np.power((data[i][0] - centorid_2[0]), 2) + np.power((data[i][1] - centorid_2[1]), 2))
        if dis1 <= dis2:
            cls1 += 1
        else:
            cls2 += 1

    firstclass = np.zeros((2, cls1))
    secondclass = np.zeros((2, cls2))

    count = 0

    for i in range(0, data_lenght):

        dis1 = np.sqrt(np.power((data[i][0] - centorid_1[0]), 2) + np.power((data[i][1] - centorid_1[1]), 2))
        dis2 = np.sqrt(np.power((data[i][0] - centorid_2[0]), 2) + np.power((data[i][1] - centorid_2[1]), 2))
        if dis1 <= dis2:
            firstclass[0, cls1_nmbr] = data[i][0]
            firstclass[1, cls1_nmbr] = data[i][1]
            if classifier[i] != 1:
                count += 1
            classifier[i] = 1
            cls1_nmbr += 1
        else:
            secondclass[0][cls2_nmbr] = data[i][0]
            secondclass[1][cls2_nmbr] = data[i][1]
            if classifier[i] != 2:
                count += 1
            classifier[i] = 2
            cls2_nmbr += 1
    if count == 0:
        print("Iterration: " ,j)
        break

    centorid_1[0] = np.mean(firstclass[0, :])
    centorid_1[1] = np.mean(firstclass[1, :])
    centorid_2[0] = np.mean(secondclass[0, :])
    centorid_2[1] = np.mean(secondclass[1, :])


plt.subplot(2, 1, 2)
plt.scatter(firstclass[0], firstclass[1], color='black')
plt.scatter(secondclass[0], secondclass[1], color='green')
plt.show()
