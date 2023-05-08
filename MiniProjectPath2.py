import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
dataset_2['High Temp']  = pandas.to_numeric(dataset_2['High Temp'])
dataset_2['Low Temp']  = pandas.to_numeric(dataset_2['Low Temp'])
dataset_2['Precipitation']  = pandas.to_numeric(dataset_2['Precipitation'])

# print(dataset_2.to_string()) #This line will print out your data

# Question 1

bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
avg_traffic = [np.mean(dataset_2[i]) for i in bridges]
for i in range(len(bridges)):
    print("Bridge: ", bridges[i], "  Average traffic: ", avg_traffic[i])

bad_bridge = np.argmin(avg_traffic)
bridges.pop(bad_bridge)
print("\nBridges to install sensors: ", bridges)


# Question 2

# Precipation vs Number of Bikers plot
plt.figure()
plt.scatter(dataset_2['Precipitation'], dataset_2['Total'])
plt.xlabel('Precipitation')
plt.ylabel('Number of Bikers')
plt.title('Daily Precipation vs Number of Bikers')

# High Temperature vs Number of Bikers plot
plt.figure()
plt.scatter(dataset_2['High Temp'], dataset_2['Total'])
plt.xlabel('High Temperature')
plt.ylabel('Number of Bikers')
plt.title('Daily High Temperature vs Number of Bikers')

# Low Temperature vs Number of Bikers plot
plt.figure()
plt.scatter(dataset_2['Low Temp'], dataset_2['Total'])
plt.xlabel('Low Temperature')
plt.ylabel('Number of Bikers')
plt.title('Daily Low Temperature vs Number of Bikers')

plt.show()

# Linear Model
X_avg = np.array([dataset_2['High Temp'], dataset_2['Low Temp']]).T
Y_avg = np.array(dataset_2['Total'])[:, np.newaxis]

avg_model = LinearRegression()
avg_model.fit(X_avg, Y_avg)

r_sqr_avg = avg_model.score(X_avg, Y_avg)
beta_avg = (np.append(avg_model.coef_[0], avg_model.intercept_))

print("Temperature average model parameters: ", beta_avg)
print("R-Squared for temperature average model: ", r_sqr_avg)


# Question 3

X = np.array([dataset_2['Brooklyn Bridge'], dataset_2['Manhattan Bridge'], dataset_2['Williamsburg Bridge'], dataset_2['Queensboro Bridge']]).T

days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
Y = np.array([days[day] for day in dataset_2["Day"]])

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

k = np.arange(2, 20)
score = []
for i in range(len(k)):
    model = KNeighborsClassifier(n_neighbors=int(k[i]))
    model.fit(X_train, y_train)
    score.append(model.score(X_test, y_test))

plt.figure()
plt.plot(k, score)
plt.xlabel('K')
plt.ylabel('Accuracy on Test Data')
plt.title('Different Values for K vs the Test Accuracy')
plt.show()

best_k = k[np.argmax(score)]
print("Best value of k: ", best_k)
print("Test set accuracy: ", max(score))


### USE VALUE OF best_k and TRAIN KNN THEN COMPUTE CONFUSION MATRIX
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
# print(y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=days)
disp.plot()
plt.show()