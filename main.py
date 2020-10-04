import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error


dataset = pd.read_csv('ChickWeight.csv')
clean_set = pd.DataFrame(columns=['Chicken', 'StartWeight','EndWeight', 'Time',  'Diet'])
# print(clean_set)

## ---------------- ##
# DATA PREPROCESSING #
## ---------------- ##
chickArray = [0] * 50

for index, row in dataset.iterrows():
  chickArray[int(row['Chick'])-1] += 1

#Just putting some variables there
chickCounter = 0
rowInChicken = 1
startWeight = 0
endWeight = 0
days = 0
diet = 0


#we create a row for each chicken
for index, row in dataset.iterrows():
  if rowInChicken == 1:
    startWeight = dataset['weight'][index]
  
  #if we reach the last row of the specific chicken we are examinating
  # print("row in Chicken:" + str(rowInChicken))
  # print("ChickArray[ChickCounter]" + str(chickArray[chickCounter]))
  if rowInChicken == chickArray[chickCounter]:
    days = dataset['Time'][index]
    endWeight = dataset['weight'][index]
    diet = dataset['Diet'][index]

    newRow = {'Chicken':chickCounter+1, 'StartWeight':startWeight, 'EndWeight':endWeight, 'Time':days, 'Diet':diet}
    clean_set = clean_set.append(newRow, ignore_index=True)
    
    rowInChicken = 0
    chickCounter +=1

  rowInChicken +=1

#removing absurd values
for index, row in clean_set.iterrows():
  if clean_set['Time'][index] <10:
    clean_set = clean_set.drop(clean_set.index[index])



#separating into training and test sets
test_set = pd.DataFrame(columns=['Chicken', 'StartWeight','EndWeight', 'Time',  'Diet'])
clean_set2 = pd.DataFrame(columns=['Chicken', 'StartWeight','EndWeight', 'Time',  'Diet'])
for index, row in clean_set.iterrows():
  if clean_set["Chicken"][index] % 4 == 0:
    test_set = test_set.append(clean_set.loc[index,:])
  else:
    clean_set2 = clean_set2.append(clean_set.loc[index,:])


## -------------- ##
# MACHINE LEARNING #
## -------------- ##




X = clean_set2[['StartWeight', 'Time', 'Diet']]
Y = clean_set2['EndWeight']
LR = linear_model.LinearRegression()
#X = [clean_set2['StartWeight'], clean_set2['Time'],clean_set2['Diet']]
LR.fit(X, Y)

#print(LR.predict(40, 21,2))
X_test = test_set[['StartWeight', 'Time', 'Diet']]
Y_test = test_set['EndWeight']
predicted_weight = LR.predict(X_test)
# error = mean_squared_error(predicted_weight, Y_test)
# error1 = mean_absolute_error(predicted_weight, Y_test)
#print("error: (MSE) " + str(error))
#print("MAPE error: " + str(error1))

while True:

  weight = int(input("Input a weight between 35-40g: "))
  while weight<35 or weight>40:
    weight = int(input("Please input a weight between 35-40g: "))
    
  growthTime = int(input("Input a growth time (between 15-21 days): "))
  while growthTime<15 or growthTime>21:
    print("Please enter an amount of days that is between 15-21 days)")
    growthTime = int(input("Input a growth time: "))
  
  diet = int(input("Which diet will the chicken be fed? (1,2,3 or 4) "))
  while diet > 4:
    print("Please enter a number betweem 1 and 4")
    diet = int(input("Which diet will the chicken be fed? "))


  X_to_predict = pd.DataFrame({'StartWeight':[weight], 'Time':[growthTime], 'Diet':[diet]})
  Y_predicted = LR.predict(X_to_predict)
  print("\nThe chicken is predicted to have a weight of " + str(int(Y_predicted[0])) + "g after " + str(growthTime) + " days.")
  print("The chicken will grow " + str(int(Y_predicted[0])-weight) + "g.")
  
  print("")