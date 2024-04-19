# Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split  # to split data
from sklearn.linear_model import LinearRegression  # to create the model
from sklearn.metrics import mean_squared_error, r2_score  # to obtain some numbers

# Reading CSV FILE
datafile = Path('Nutricion.csv')
df = pd.read_csv(datafile, index_col=0)


# Obtaining analysis data like median from the columns and inserting them inside vectors

for j in range(0, 5):
    A.insert(j, df[vard[j]].mean())
    B.insert(j, df[vard[j]].median())
    C.insert(j, df[vard[j]].mode())
    E.insert(j, df[vard[j]].var())
    F.insert(j, df[vard[j]].std())

# Creating figures


fig2, axes2 = plt.subplots(2, 3)
fig3, axes3 = plt.subplots(2, 2)

# Creating normalized histograms and adding them to subplot figures

# Printing Dataframe table with the data analysis like median, mode, etc.

DIO = pd.DataFrame({'Val': ['mean', 'median', 'mode', 'variance', 'standard d'], 'Calories': [A[0], B[0], C[0], E[0],
                                                                                              F[0]],
                    'Carbo': [A[1], B[1], C[1], E[1], F[1]], 'Fats': [A[2], B[2], C[2], E[2], F[2]],
                    'Protein': [A[3], B[3], C[3],
                                E[3], F[3]], 'Sodio': [A[4], B[4], C[4], E[4], F[4]]})

print(tabulate(DIO, headers='keys', tablefmt='psql'))

# model analysis with OLS and printing parameters and predictions

model = smf.ols('Y ~ X1 + X2 + X3 + X4', data=df).fit()
parameters = model.params
paramErrors = model.bse
predictions = model.predict()

n = 5
print('\nParameters:\n', parameters)
print('\nStandard errors:\n', paramErrors)
print('\nPredicted values:\n', predictions[:n])

# Check Residues and print it's histogram

x_cal = Y.dropna().values.tolist()
# print(len(x_cal))
# print('\n')
Pre = predictions[:len(x_cal)].tolist()
# print(len(Pre))
# print('\n')
# print(x_cal)
print('\n')
# print(Pre)
for i in range(len(Pre)):
    G.insert(i, x_cal[i] - Pre[i])

residuos = G
print('\nResiduos:\n', residuos)

suma = sum(residuos)
print('\nSumatoria de residuos:\n', suma)

len(residuos)
sns.histplot(residuos, kde=True, ax=axes3[0, 0])
axes3[0, 0].set_title('Histograma de Residuos')
scatti = [x / 215.55 for x in residuos]
printableY = []
for k in range(len(scatti)):
    printableY.insert(k, Y[k])

# Print proofs of variance and independence

sns.scatterplot(x=scatti, y=printableY, ax=axes3[0, 1])
axes3[0, 1].xaxis.set_label_text('Pronóstico KCalorías')
axes3[0, 1].yaxis.set_label_text('Residuos')
axes3[0, 1].set_title('Prueba de Varianza Homogénea')

sns.lineplot(data=residuos, ax=axes3[1, 0])
axes3[1, 0].xaxis.set_label_text('# de alimento')
axes3[1, 0].yaxis.set_label_text('Residuos')
axes3[1, 0].set_title('Prueba de Independencia')

fig2.tight_layout()
fig3.tight_layout()

# Create a trained prediction model

x = np.nan_to_num(df[[vard[1], vard[2], vard[3], vard[4]]])
y = np.nan_to_num(Y)

# make sub-samples of training and test data using a proportion of 1/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.60)
train_errors, val_errors = [], []

print('Elements in training sample: %d' % len(y_train))
print('Elements in test     sample: %d' % len(y_test))

# creating a model
print('\nBuilding linear regression model\n')
model1 = LinearRegression()

# training model
print('\ntraining model ...\n')
model1.fit(x_train, y_train)

# print the coefficients
print('Coefficients: \n', model1.coef_)

# print intercept
print('intercept: \n', model1.intercept_)

# print model
print('\nModel:\nY(x) = %f' % model1.intercept_)
i = 1

for par in model1.coef_:
    print(' %f x{%d}' % (1 * par, i))
    i += 1

# getting predictions from the model, note that for this we use the test sample,
# not the training sample
model_pred = model1.predict(np.nan_to_num(x_test))
npred = 5
print('\nshowing first %d predictions:\n' % npred)
print(model_pred[:npred])

# print mean squared error
print('\nMean squared error: %.2f' % mean_squared_error(y_test, model_pred))
# print the determination coefficient
print('Coefficient of determination: %.2f' % r2_score(y_test, model_pred))

y_model = model1.predict(x_train)  # this is the model, and it will be unique
y_model2 = model1.predict(x_test)
plotx = np.array(range(1800))
ploty = (model1.coef_[0]*plotx)+(model1.coef_[1]*plotx)+(model1.coef_[2]*plotx)+(model1.coef_[3]*plotx)+model1.intercept_
figure3 = plt.figure()
# plt.scatter(x_train, y_train, color="red")
plt.title("Calories vs Macronutrients y micronutrients (Training set)")
plt.xlabel("Carbs, fats, etc. ")
plt.ylabel("Calories")
plt.plot(x_train, y_model, color="black", marker='+')

figure4 = plt.figure()
# plt.scatter(x_test, y_test, color="red")
plt.title("Calories vs Macronutrients y micronutrients (Testing set)")
plt.xlabel("Carbs, fats, etc.")
plt.ylabel("Calories")
plt.plot(x_test, y_model2, color="blue", marker='+')
plt.plot(plotx, ploty, color="red")

figure5 = plt.figure()

for m in range(1, len(x_train)):
    model1.fit(x_train[:m], y_train[:m])
    y_train_predict = model1.predict(x_train[:m])
    y_val_predict = model1.predict(x_test)
    train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
    val_errors.append(mean_squared_error(y_test, y_val_predict))
plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="training set")
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation set")
plt.legend(loc='best', frameon=False)
plt.ylabel('RMSE')
plt.xlabel('training set size')

plt.show()
