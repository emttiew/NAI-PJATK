"""@Authors Mateusz Woźniak 18182, Jakub Włoch 16912"""
"""Reference: """
'''https://waterprogramming.wordpress.com/2019/10/21/beginners-guide-to-tensorflow-and-keras/'''


'''Import relevant libraries'''
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''Definiton of method showing plot with mse and mae '''
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Rings^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Mean Squared Error')
    plt.plot(hist['epoch'], hist['mae'],
             label='Mean Absolute Error')
    plt.legend()
    plt.show()


'''Import dataset'''
dataset = pd.read_csv('abalone.csv')

'''Print number of observations and features'''
print('This dataset has {} observations with {} features.'.format(dataset.shape[0], dataset.shape[1]))

'''Check for null values'''
dataset.info()
Sex = dataset.pop('Sex')
dataset['M'] = (Sex == 'M') * 1.0
dataset['F'] = (Sex == 'F') * 1.0
dataset['I'] = (Sex == 'I') * 1.0

'''Reorder Columns'''
dataset = dataset[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'M', 'F', 'I', 'Rings']]
print(dataset)

'''Separate input data and labels'''
X = dataset.iloc[:, 0:10]
y = dataset.iloc[:, 10].values

'''Normalize the data using the min-max scalar'''
scalar = MinMaxScaler()
X = scalar.fit_transform(X)
y = y.reshape(-1, 1)
y = scalar.fit_transform(y)

'''Split data into training and testing'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''Build Keras Model'''
model = Sequential()
model.add(Dense(units=10, input_dim=10, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
modelHistory = model.fit(X_train, y_train, batch_size=5, validation_split=0.2, callbacks=[early_stop], epochs=100)

'''Model summary for number of parameters use in the algorithm'''
model.summary()
plot_history(modelHistory)

'''Predict testing labels'''
y_pred = model.predict(X_test)

'''undo normalization'''
y_pred_transformed = scalar.inverse_transform(y_pred.reshape(-1, 1))
y_test_transformed = scalar.inverse_transform(y_test)

'''visualize performance'''
fig, ax = plt.subplots()
ax.scatter(y_test_transformed, y_pred_transformed)
ax.plot([y_test_transformed.min(), y_test_transformed.max()], [y_test_transformed.min(), y_test_transformed.max()], 'k--', lw=4)
ax.set_xlabel('Measured (Rings)')
ax.set_ylabel('Predicted (Rings)')
plt.show()

'''Calculate RMSE and R^2'''
rms = sqrt(mean_squared_error(y_test_transformed, y_pred_transformed))
r_squared = r2_score(y_test_transformed, y_pred_transformed)
print('score: ', r_squared)
