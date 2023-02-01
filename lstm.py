import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout
from matplotlib import pyplot as plt


SPY = pd.read_csv("resources\\SPY.csv")
open_price = SPY["Open"].values

#separate training and test values
train = SPY[:int(0.7*len(SPY))]
#set is the open price for the last 5231 days.
set = train.iloc[:,1:2]

#scale values so it can be trained. Will reverse this when doing final predictions.
sc = MinMaxScaler()
scaled_train = sc.fit_transform(set)

x_train = []
y_train = []
for i in range(70, 5162):
    x_train.append(scaled_train[i-70:i, 0])
    y_train.append(scaled_train[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


test = SPY[int(0.8*len(SPY))+1:] # test starts where train ended
real_stock_price = test.iloc[:,1:2].values

dataset_total = pd.concat((train['Open'], test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(test)-70:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(70,250+70):
    X_test.append(inputs[i-70:i, 0]) # consider up to 70 rows before the one we are predicting
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

regressor = Sequential()
regressor.add(LSTM(units=55, return_sequences=True,input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=55, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=55))
regressor.add(Dropout(0.2))
regressor.add(Dense(1))

regressor.compile(optimizer="adam", loss="mean_squared_error")
regressor.fit(x_train, y_train, epochs=100, batch_size=32)


predicted_stock_price = regressor.predict(X_test)
#scales back values to predict.
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='red', label='Real S&P 500 Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted S&P 500 Stock Price')
plt.title("S&P 500 Stock Price Prediction")
plt.xlabel('time(days)')
plt.ylabel('S&P 500 stock Price')
plt.legend()
plt.show()
