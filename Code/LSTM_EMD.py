import numpy as np
import keras
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import r2_score

rd = np.random.randint(7777777)
callbacks = [EarlyStopping(monitor='val_loss',patience=50,verbose=2),
             ModelCheckpoint('LSTMcheckpoints/LSTMbest_%s.h5'%(rd),monitor='val_loss',
                             save_best_only=1, verbose=0)]


data = np.array(pd.read_csv("../demo_data/Hs.csv"))[:,1:]
emd = np.array(pd.read_csv("../demo_data/Hs_EMD(Leak).csv"))[:,1:]  # Leak version
# emd = np.array(pd.read_csv("../demo_data/Hs_EMD(NoLeak).csv"))[:,1:]  # No Leak version


data = np.concatenate((data,emd),axis=1)
wavedata = pd.DataFrame(data)


split_ratio = 0.25
res = 30
n_in = 12
selected_feature = [5, 6]
# print("The sequence used for training：",selected_feature)

unit1 = 32
unit2 = 32
epoch = 1000
batch_size = 32
lr = 0.01

val_freq = 1
opt = keras.optimizers.Adam(lr=lr , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000 , amsgrad=False )


def select_data(time_resolution=30):
    raw_csv = np.array(wavedata)
    timestep = int(time_resolution / 30)
    selected = np.arange(0, len(raw_csv), timestep)
    selected_csv = raw_csv[selected]
    return pd.DataFrame(selected_csv)


raw_data = select_data(time_resolution = res)
selected_data = np.array(raw_data.iloc[:,[0]+selected_feature])
list_a = selected_data[:,0].tolist()
mn = np.min(list_a)
mx = np.max(list_a)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(selected_data)

def series_to_supervised(data, n_in=2, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    return np.array(agg)[:, :-data.shape[1] + 1]

m = np.array(series_to_supervised(scaled, n_in= n_in, n_out=1, dropnan=True))

data_y = m[:,-1]
data_xx = m[:,:-1]
data_x = data_xx.reshape((data_xx.shape[0], int(n_in) ,data_xx.shape[1]//int(n_in)))

sep = int((data_y.shape[0])*(1-split_ratio))
train_X = data_x[:sep]
train_y = data_y[:sep]
validate_X = data_x[sep:]
validate_y = data_y[sep:]

print(train_X.shape)
model = Sequential()
model.add(LSTM(unit1,activation='softsign',input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences= 1))
model.add(LSTM(unit2,activation='relu',return_sequences=False))
model.add(Dense(units = 16,activation='relu'))
model.add(Dense(units = 1,activation='relu'))
model.compile(loss= "mae", optimizer=opt, metrics=['mae'])
model.summary()

history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch_size, verbose=1, shuffle=True,
                    callbacks = callbacks,
                    validation_data = (validate_X,validate_y),validation_freq = val_freq)

def RMSE(real,pred):
    real = np.reshape(real,(len(real),1))
    pred = np.reshape(pred,(len(pred),1))
    rmse = np.sqrt(sum((real - pred)**2)/float(len(real)))
    return rmse[0]
def MAE(real,pred):
    real = np.reshape(real,(len(real),1))
    pred = np.reshape(pred,(len(pred),1))
    mae = sum(np.abs(real-pred))/float(len(real))
    return mae[0]

lstm_model = model
lstm_model.load_weights('LSTMcheckpoints/LSTMbest_%s.h5'%(rd))
lstm_model.compile(loss="mae",optimizer=opt,metrics=['mae'])

yhat = lstm_model.predict(validate_X)
ytrainhat = lstm_model.predict(train_X)
loss1 = lstm_model.evaluate(validate_X,validate_y)[0]
loss2 = lstm_model.evaluate(train_X,train_y)[0]

print('LSTMcheckpoints/LSTMbest_%s.h5'%(rd))

#MAE
pre = yhat*(mx-mn)+mn
want = selected_data[12:,0]
wanted = want[sep:]
val_mae = MAE(wanted,pre)
print('MAE:',val_mae)

#MSE
def MSE(real,pred):
    real = np.reshape(real,(len(real),1))
    pred = np.reshape(pred,(len(pred),1))
    mse = sum((real - pred)**2)/float(len(real))
    return mse[0]

val_mse = MSE(wanted,pre)
print('MSE:', val_mse)

#RMSE
wanted=wanted.astype('float')
pre=pre.astype('float')
val_rmse = RMSE(wanted,pre)
print('RMSE:', val_rmse)


w = wanted.reshape(pre.shape[0], 1)
test_new = []
predict_new = []
for k in range(len(w)):
    if w[k] != 0:
        test_new.append(w[k])
        predict_new.append(pre[k])

# MAPE
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.fabs((y_true - y_pred) / y_true)) * 100

MAPE = MAPE(test_new, predict_new)
print('MAPE:', MAPE)

r2 = r2_score(wanted, pre)
print("R2 score on validation set: {:.8f}".format(r2))
