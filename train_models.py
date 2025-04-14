### Here NN was structured and trained on prepared data, to give prediction of generation power,
### what we will use futher

import keras as k
import sklearn
from sklearn import preprocessing
import pandas as pd
import numpy as np
import tensorflow
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
pd.set_option('display.max_columns', None)

### preparing data for training and testing NN
start_time = time.time()
data_frame = pd.read_csv("turbs_scal.csv")
input_names = [ 'latitude1', 'longitude1', 'elevation1', 'wind_speed1', 'nacelle_position1',
                'latitude2', 'longitude2', 'elevation2', 'wind_speed2', 'nacelle_position2',
                'latitude3', 'longitude3', 'elevation3', 'wind_speed3', 'nacelle_position3',
                'latitude4', 'longitude4', 'elevation4', 'wind_speed4', 'nacelle_position4',
                'latitude5', 'longitude5', 'elevation5', 'wind_speed5', 'nacelle_position5',
                'latitude6', 'longitude6', 'elevation6', 'wind_speed6', 'nacelle_position6',]

output_names = ['production1', 'production2', 'production3',
                'production4', 'production5', 'production6']

scaler = preprocessing.MinMaxScaler()
names = data_frame.columns
train_x = data_frame[input_names][:744]
train_y = data_frame[output_names][:744]
test_x = data_frame[input_names][744:]
test_y = data_frame[output_names][744:]

### structue of NN
model = k.Sequential()
#input layer
model.add(k.layers.Dense(units = 5, activation="relu"))
#hidden layer
model.add(k.layers.Dense(units = 128, activation="tanh"))
#output layer
model.add(k.layers.Dense(units = 6, activation="tanh"))
model.compile(loss="mae", optimizer="adam")
fit_results = model.fit(x=train_x, y=train_y, epochs=100, batch_size=32, validation_split=0.2, shuffle=False, verbose=1)

predicted_test = model.predict(test_x)
print('Results of NN forcasting')
print(predicted_test)

lat_max, lat_min = 52.403834, 52.398781
long_max, long_min = -0.936093, -0.949527
elev_max, elev_min = 156.577, 135.039
d = pd.DataFrame()

real_data = data_frame.iloc[744:][input_names + output_names]
# 2050 - max production
print(real_data)
for j in range(1,7):
    for i in range(744, 1416):
        real_data[f'production{j}'][i] = real_data[f'production{j}'][i] * 2050

print(real_data)
print(predicted_test)


### Determination of the best generating turbine based on NN forcasting
max_prod_pred = real_data['production1'].sum()
point = 1
for i in range(2,7):
    if(real_data[f'production{i}'].sum() > max_prod_pred):
        max_prod_pred = real_data[f'production{i}'].sum()
        point = i
print('the most generating turbine: ', point)
print('summary of forcasted generation')
for i in range (1,7):
    print(f'summary of turbine {i}: ', real_data[f'production{i}'].sum())