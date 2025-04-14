### in this file was performed data preparation for using it for neural network
### data was parsed, scaled and distributed by different csv files for  futher comfortable work
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import csv
# for j in range(720, 1416):
#     for i in range (1, 7):
#         file_din = open(f'Turbine_Data_Kelmarsh_{i}_2018-01-01_-_2019-01-01.csv', 'r')
#         file_const = open('Kelmarsh_WT_static.csv', 'r')
#         type(file_din)
#         type(file_const)
#         csvreader1 = csv.reader(file_din)
#         csvreader2 = csv.reader(file_const)
#
#         rows1 = []
#         for row in csvreader1:
#             rows1.append(row)
#         print(rows1[9][1], rows1[9][61], rows1[9][27])
#         print(rows1[10+j*6][1], rows1[10+j*6][61], rows1[10+j*6][27])
#         print(rows1[10+j*6][1], rows1[10+j*6][61], rows1[10+j*6][27])
#         rows2 = []
#         for row in csvreader2:
#             rows2.append(row)
#         'title_of_turbine', 'placement_of_turbine', 'wind_speed', 'wind_direction', 'nacelle_position'
#
#         pd.set_option('display.max_columns', None)
#         d = {'data_time': [rows1[10+j*6][0]], 'latitude': rows2[i][9],
#               'longitude': rows2[i][10], 'elevation': rows2[i][11],
#               'wind_speed': rows1[10+j*6][1], 'wind_direction': rows1[10+j*6][15],
#               'nacelle_position': rows1[10+j*6][16], 'title_of_turbine': rows2[i][1],
#               'production': rows1[10+j*6][27]}
#
#         df = pd.DataFrame(data = d)
#         df.index = df['data_time']
#         df = df.drop(columns = ['data_time'])
#         print(df)
#         df.to_csv(f'turb{i}', mode = 'a', header = False)
# df_sum = pd.DataFrame()
# for i in range(1,7):
#     file_din = open(f'Turbine_Data_Kelmarsh_{i}_2018-01-01_-_2019-01-01.csv', 'r')
#     file_const = open(f'turb{i}.csv', 'r')
#     csvreader1 = csv.reader(file_din)
#     csvreader2 = csv.reader(file_const)
#     rows1 = []
#     for row in csvreader1:
#         rows1.append(row)
#     # print(rows1[9][61])
#     # print(rows1[10][61])
#
#     rows2 = []
#     for row in csvreader2:
#         rows2.append(row)
#     # print(rows2[0][8])
#     # print(rows2[1][8])
#
#     for j in range(1,1417):
#         rows2[j][8] = rows1[10+j][61]
#     # print(rows2[0][0])
#     # print(rows2[1][0])
#     df_sum = pd.DataFrame()
#     for j in range(1417):
#         d = {'data_time': [rows2[j][0]], 'latitude': rows2[j][1],
#             'longitude': rows2[j][2], 'elevation': rows2[j][3],
#             'wind_speed': rows2[j][4], 'wind_direction': rows2[j][5],
#             'nacelle_position': rows2[j][6], 'title_of_turbine': rows2[j][7],
#             'production': rows2[j][8]}
#
#         df = pd.DataFrame(data = d)
#         df.index = df['data_time']
#         df = df.drop(columns = ['data_time'])
#         print(rows1[10 + 1415][61])
#         df_sum.append(df)
#         #df.to_csv(f'turb{i}', mode = 'a', header = False)
#         #df.to_csv(f'turb', mode = 'a', header = False)

df1 = pd.read_csv('turb1.csv')
df2 = pd.read_csv('turb2.csv')
df3 = pd.read_csv('turb3.csv')
df4 = pd.read_csv('turb4.csv')
df5 = pd.read_csv('turb5.csv')
df6 = pd.read_csv('turb6.csv')

df_sum = pd.read_csv('turbs.csv')
df_sum.index = df_sum['data_time']
df_sum = df_sum.drop(columns = ['data_time', 'data_time.1', 'data_time.2',
                                'data_time.3','data_time.4', 'data_time.5'])
pd .set_option('display.max_columns', None)
print(df_sum)
#df_sum.to_csv(f'turbss.csv', mode = 'a', header = False)

#data_time,latitude,longitude,elevation,wind_speed,wind_direction,nacelle_position,title_of_turbine,production
data_frame = pd.read_csv("turbss.csv")
print(data_frame)
data_frame = data_frame.drop(['data_time', 'title_of_turbine1', 'title_of_turbine2',
                              'title_of_turbine3', 'title_of_turbine4',
                              'title_of_turbine5', 'title_of_turbine6'], axis = 1)
print(data_frame)

# print(data_frame['wind_speed6'][1])
# print(data_frame['wind_direction6'][1])
# print(data_frame['production6'][1])
# print(data_frame['wind_speed6'][1415])
# print(data_frame['wind_direction6'][1415])
# print(data_frame['production6'][1415])
print(data_frame['latitude1'][1])
print(data_frame['longitude1'][1])
print(data_frame['elevation1'][1])
lat_max, lat_min = 52.403834, 52.398781
long_max, long_min = -0.936093, -0.949527
elev_max, elev_min = 156.577, 135.039
d = pd.DataFrame()


for i in range(1,7):
    lat = (data_frame[f'latitude{i}'][2] - lat_min) / (lat_max - lat_min)
    long = (data_frame[f'longitude{i}'][2] - long_min) / (long_max - long_min)
    elev = (data_frame[f'elevation{i}'][2] - elev_min) / (elev_max - elev_min)
    for j in range(len(data_frame)):
        data_frame[f'latitude{i}'][j] = lat
        data_frame[f'longitude{i}'][j] = long
        data_frame[f'elevation{i}'][j] = elev

print(data_frame)
minmax_trans = sklearn.pipeline.Pipeline(steps=['minmax', sklearn.preprocessing.MinMaxScaler()])
columns = ['wind_speed1', 'wind_direction1', 'nacelle_position1', 'production1',
                                 'wind_speed2', 'wind_direction2', 'nacelle_position2', 'production2',
                                 'wind_speed3', 'wind_direction3', 'nacelle_position3', 'production3',
                                 'wind_speed4', 'wind_direction4', 'nacelle_position4', 'production4',
                                 'wind_speed5', 'wind_direction5', 'nacelle_position5', 'production5',
                                 'wind_speed6', 'wind_direction6', 'nacelle_position6', 'production6']
preprocessor = ColumnTransformer(
    [('minmax', MinMaxScaler(), ['wind_speed1', 'wind_direction1', 'nacelle_position1', 'production1',
                                 'wind_speed2', 'wind_direction2', 'nacelle_position2', 'production2',
                                 'wind_speed3', 'wind_direction3', 'nacelle_position3', 'production3',
                                 'wind_speed4', 'wind_direction4', 'nacelle_position4', 'production4',
                                 'wind_speed5', 'wind_direction5', 'nacelle_position5', 'production5',
                                 'wind_speed6', 'wind_direction6', 'nacelle_position6', 'production6'])])
d_scal = preprocessor.fit_transform(data_frame)
#d_scal = d_scal.rename(columns = columns)
print(d_scal[0][0])


# for i in range(1,7):
#     for j in range(len(data_frame)):
#         data_frame[f'wind_speed{i}'][j] = d_scal[i][j]
#         data_frame[f'wind_direction{i}'][j] = d_scal[i][j]
#         data_frame[f'nacelle_position{i}'][j] = d_scal[i][j]
#         data_frame[f'production{i}'][j] = d_scal[i][j]

print(d_scal)
print(d_scal[0][0])
print(d_scal[1400][0])


for j in range(len(d_scal)):
    data_frame['wind_speed1'][j] = d_scal[j][0]
    data_frame[f'wind_direction1'][j] = d_scal[j][1]
    data_frame[f'nacelle_position1'][j] = d_scal[j][2]
    data_frame[f'production1'][j] = d_scal[j][3]

    data_frame['wind_speed2'][j] = d_scal[j][4]
    data_frame[f'wind_direction2'][j] = d_scal[j][5]
    data_frame[f'nacelle_position2'][j] = d_scal[j][6]
    data_frame[f'production2'][j] = d_scal[j][7]

    data_frame['wind_speed3'][j] = d_scal[j][8]
    data_frame[f'wind_direction3'][j] = d_scal[j][9]
    data_frame[f'nacelle_position3'][j] = d_scal[j][10]
    data_frame[f'production3'][j] = d_scal[j][11]

    data_frame['wind_speed4'][j] = d_scal[j][12]
    data_frame[f'wind_direction4'][j] = d_scal[j][13]
    data_frame[f'nacelle_position4'][j] = d_scal[j][14]
    data_frame[f'production4'][j] = d_scal[j][15]

    data_frame['wind_speed5'][j] = d_scal[j][16]
    data_frame[f'wind_direction5'][j] = d_scal[j][17]
    data_frame[f'nacelle_position5'][j] = d_scal[j][18]
    data_frame[f'production5'][j] = d_scal[j][19]

    data_frame['wind_speed6'][j] = d_scal[j][20]
    data_frame[f'wind_direction6'][j] = d_scal[j][21]
    data_frame[f'nacelle_position6'][j] = d_scal[j][22]
    data_frame[f'production6'][j] = d_scal[j][23]


print(data_frame)

data_frame.to_csv(f'turbs_scal.csv', mode = 'a', header = True, index = False)


# input_names = ['latitude1', 'longitude1', 'elevation1', 'wind_speed1', 'nacelle_position1',
#                'latitude2', 'longitude2', 'elevation2', 'wind_speed2', 'nacelle_position2',
#                'latitude3', 'longitude3', 'elevation3', 'wind_speed3', 'nacelle_position3',
#                'latitude4', 'longitude4', 'elevation4', 'wind_speed4', 'nacelle_position4',
#                'latitude5', 'longitude5', 'elevation5', 'wind_speed5', 'nacelle_position5',
#                'latitude6', 'longitude6', 'elevation6', 'wind_speed6', 'nacelle_position6']
#
# output_names = ["production1", 'production2', 'production3',
#                 'production4', 'production5', 'production6']
# names = ['latitude1', 'longitude1', 'elevation1', 'wind_speed1', 'nacelle_position1', 'production1',
#          'latitude2', 'longitude2', 'elevation2', 'wind_speed2', 'nacelle_position2', 'production2',
#          'latitude3', 'longitude3', 'elevation3', 'wind_speed3', 'nacelle_position3', 'production3',
#          'latitude4', 'longitude4', 'elevation4', 'wind_speed4', 'nacelle_position4', 'production4',
#          'latitude5', 'longitude5', 'elevation5', 'wind_speed5', 'nacelle_position5', 'production5',
#          'latitude6', 'longitude6', 'elevation6', 'wind_speed6', 'nacelle_position6', 'production6']
# scaler = preprocessing.MinMaxScaler()
# names = data_frame.columns
# print(data_frame)
# data_frame = data_frame.drop(['data_time', 'title_of_turbine1', 'title_of_turbine2', 'title_of_turbine3', 'title_of_turbine4',
#                               'title_of_turbine5', 'title_of_turbine6'], axis = 1)
# d = scaler.fit_transform(data_frame)
# print(d)
# scaled_df = pd.DataFrame(data_frame, columns = names)
# scaled_df.head()
# print(scaled_df.head())
# scaled_df = scaled_df.dropna(axis = 1)
# print(scaled_df)
# sd = scaler.fit_transform(scaled_df)
# print(sd)

data_frame1 = pd.read_csv("turbs_scal.csv")
data_frame1 = data_frame1.drop(['latitude1', 'longitude1', 'elevation1', 'wind_speed1', 'wind_direction1' ,'nacelle_position1', 'production1',
          'latitude2', 'longitude2', 'elevation2', 'wind_speed2', 'wind_direction2' ,'nacelle_position2', 'production2',
          'latitude3', 'longitude3', 'elevation3', 'wind_speed3', 'wind_direction3' ,'nacelle_position3', 'production3',
          'latitude4', 'longitude4', 'elevation4', 'wind_speed4', 'wind_direction4' ,'nacelle_position4', 'production4',
          'latitude5', 'longitude5', 'elevation5', 'wind_speed5', 'wind_direction5' ,'nacelle_position5', 'production5'], axis = 1)
print(data_frame1)
data_frame1.to_csv(f'turb6_scal.csv', mode = 'a', header = True, index = False)