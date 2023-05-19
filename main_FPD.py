# from FPD_functions import Agent
# from FPD_functions import Data2
import numpy as np
import pandas as pd

# from FPD_functions import initialization
from FPD_functions import *
# from Second_closed_loop import *
# import csv
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly.express as px

agent, data2, system = main(10, "FALSE")


# agent, data2, system, data1, user = main_second(500, 'FALSE', 10)

# class NumpyEncoder(json.JSONEncoder):
#     """ Special json encoder for numpy types """
#
#     def default(self, obj):
#         if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
#                             np.int16, np.int32, np.int64, np.uint8,
#                             np.uint16, np.uint32, np.uint64)):
#             return int(obj)
#         elif isinstance(obj, (np.float_, np.float16, np.float32,
#                               np.float64)):
#             return float(obj)
#         elif isinstance(obj, (np.ndarray,)):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
#
#
# json_str = json.dumps(data2, cls=NumpyEncoder)
# print(json_str)

# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
#
#
# # jsonStr = json.dumps(data2.__dict__)
# json_str = json.dumps({'nums': data2}, cls=NpEncoder)
# print(json_str)

json_str= data2.toJSON()

# data_states = data2.states
# data_actions = data2.actions
#
# number_of_s = np.zeros(agent.ss)
# number_of_a = np.zeros(agent.aa)
#
# for j in range(agent.ss):
#     number_of_s[j] = np.sum(data_states[:] == j)
#
# for k in range(agent.aa):
#     number_of_a[k] = np.sum(data_actions[:] == k)
#
# data = {'States': list(np.arange(0, agent.ss)),
#         'Number of states': number_of_s}
#
# df_s = pd.DataFrame(data)
# # df_a = pd.DataFrame(number_of_a, columns=['Number of actions'])
#
# df_s.to_csv('data_states_try.csv')
#
# # px.bar(df_s, x='States', y='Number of states')
#
# # plt.figure(1)
# # plt.subplot(211)
# # plt.bar(range(agent.ss), number_of_s, width=0.4)
# # plt.subplot(212)
# # plt.bar(range(agent.aa), number_of_a, width=0.4)
# # plt.show()
#
# # plt.figure(1)
# # plt.subplot(211)
# # plt.bar(range(system.ss), pocets, width=0.4)
# # plt.subplot(212)
# # plt.bar(range(agent.aa), poceta, width=0.4)
# # plt.show()
#
# # with open('C:/Users/Tereza/Desktop/marko/data/data1.csv', 'w', encoding='UTF8', newline='') as f:
# #     writer = csv.writer(f)
# #
# #
# # # write a row to the csv file
# # writer.writerows(agent)
# #
# # writer.writerows(data2)
# #
# # # close the file
# # f.close()
