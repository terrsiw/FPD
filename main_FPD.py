# from FPD_functions import Agent
# from FPD_functions import Data2
import numpy as np
import pandas as pd

# from FPD_functions import initialization
from FPD_functions import *
# from Second_closed_loop import *
# import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

agent, data2, system = main(500, "FALSE")
# agent, data2, system, data1, user = main_second(500, 'FALSE', 10)

data_states = data2.states
data_actions = data2.actions

number_of_s = np.zeros(agent.ss)
number_of_a = np.zeros(agent.aa)

for j in range(agent.ss):
    number_of_s[j] = np.sum(data_states[:] == j)

for k in range(agent.aa):
    number_of_a[k] = np.sum(data_actions[:] == k)

data = {'States': list(np.arange(0, agent.ss)),
        'Number of states': number_of_s}

df_s = pd.DataFrame(data)
# df_a = pd.DataFrame(number_of_a, columns=['Number of actions'])

df_s.to_csv('data_states_try.csv')

# px.bar(df_s, x='States', y='Number of states')

# plt.figure(1)
# plt.subplot(211)
# plt.bar(range(agent.ss), number_of_s, width=0.4)
# plt.subplot(212)
# plt.bar(range(agent.aa), number_of_a, width=0.4)
# plt.show()

# plt.figure(1)
# plt.subplot(211)
# plt.bar(range(system.ss), pocets, width=0.4)
# plt.subplot(212)
# plt.bar(range(agent.aa), poceta, width=0.4)
# plt.show()

# with open('C:/Users/Tereza/Desktop/marko/data/data1.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#
#
# # write a row to the csv file
# writer.writerows(agent)
#
# writer.writerows(data2)
#
# # close the file
# f.close()
