import dash
import plotly.express as px
from dash import dcc, html
from dash import callback, Input, Output
# import pandas as pd

# from Second_closed_loop import *
from FPD_functions import *

#
# agent, data2, system, data1, user = main_second(500, 'FALSE', 10)
# import pathlib
# import dash_core_components as dcc
# import dash_html_components as html
# import pandas as pd
# import plotly.graph_objs as go
# from dash.dependencies import Input, Output
# from plotly import tools

dash.register_page(__name__)
df = px.data.gapminder()


layout = html.Div(
    [
        html.Div(
            dcc.Slider(0, 1000, 100,
                       value=500,
                       id='my-slider'
                       )),
        html.Div(
            dcc.Graph(id='bar-fig_states'))

    ]
)


# dcc.Graph(id='bar-fig_states',
#           figure=px.bar(df_s, x='States', y='Number of states'))
# dcc.Graph(id='bar-fig-actions',
#           figure=px.bar(df_a, x='Actions', y='Number of actions'))
# dcc.Slider(0, 10000, 50,
#            value=500,
#            id='my-slider'
#            ),
@callback(
    Output(component_id='bar-fig_states', component_property='figure'),
    Input(component_id='my-slider', component_property='value'))
def update_figure(value):
    # agent, data2, system = main(value, "FALSE")
    #
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
    # # fig = plt.figure(figsize = (10, 5))
    #
    # data_states = data2.states
    # # data_actions = data2.actions
    #
    # number_of_s = np.zeros(agent.ss)
    # # number_of_a = np.zeros(agent.aa)
    #
    # for j in range(agent.ss):
    #     number_of_s[j] = np.sum(data_states[:] == j)
    #
    # # for k in range(agent.aa):
    # #     number_of_a[k] = np.sum(data_actions[:] == k)
    #
    # data_state = {'States': list(np.arange(0, agent.ss)),
    #               'Number of states': number_of_s}
    #
    # # data_actions = {'Actions': list(np.arange(0, agent.aa)),
    # # 'Number of actions': number_of_a}
    #
    # df_s = pd.DataFrame(data_state)
    # # df_a = pd.DataFrame(data_actions)
    df_s = load_object("data_states")
    fig = px.bar(df_s, x='States', y='Number of states')

    fig.update_layout(transition_duration=500)

    return fig


# from demo_utils import demo_callbacks, demo_explanation

# fig = plt.figure(figsize = (10, 5))
# # states = [range(agent.ss), number_of_s]
# df_s = pd.DataFrame(number_of_s, columns='Number of states')
# # df_s = pd.DataFrame(states, columns=['States', 'Number of states'])
# df_a = pd.DataFrame(number_of_a, columns='Number of actions')
# # actions = [range(agent.aa), number_of_a]
# # df_a = pd.DataFrame(actions, columns=['Actions', 'Number of actions'])

# df_s = load_object("data_states")