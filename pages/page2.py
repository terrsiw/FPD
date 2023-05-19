import dash
import plotly.express as px
from dash import dcc, html
from dash import callback, Input, Output
import json
import pandas as pd

from Second import *

# from FPD_functions import *

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
        html.Button('Run', id='submit-val', n_clicks=0),
        html.Div(
            dcc.Slider(1, 5, 1,
                       value=3,
                       id='my_slider'
                       )),
        html.Br(),
        html.Div(
            dcc.Graph(id='bar_hist_states')),
        html.Br(),
        html.Div(
            dcc.Graph(id='bar_hist_actions')),

        dcc.Store(id='store-data', data=[], storage_type='memory'),
        dcc.Store(id='store-data2', data=[], storage_type='memory')

    ]
)


@callback(
    Output('store-data', 'data'),
    Input('submit-val', 'n_clicks')
)
def store_data(n_clicks):
    var_init = initialization(500, "FALSE", 10)
    system = var_init[0]
    agent = var_init[1]
    data2 = var_init[2]
    user = var_init[3]
    data1 = var_init[4]

    if data1.t <= data2.length_sim / data1.length_sim:
        user.calculate_alfa()
        s1 = data1.states[data1.t]
        a = dnoise(user.r[:, s1])
        data1.actions.append(a)
        data1, data2, agent = system.small_loop(agent, data2, data1)

    w = agent.w
    # s0 = data2.states[data2.t]
    nu = agent.nu
    gam = agent.gam
    model = agent.model
    mi = agent.mi
    ri = agent.ri
    r = agent.r
    V = agent.V
    data_states = data2.states
    data_actions = data2.actions
    data_t = data2.t
    data1_states = data1.states
    data1_actions = data1.actions
    data1_marks = data1.marks
    data1_t = data1.t
    user_gam = user.gam
    user_model = user.model
    user_mi = user.mi
    user_ri = user.ri
    user_r = user.r
    user_V = user.V

    obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states, data1_actions,
                      data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V]

    return obj_store_data


# @callback(
#     Output('bar-hist_states', component_property='figure'),
#     Input('store-data', 'data')
# )
# def create_graph(data):
#     # data2 = pd.read_json(data, orient='split')
#     datasets = json.loads(data)
#     dff = pd.read_json(datasets['df_1'], orient='split')
#     print(dff.ss)

@callback([
    Output(component_id='bar_hist_states', component_property='figure'),
    Output(component_id='bar_hist_actions', component_property='figure'),
],
    [Input('store-data', 'data')]
    # [Input(component_id='my_slider', component_property='value')]
)
def update_figure(data):
    # var_init = initialization(500, "FALSE", 10)
    # system = var_init[0]
    # agent = var_init[1]
    # data2 = var_init[2]
    # user = var_init[3]
    # data1 = var_init[4]
    #
    # if data1.t <= data2.length_sim / data1.length_sim:
    #     user.calculate_alfa()
    #     s1 = data1.states[data1.t]
    #     a = dnoise(user.r[:, s1])
    #     data1.actions.append(a)
    #     data1, data2, agent = system.small_loop(agent, data2, data1)

    # agent, data2, system = main(value, "FALSE")

    var_init = initialization(500, "FALSE", 10)
    system = var_init[0]
    agent = var_init[1]
    data2 = var_init[2]
    user = var_init[3]
    data1 = var_init[4]

    agent.w = data[0]
    agent.nu = data[1]
    agent.gam = data[2]
    agent.model = data[3]
    agent.mi = data[4]
    agent.ri = data[5]
    agent.r = data[6]
    agent.V = data[7]
    data2.states = data[8]
    data2.actions = data[9]
    data2.t = data[10]
    data1.states = data[11]
    data1.actions = data[12]
    data1.marks = data[13]
    data1.t = data[14]
    user.gam = data[15]
    user.model = data[16]
    user.mi = data[17]
    user.ri = data[18]
    user.r = data[19]
    user.V = data[20]

    data_states = data2.states[(data2.t - data1.length_sim):data2.t]
    data_actions = data2.actions[(data2.t - data1.length_sim):data2.t - 1]

    number_of_s = np.zeros(agent.ss)
    number_of_a = np.zeros(agent.aa)

    data_states = np.array(data_states)
    data_actions = np.array(data_actions)

    for j in range(agent.ss):
        number_of_s[j] = np.sum(data_states[:] == j)

    for k in range(agent.aa):
        number_of_a[k] = np.sum(data_actions[:] == k)

    data_state = {'States': list(np.arange(0, agent.ss)),
                  'Number of states': number_of_s}

    data_action = {'Actions': list(np.arange(0, agent.aa)),
                   'Number of actions': number_of_a}

    df_s = pd.DataFrame(data_state)
    # print(df_s)
    df_a = pd.DataFrame(data_action)
    # df_s = load_object("data_states")
    bar_hist_states = px.bar(df_s, x='States', y='Number of states')
    bar_hist_actions = px.bar(df_a, x='Actions', y='Number of actions')

    bar_hist_states.update_layout(transition_duration=500)
    bar_hist_actions.update_layout(transition_duration=500)

    return [bar_hist_states, bar_hist_actions]


# @callback(
#     Output('store-data2', 'data'),
#     [Input('store-data', 'data'),
#      Input('my_slider', 'value')]
# )
# def store_data2(value, data):
#     var_init = initialization(500, "FALSE", 10)
#     system = var_init[0]
#     agent = var_init[1]
#     data2 = var_init[2]
#     user = var_init[3]
#     data1 = var_init[4]
#
#     agent.w = data[0]
#     agent.nu = data[1]
#     agent.gam = data[2]
#     agent.model = data[3]
#     agent.mi = data[4]
#     agent.ri = data[5]
#     agent.r = data[6]
#     agent.V = data[7]
#     data2.states = data[8]
#     data2.actions = data[9]
#     data2.t = data[10]
#     data1.states = data[11]
#     data1.actions = data[12]
#     data1.marks = data[13]
#     data1.t = data[14]
#     user.gam = data[15]
#     user.model = data[16]
#     user.mi = data[17]
#     user.ri = data[18]
#     user.r = data[19]
#     user.V = data[20]
#
#     m = value
#     mm = 0
#     if m == 0:
#         m = data1.marks[data1.t]
#
#     k = int(data1.marks[data1.t])
#     if m - int(k) > 0:
#         mm = 2
#     if m - int(k) < 0:
#         mm = 0
#     if m - int(k) == 0:
#         mm = 1
#
#     data1.marks.append(m)
#     data1.states.append(mm)
#
#     data1.t = data1.t + 1
#     user.learn(data1)
#
#     if data1.t <= data2.length_sim / data1.length_sim:
#         user.calculate_alfa()
#         s1 = data1.states[data1.t]
#         a = dnoise(user.r[:, s1])
#         data1.actions.append(a)
#         data1, data2, agent = system.small_loop(agent, data2, data1)
#
#     w = agent.w
#     # s0 = data2.states[data2.t]
#     nu = agent.nu
#     gam = agent.gam
#     model = agent.model
#     mi = agent.mi
#     ri = agent.ri
#     r = agent.r
#     V = agent.V
#     data_states = data2.states
#     data_actions = data2.actions
#     data_t = data2.t
#     data1_states = data1.states
#     data1_actions = data1.actions
#     data1_marks = data1.marks
#     data1_t = data1.t
#     user_gam = user.gam
#     user_model = user.model
#     user_mi = user.mi
#     user_ri = user.ri
#     user_r = user.r
#     user_V = user.V
#
#     obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states, data1_actions,
#                       data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V]
#
#     return obj_store_data

# @callback([
#     Output(component_id='bar_hist_states', component_property='figure'),
#     Output(component_id='bar_hist_actions', component_property='figure'),
# ],
#
#     [Input(component_id='my_slider', component_property='value')]
# )
# def improvement(value):
#     var_init = initialization(500, "FALSE", 10)
#     system = var_init[0]
#     agent = var_init[1]
#     data2 = var_init[2]
#     user = var_init[3]
#     data1 = var_init[4]
#
#
#     agent.w = w
#     agent.s0 = s0
#     agent.nu = nu
#     agent.gam = gam
#     agent.model = model
#     agent.mi = mi
#     agent.ri = ri
#     agent.r = r
#     agent.V = V
#     data2.states = data_states
#     data2.actions = data_actions
#     data2.t = data_t
#     data1.states = data1_states
#     data1.actions = data1_actions
#     data1.marks = data1_marks
#     data1.t = data1_t
#
#
#     m = value
#     mm = 0
#     if m == 0:
#         m = data1.marks[data1.t]
#     k = int(data1.marks[data1.t])
#
#     if m - int(k) > 0:
#         mm = 2
#     if m - int(k) < 0:
#         mm = 0
#     if m - int(k) == 0:
#         mm = 1
#
#     data1.marks.append(m)
#     data1.states.append(mm)
#
#     data1.t = data1.t + 1
#
#     if data1.t <= data2.length_sim / data1.length_sim:
#         user.calculate_alfa()
#         s1 = data1.states[data1.t]
#         a = dnoise(user.r[:, s1])
#         data1.actions.append(a)
#         data1, data2, agent = system.small_loop(agent, data2, data1)
#
#     # agent, data2, system = main(value, "FALSE")
#
#     data_states = data2.states[(data2.t - data1.length_sim):data2.t]
#     data_actions = data2.actions[(data2.t - data1.length_sim):data2.t - 1]
#
#     number_of_s = np.zeros(agent.ss)
#     number_of_a = np.zeros(agent.aa)
#
#     data_states = np.array(data_states)
#     data_actions = np.array(data_actions)
#
#     for j in range(agent.ss):
#         number_of_s[j] = np.sum(data_states[:] == j)
#
#     for k in range(agent.aa):
#         number_of_a[k] = np.sum(data_actions[:] == k)
#
#     data_state = {'States': list(np.arange(0, agent.ss)),
#                   'Number of states': number_of_s}
#
#     data_action = {'Actions': list(np.arange(0, agent.aa)),
#                    'Number of actions': number_of_a}
#
#     df_s = pd.DataFrame(data_state)
#     # print(df_s)
#     df_a = pd.DataFrame(data_action)
#     # df_s = load_object("data_states")
#     bar_hist_states = px.bar(df_s, x='States', y='Number of states')
#     bar_hist_actions = px.bar(df_a, x='Actions', y='Number of actions')
#
#     bar_hist_states.update_layout(transition_duration=500)
#     bar_hist_actions.update_layout(transition_duration=500)
#
#     return [bar_hist_states, bar_hist_actions]

# from demo_utils import demo_callbacks, demo_explanation

# fig = plt.figure(figsize = (10, 5))
# # states = [range(agent.ss), number_of_s]
# df_s = pd.DataFrame(number_of_s, columns='Number of states')
# # df_s = pd.DataFrame(states, columns=['States', 'Number of states'])
# df_a = pd.DataFrame(number_of_a, columns='Number of actions')
# # actions = [range(agent.aa), number_of_a]
# # df_a = pd.DataFrame(actions, columns=['Actions', 'Number of actions'])

# df_s = load_object("data_states")


# import dash
# import plotly.express as px
# from dash import dcc, html
# from dash import callback, Input, Output
# import pandas as pd
#
# from Closed_loop import *
#
# dash.register_page(__name__)
# df = px.data.gapminder()
#
# layout = html.Div(
#     [
#         # html.Button('Run', id='button-run', n_clicks=0),
#         html.Div(
#             dcc.Slider(1, 5, 1,
#                        value=3,
#                        id='my-slider'
#                        )),
#         html.Br(),
#         html.Div(
#             dcc.Graph(id='bar-fig_states')),
#         html.Br(),
#         html.Div(
#             dcc.Graph(id='bar-fig_actions'))
#
#     ]
# )
#
#
# @callback(
#     [
#         Output(component_id='bar-fig_states', component_property='figure'),
#         Output(component_id='bar-fig_actions', component_property='figure')
#     ],
#     Input(component_id='button-run', component_property='n_clicks')
#
#     #     [dash.dependencies.State('input-box', 'value')]
# )
# def create_graphs(btn):
#     var_init = initialization(500, 'FALSE', 10)
#     system = var_init[0]
#     agent = var_init[1]
#     data2 = var_init[2]
#     user = var_init[3]
#     data1 = var_init[4]
#
#     user.calculate_alfa()
#     data2, data1, agent = system.generate_second(agent, data2, user, data1)
#     data_states = data2.states
#     data_actions = data2.actions
#     number_of_s = np.zeros(agent.ss)
#     number_of_a = np.zeros(agent.aa)
#
#     for j in range(agent.ss):
#         number_of_s[j] = np.sum(data_states[:] == j)
#
#     for k in range(agent.aa):
#         number_of_a[k] = np.sum(data_actions[:] == k)
#
#     data_state = {'States': list(np.arange(0, agent.ss)),
#                   'Number of states': number_of_s}
#
#     data_actions = {'Actions': list(np.arange(0, agent.aa)),
#                     'Number of actions': number_of_a}
#
#     df_s = pd.DataFrame(data_state)
#     df_a = pd.DataFrame(data_actions)
#
#     fig1 = px.bar(df_s, x='States', y='Number of states')
#     fig2 = px.bar(df_a, x='Actions', y='Number of actions')
#     fig1.update_layout(transition_duration=500)
#     fig2.update_layout(transition_duration=500)
#
#     return fig1, fig2
#
#
# # @callback(
# #     [
# #         Output(component_id='bar-fig_states', component_property='figure'),
# #         Output(component_id='bar-fig_actions', component_property='figure')
# #     ],
# #     Input(component_id='my-slider', component_property='value')
# #
# #     #     [dash.dependencies.State('input-box', 'value')]
# # )
# # def update_graphs(value):
# #     data1 = generate_third(data1, value)
# #     user.learn(data1)
# #     user.learn()
# #     user.calculate_alfa()
# #     data2, data1, agent = system.generate_second(agent, data2, user, data1)
# #     data_states = data2.states
# #     data_actions = data2.actions
# #     number_of_s = np.zeros(agent.ss)
# #     number_of_a = np.zeros(agent.aa)
# #
# #     for j in range(agent.ss):
# #         number_of_s[j] = np.sum(data_states[:] == j)
# #
# #     for k in range(agent.aa):
# #         number_of_a[k] = np.sum(data_actions[:] == k)
# #
# #     data_state = {'States': list(np.arange(0, agent.ss)),
# #                   'Number of states': number_of_s}
# #
# #     data_actions = {'Actions': list(np.arange(0, agent.aa)),
# #                     'Number of actions': number_of_a}
# #
# #     df_s = pd.DataFrame(data_state)
# #     df_a = pd.DataFrame(data_actions)
# #
# #     fig1 = px.bar(df_s, x='States', y='Number of states')
# #     fig2 = px.bar(df_a, x='Actions', y='Number of actions')
# #     fig1.update_layout(transition_duration=500)
# #     fig2.update_layout(transition_duration=500)
# #
# #     return fig1, fig2
#
# #
# # data1 = generate_third(data1, m)
# # user.learn(data1)
#
# # @callback(
# #     Output(component_id='bar-fig_states', component_property='fig1'),
# #     Output(component_id='bar-fig_actions', component_property='fig2'),
# #     Input(component_id='button-run', component_property='click')
# #
# # )
# # def create_graphs(click):
# #     agent, data2, system, data1, user = main_second(500, 'FALSE', 10)
# # data_states = data2.states
# # data_actions = data2.actions
# # number_of_s = np.zeros(agent.ss)
# # number_of_a = np.zeros(agent.aa)
# #
# # for j in range(agent.ss):
# #     number_of_s[j] = np.sum(data_states[:] == j)
# #
# # for k in range(agent.aa):
# #     number_of_a[k] = np.sum(data_actions[:] == k)
# #
# # data_state = {'States': list(np.arange(0, agent.ss)),
# #               'Number of states': number_of_s}
# #
# # data_actions = {'Actions': list(np.arange(0, agent.aa)),
# #                 'Number of actions': number_of_a}
# #
# # df_s = pd.DataFrame(data_state)
# # df_a = pd.DataFrame(data_actions)
# #
# # fig1 = px.bar(df_s, x='States', y='Number of states')
# # fig2 = px.bar(df_a, x='Actions', y='Number of actions')
# # fig1.update_layout(transition_duration=500)
# # fig2.update_layout(transition_duration=500)
# #
# # return fig1, fig2
#
# # @callback(
# #     Output(component_id='bar-fig_states', component_property='figure'),
# #     Input(component_id='my-slider', component_property='value'))
# # def update_figure_states(value):
# #     # agent, data2, system = main(value, "FALSE")
# #     agent, data2, system, data1, user = main_second(500, 'FALSE', 10)
# #
# #     data_states = data2.states
# #     number_of_s = np.zeros(agent.ss)
# #
# #     for j in range(agent.ss):
# #         number_of_s[j] = np.sum(data_states[:] == j)
# #
# #     data_state = {'States': list(np.arange(0, agent.ss)),
# #                   'Number of states': number_of_s}
# #
# #     df_s = pd.DataFrame(data_state)
# #
# #     fig = px.bar(df_s, x='States', y='Number of states')
# #
# #     fig.update_layout(transition_duration=500)
# #
# #     number_of_a = np.zeros(agent.aa)
# #
# #     for k in range(agent.aa):
# #         number_of_a[k] = np.sum(data_actions[:] == k)
# #
# #     data_actions = {'Actions': list(np.arange(0, agent.aa)),
# #                     'Number of actions': number_of_a}
# #
# #     df_a = pd.DataFrame(data_actions)
# #     fig = px.bar(df_a, x='Actions', y='Number of actions')
# #
# #     fig.update_layout(transition_duration=500)
# #
# #     return fig
# #
# #
# # @callback(
# #     Output(component_id='bar-fig_actions', component_property='figure'),
# #     Input(component_id='my-slider', component_property='value'))
# # def update_figure_actions(value):
# #     agent, data2, system, data1, user = main_second(value, 'FALSE', 10)
# #     # agent, data2, system = main(value, "FALSE")
# #
# #     data_actions = data2.actions
# #     number_of_a = np.zeros(agent.aa)
# #
# #     for k in range(agent.aa):
# #         number_of_a[k] = np.sum(data_actions[:] == k)
# #
# #     data_actions = {'Actions': list(np.arange(0, agent.aa)),
# #                     'Number of actions': number_of_a}
# #
# #     df_a = pd.DataFrame(data_actions)
# #     fig = px.bar(df_a, x='Actions', y='Number of actions')
# #
# #     fig.update_layout(transition_duration=500)
# #
# #     return fig
