import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash
import pandas as pd
import plotly.express as px
from dash import callback, Input, Output
from dash import dcc, html
from dash.exceptions import PreventUpdate

from Second import *

dash.register_page(__name__)

layout = html.Div([
    dcc.Markdown('# These are the results'),
    html.Br(),
    html.Button("Show Plots", id='update-button', n_clicks=0),
    # html.Div(id='plots-container'),
    dcc.Store(id='store-data', storage_type='memory'),
    html.Br(),
    html.Div(
        dcc.Graph(id='bar_hist_all_states'),
        style={'width': '50%', 'display': 'inline-block'}),
    html.Div(
        dcc.Graph(id='bar_hist_all_actions'),
        style={'width': '50%', 'display': 'inline-block'}),
    # dcc.Store(id='store-data', storage_type='memory'),
]
)


@callback(
    [Output(component_id='bar_hist_all_states', component_property='figure'),
     Output(component_id='bar_hist_all_actions', component_property='figure'),
     ],
    [Input('update-button', 'n_clicks'),
     Input('store-data', 'data')]
)
def display_graph(d_clicks, data):
    # if pathname == '/page3':
    if d_clicks is not None and d_clicks != 0 and data is not None:
        # data = json.loads(data)

        var_init = initialization(500, "FALSE", 20)

        agent = var_init[1]
        data2 = var_init[2]
        data1 = var_init[4]

        data2.states = data[8]
        data2.actions = data[9]
        data2.t = data[10]
        data1.states = data[11]
        data1.actions = data[12]
        data1.marks = data[13]
        data1.t = data[14]

        data_states = data2.states
        data_actions = data2.actions

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
        bar_hist_all_states = px.bar(df_s, x='States', y='Number of states')
        bar_hist_all_actions = px.bar(df_a, x='Actions', y='Number of actions')

        bar_hist_all_states.update_layout(transition_duration=500)
        bar_hist_all_actions.update_layout(transition_duration=500)

        # bar_plot1 = dcc.Graph(figure=bar_hist_states)
        # bar_plot2 = dcc.Graph(figure=bar_hist_actions)

        return [bar_hist_all_states, bar_hist_all_actions]

        # return html.Div(html.Div(bar_plot1,
        #                          style={'width': '50%', 'display': 'inline-block'}),
        #                 html.Div(bar_plot2,
        #                          style={'width': '50%', 'display': 'inline-block'}))
    else:
        return {}




