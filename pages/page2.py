import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback, Input, Output, State
from dash import dcc, html
from dash.exceptions import PreventUpdate

from Second import *

dash.register_page(__name__)
df = px.data.gapminder()

# Initial values for calculations
total_calculations = 25
completed_calculations = 0

layout = html.Div([

    html.Div(
        children=[

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Remind me rules",
                            id="popup-button",
                            color="primary",
                            className="mr-3",
                            outline=True,
                            size="sm",
                        )),
                    dbc.Col(
                        html.Div(id='text-container',
                                 children=[
                                     html.P("Please, navigate to other page."),
                                 ],
                                 style={'display': 'none'}
                                 )
                    ),
                    dbc.Col(
                        # html.Div(id='link-container',
                        #          children=[
                        dcc.Link('See the final results', href='/page3'),
                        # ],
                        # style={'display': 'none'}
                        # style={'text-align': 'right', 'margin-right': '20px'}
                        # )
                    ),
                    dbc.Col(
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Rules"),
                                dbc.ModalBody(
                                    "You should want state number 7 and action 4, which means that you like 22 "
                                    "degrees and you want to pay as little as possible for it."
                                    "Rate the results 25 times or click on one of the blue/red button"
                                    " to evaluate the results without more rating."),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="close-button", className="ml-auto", color="secondary")
                                ),
                            ],
                            id="popup-modal",
                            centered=True,
                            is_open=False,
                        ),
                        width="auto",
                    ),
                ]),
            dbc.Row(
                [
                    dbc.Col(
                        html.Button('Run', id='submit-val', n_clicks=0, disabled=False),
                        # color="success", #className="mr-3",
                        # outline=True, size="sm"
                        # ),
                        width="auto",
                    ),
                    dbc.Col(html.Div(), width="auto"),
                    dbc.Col(
                        html.Div(
                            children=[
                                dbc.Button("I like the results and I do not want to tune parameters any more.",
                                           id="button-1", color="info", outline=True, size="sm"),
                                dbc.Button("I am bored and I do not want to rate anymore.", id="button-2",
                                           color="danger", outline=True, size="sm"),
                            ],
                            className="d-flex justify-content-end",
                        ),
                        width=True,

                    ),
                ],
                justify="between",
            ),

            # ]),
        ],
        style={'margin-left': '1cm'}
    ),

    html.Br(),
    # html.Div(
    #     children=[
    #         dbc.Row(
    #             [

    # dbc.Col(
    #
    # ),
    # dbc.Col(
    html.Div(
        children=[
            html.Div(id='slider-container',
                     children=[
                         dbc.Row(
                             [
                                 dbc.Col(
                                     html.P(
                                         "Rate how much you like the results of states and actions for last 20 steps: ")),
                                 dbc.Col(
                                     dcc.Slider(1, 5, 1,
                                                value=None,
                                                id='my_slider'
                                                ),

                                 ),
                                 dbc.Col(
                                     # html.Div(id='button2-container', children=[
                                     html.Button('Leave the same mark',
                                                 id='mark_button',
                                                 n_clicks=0),  # ],
                                     # style={'display': 'none'})
                                 )

                             ])],
                     style={'display': 'none',
                            # 'justify-content': 'center',
                            # 'width': '50%',
                            'margin': '0 auto'
                            }
                     )],
        style={'margin-left': '1cm'}

    ),
    html.Div(
        # className="main-container",
        children=[
            html.Div(className="spacer"),  # Add spacer for horizontal alignment
            html.Div(
                # className="content-container",
                children=[
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6("Calculations Completed"),
                                    html.Div(
                                        id="progress-value",
                                        children=f"{completed_calculations} / {total_calculations}",
                                        # className="progress-value",
                                    ),
                                ]
                            )
                        ],
                        # className="progress-card",
                    ),
                ],
            ),
        ],
    ),
    html.Br(),
    html.Div(id='text-wait',
             children=[
                 html.P("Please, wait a while the simulation is running."),
             ],
             style={'display': 'none'}
             ),
    html.Br(),
    html.P("There are the results of number of states and actions in last 20 time steps."),
    html.Div(
        dcc.Graph(id='bar_hist_states'),
        style={'width': '50%', 'display': 'inline-block'}),
    html.Div(
        dcc.Graph(id='bar_hist_actions'),
        style={'width': '50%', 'display': 'inline-block'}),
    html.Br(),
    html.Div(
        dcc.Graph(id='plot_states'),
        style={'width': '50%', 'display': 'inline-block'}),
    html.Div(
        dcc.Graph(id='plot_actions'),
        style={'width': '50%', 'display': 'inline-block'}),
    # dcc.Store(id='my-data-store', storage_type='memory')
    dcc.Store(id='store-data', data=[], storage_type='memory'),
    # dcc.Location(id='url', refresh=False),
    # dcc.Location(id='url', refresh=False)
    # dcc.Store(id='store-data2', data=[], storage_type='memory')

]
)


@callback(
    Output("popup-modal", "is_open"),
    [Input("popup-button", "n_clicks"), Input("close-button", "n_clicks")],
    [dash.dependencies.State("popup-modal", "is_open")],
)
def toggle_popup_modal(popup_clicks, close_clicks, is_open):
    if popup_clicks or close_clicks:
        return not is_open
    return is_open


@callback(
    [Output('store-data', 'data'),
     Output('submit-val', 'disabled'),
     Output('text-container', 'style'),
     # Output('link-container', 'style')
     ],
    [Input('submit-val', 'n_clicks'),
     Input('my_slider', 'value'),
     Input('mark_button', 'n_clicks'),  # Input('url', 'pathname')
     Input('button-1', 'n_clicks'),
     Input('button-2', 'n_clicks'),
     Input('store-data', 'data')]
)
def store_data(n_clicks, value, mark_clicks, m_clicks, mm_clicks, data):
    ctx = dash.callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]
    if n_clicks is not None and n_clicks != 0 and not data and value is None and \
            (m_clicks is None or m_clicks == 0) and (mm_clicks is None or mm_clicks == 0):
        var_init = initialization(500, "FALSE", 20)
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
        why = "None"

        obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
                          data1_actions,
                          data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V, why]

        return obj_store_data, True, {'display': 'none'}  # ,{'display': 'none'}

    elif n_clicks is not None and n_clicks != 0 and data and value is not None and (
            m_clicks == 0 or m_clicks is None) and \
            (mm_clicks == 0 or mm_clicks is None):
        if triggered_component == 'my_slider' or (triggered_component == 'mark_button' and mark_clicks > 0):
            var_init = initialization(500, "FALSE", 20)
            system = var_init[0]
            agent = var_init[1]
            data2 = var_init[2]
            user = var_init[3]
            data1 = var_init[4]

            agent.w = data[0]
            agent.nu = data[1]
            agent.gam = np.array(data[2])
            agent.model = np.array(data[3])
            agent.mi = np.array(data[4])
            agent.ri = np.array(data[5])
            agent.r = np.array(data[6])
            agent.V = np.array(data[7])
            data2.states = data[8]
            data2.actions = data[9]
            data2.t = data[10]
            data1.states = data[11]
            data1.actions = data[12]
            data1.marks = data[13]
            data1.t = data[14]
            user.gam = np.array(data[15])
            user.model = np.array(data[16])
            user.mi = np.array(data[17])
            user.ri = np.array(data[18])
            user.r = np.array(data[19])
            user.V = np.array(data[20])

            m = value
            mm = 0
            if m == 0:
                m = data1.marks[data1.t]

            k = int(data1.marks[data1.t])
            if m - int(k) > 0:
                mm = 2
            if m - int(k) < 0:
                mm = 0
            if m - int(k) == 0:
                mm = 1

            data1.marks.append(m)
            data1.states.append(mm)

            data1.t = data1.t + 1
            user.learn(data1)

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
            why = "None"

            obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
                              data1_actions,
                              data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V, why]
            return obj_store_data, True, {'display': 'none'}  # , {'display': 'none'}

    elif n_clicks is not None and n_clicks != 0 and data and value is not None and \
            (m_clicks is not None and m_clicks != 0) or (mm_clicks is not None and mm_clicks != 0):
        if (triggered_component == 'button-1' and m_clicks > 0) or (
                triggered_component == 'button-2' and mm_clicks > 0):
            var_init = initialization(500, "FALSE", 20)
            system = var_init[0]
            agent = var_init[1]
            data2 = var_init[2]
            user = var_init[3]
            data1 = var_init[4]

            agent.w = data[0]
            agent.nu = data[1]
            agent.gam = np.array(data[2])
            agent.model = np.array(data[3])
            agent.mi = np.array(data[4])
            agent.ri = np.array(data[5])
            agent.r = np.array(data[6])
            agent.V = np.array(data[7])
            data2.states = data[8]
            data2.actions = data[9]
            data2.t = data[10]
            data1.states = data[11]
            data1.actions = data[12]
            data1.marks = data[13]
            data1.t = data[14]
            user.gam = np.array(data[15])
            user.model = np.array(data[16])
            user.mi = np.array(data[17])
            user.ri = np.array(data[18])
            user.r = np.array(data[19])
            user.V = np.array(data[20])

            while data2.t <= data2.length_sim:
                agent.calculate_alfa()
                data2 = system.generate(agent, data2)
                agent.learn(data2)

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

            if m_clicks > 0:
                why = ['liked it']
            elif mm_clicks > 0:
                why = ['bored']

            obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
                              data1_actions,
                              data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V, why]

            return obj_store_data, True, {'display': 'block', 'color': 'red', 'font-size': '30px',
                                          'margin': 'center'}  # {
            # 'text-align': 'right', 'margin-right': '20px'}

    else:
        raise PreventUpdate


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
    Output(component_id='slider-container', component_property='style'),
    # Output(component_id='button-container', component_property='style'),
    # Output(component_id='button2-container', component_property='style')
    Output("progress-value", "children"),
    # Output('link-container', 'style')
],
    [Input('store-data', 'data')]
    # [Input(component_id='my_slider', component_property='value')]
)
def update_figure(data):
    if data:
        if data[14] < 24 and data[21] == "None":
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

            completed = data1.t

            return [bar_hist_states, bar_hist_actions, {'display': 'block',
                                                        # 'justify-content': 'center',
                                                        # 'width': '50%',
                                                        'margin': '0 auto'},  # {'display': 'block'}

                    f"{completed} out of {total_calculations}",
                    # {'display': 'none'}

                    ]
        elif data[14] >= 25 or data[21] == ["liked it"] or data[21] == ['bored']:
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

            bar_hist_states.update_layout(title='Number of each state in last 20 time steps',
                                          transition_duration=500)
            bar_hist_actions.update_layout(title='Number of each action in last 20 time steps',
                                           transition_duration=500)

            completed = 25
            return [bar_hist_states, bar_hist_actions, {'display': 'none',
                                                        # 'justify-content': 'center',
                                                        # 'width': '50%',
                                                        'margin': '0 auto'},  # {'display': 'block'}

                    f"{completed} out of {total_calculations}",
                    # {'text-align': 'right', 'margin-right': '20px'}
                    ]

    else:
        PreventUpdate


@callback([
    Output(component_id='plot_states', component_property='figure'),
    Output(component_id='plot_actions', component_property='figure')
],
    [Input('store-data', 'data')]
    # [Input(component_id='my_slider', component_property='value')]
)
def update_plots(data):
    if data:
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

        data_states = data2.states[(data2.t - data1.length_sim):data2.t]
        data_actions = data2.actions[(data2.t - data1.length_sim):data2.t - 1]

        data_state = {'Time steps': list(np.arange(0, 20)),
                      'States': data_states}

        data_action = {'Time steps': list(np.arange(0, 19)),
                       'Actions': data_actions}

        df_s = pd.DataFrame(data_state)
        # print(df_s)
        df_a = pd.DataFrame(data_action)
        # df_s = load_object("data_states")

        plot_states = go.Figure(data=[go.Scatter(x=df_s['Time steps'], y=df_s['States'], mode='markers')])
        plot_actions = go.Figure(data=[go.Scatter(x=df_a['Time steps'], y=df_a['Actions'], mode='markers')])

        plot_states.update_layout(transition_duration=500)
        plot_states.update_layout(title='Evolution of states in time',
                                  xaxis_title='Time steps',
                                  yaxis_title='States',
                                  yaxis={'range': [-1, 14], 'fixedrange': True}
                                  )
        plot_actions.update_layout(transition_duration=500)
        plot_actions.update_layout(title='Evolution of actions in time',
                                   xaxis_title='Time steps',
                                   yaxis_title='Actions',
                                   yaxis={'range': [-1, 7], 'fixedrange': True}
                                   )

        return [plot_states, plot_actions]
    else:
        raise PreventUpdate


@callback(
    Output('text-wait', 'style'),
    [Input('button-1', 'n_clicks'),
     Input('button-2', 'n_clicks')]
)
def handle_submission(m_clicks, mm_clicks):
    if (m_clicks is not None and m_clicks > 0) or (mm_clicks is not None and mm_clicks > 0):
        # Perform actions with the submitted data
        # For example, save the data to a database or perform calculations
        # print("Form submitted!")
        # Reset the button to prevent multiple submissions
        return {'display': 'block',
                'color': 'red',
                'font-size': '30px',
                'margin': 'center'}
    else:
        return {'display': 'none'}

# @callback(
#     [Output('store-data', 'data')],
#     [Input('button-1', 'n_clicks'),
#      Input('button-2', 'n_clicks')  # Input('url', 'pathname')
#      ],
#     [State('store-data', 'data')]
# )
# def complete_calculation(n_clicks, m_clicks, data):
#     ctx = dash.callback_context
#     triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]
#     if (n_clicks is not None and n_clicks != 0) or (m_clicks is not None and m_clicks != 0) and not data:
#
#         var_init = initialization(500, "FALSE", 20)
#         system = var_init[0]
#         agent = var_init[1]
#         data2 = var_init[2]
#         user = var_init[3]
#         data1 = var_init[4]
#
#         agent.w = data[0]
#         agent.nu = data[1]
#         agent.gam = np.array(data[2])
#         agent.model = np.array(data[3])
#         agent.mi = np.array(data[4])
#         agent.ri = np.array(data[5])
#         agent.r = np.array(data[6])
#         agent.V = np.array(data[7])
#         data2.states = data[8]
#         data2.actions = data[9]
#         data2.t = data[10]
#         data1.states = data[11]
#         data1.actions = data[12]
#         data1.marks = data[13]
#         data1.t = data[14]
#         user.gam = np.array(data[15])
#         user.model = np.array(data[16])
#         user.mi = np.array(data[17])
#         user.ri = np.array(data[18])
#         user.r = np.array(data[19])
#         user.V = np.array(data[20])
#
#         while data2.t <= data2.length_sim:
#             agent.calculate_alfa()
#             data2 = system.generate(agent, data2)
#             agent.learn(data2)
#
#         w = agent.w
#         # s0 = data2.states[data2.t]
#         nu = agent.nu
#         gam = agent.gam
#         model = agent.model
#         mi = agent.mi
#         ri = agent.ri
#         r = agent.r
#         V = agent.V
#         data_states = data2.states
#         data_actions = data2.actions
#         data_t = data2.t
#         data1_states = data1.states
#         data1_actions = data1.actions
#         data1_marks = data1.marks
#         data1_t = data1.t
#         user_gam = user.gam
#         user_model = user.model
#         user_mi = user.mi
#         user_ri = user.ri
#         user_r = user.r
#         user_V = user.V
#
#         obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
#                           data1_actions,
#                           data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V]
#
#         return obj_store_data
#
#     else:
#         raise PreventUpdate

# @callback(
#     Output('submit-val', 'style'),
#     [Input('submit-val', 'n_clicks')]
# )
# def hide_button(n_clicks):
#     if n_clicks > 0:
#         return {'display': 'none'}
#     else:
#         return {'display': 'block'}


# @callback(Output('data-store', 'data'),
#           [Input('url', 'pathname')])
# def store_data_to3(pathname):
#     # Perform data storage here
#     data = {'message': 'Data from page 2'}
#     return json.dumps(data)

# @callback(
#     Output('my-component', 'children'),
#     [Input('my-data-store', 'data')]
# )
# def update_page(data):
#     # Use the stored data to initialize or populate the components on the page
#     if data is not None:
#         # Do something with the data
#         return html.Div('Stored Data: {}'.format(data))
#     else:
#         return html.Div('No data available')
# @callback(
#     Output('store-data', 'data'),
#     [Input('my_slider', 'value'),
#      Input('store-data', 'data')]
# )
# def store_data2(data, value):
#     if data and value is not None:
#         var_init = initialization(500, "FALSE", 10)
#         system = var_init[0]
#         agent = var_init[1]
#         data2 = var_init[2]
#         user = var_init[3]
#         data1 = var_init[4]
#
#         agent.w = data[0]
#         agent.nu = data[1]
#         agent.gam = np.array(data[2])
#         agent.model = np.array(data[3])
#         agent.mi = np.array(data[4])
#         agent.ri = np.array(data[5])
#         agent.r = np.array(data[6])
#         agent.V = np.array(data[7])
#         data2.states = data[8]
#         data2.actions = data[9]
#         data2.t = data[10]
#         data1.states = data[11]
#         data1.actions = data[12]
#         data1.marks = data[13]
#         data1.t = data[14]
#         user.gam = np.array(data[15])
#         user.model = np.array(data[16])
#         user.mi = np.array(data[17])
#         user.ri = np.array(data[18])
#         user.r = np.array(data[19])
#         user.V = np.array(data[20])
#
#         m = value
#         mm = 0
#         if m == 0:
#             m = data1.marks[data1.t]
#
#         k = int(data1.marks[data1.t])
#         if m - int(k) > 0:
#             mm = 2
#         if m - int(k) < 0:
#             mm = 0
#         if m - int(k) == 0:
#             mm = 1
#
#         data1.marks.append(m)
#         data1.states.append(mm)
#
#         data1.t = data1.t + 1
#         user.learn(data1)
#
#         if data1.t <= data2.length_sim / data1.length_sim:
#             user.calculate_alfa()
#             s1 = data1.states[data1.t]
#             a = dnoise(user.r[:, s1])
#             data1.actions.append(a)
#             data1, data2, agent = system.small_loop(agent, data2, data1)
#
#         w = agent.w
#         # s0 = data2.states[data2.t]
#         nu = agent.nu
#         gam = agent.gam
#         model = agent.model
#         mi = agent.mi
#         ri = agent.ri
#         r = agent.r
#         V = agent.V
#         data_states = data2.states
#         data_actions = data2.actions
#         data_t = data2.t
#         data1_states = data1.states
#         data1_actions = data1.actions
#         data1_marks = data1.marks
#         data1_t = data1.t
#         user_gam = user.gam
#         user_model = user.model
#         user_mi = user.mi
#         user_ri = user.ri
#         user_r = user.r
#         user_V = user.V
#
#         obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
#                           data1_actions,
#                           data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V]
#
#         return obj_store_data
#     else:
#         PreventUpdate
#         return None

# @callback([
#     Output(component_id='bar_hist_states', component_property='figure'),
#     Output(component_id='bar_hist_actions', component_property='figure'),
# ],
#     [Input('store-data', 'data'),
#      Input(component_id='my_slider', component_property='value')]
# )
# def improvement(data):
#     if data:
#         var_init = initialization(500, "TRUE", 10)
#
#         agent = var_init[1]
#         data2 = var_init[2]
#         data1 = var_init[4]
#
#         data2.states = data[8]
#         data2.actions = data[9]
#         data2.t = data[10]
#         data1.states = data[11]
#         data1.actions = data[12]
#         data1.marks = data[13]
#         data1.t = data[14]
#
#         data_states = data2.states[(data2.t - data1.length_sim):data2.t]
#         data_actions = data2.actions[(data2.t - data1.length_sim):data2.t - 1]
#
#         number_of_s = np.zeros(agent.ss)
#         number_of_a = np.zeros(agent.aa)
#
#         data_states = np.array(data_states)
#         data_actions = np.array(data_actions)
#
#         for j in range(agent.ss):
#             number_of_s[j] = np.sum(data_states[:] == j)
#
#         for k in range(agent.aa):
#             number_of_a[k] = np.sum(data_actions[:] == k)
#
#         data_state = {'States': list(np.arange(0, agent.ss)),
#                       'Number of states': number_of_s}
#
#         data_action = {'Actions': list(np.arange(0, agent.aa)),
#                        'Number of actions': number_of_a}
#
#         df_s = pd.DataFrame(data_state)
#         # print(df_s)
#         df_a = pd.DataFrame(data_action)
#         # df_s = load_object("data_states")
#         bar_hist_states = px.bar(df_s, x='States', y='Number of states')
#         bar_hist_actions = px.bar(df_a, x='Actions', y='Number of actions')
#
#         bar_hist_states.update_layout(transition_duration=500)
#         bar_hist_actions.update_layout(transition_duration=500)
#
#         return [bar_hist_states, bar_hist_actions]
#     else:
#         PreventUpdate

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
