import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash
import pandas as pd
import plotly.express as px
from dash import callback, Input, Output
from dash import dcc, html
from dash.exceptions import PreventUpdate
# import psycopg2
from datetime import date

from Second import *

dash.register_page(__name__)

# conn = psycopg2.connect(
#     host="postgresql.r2.websupport.sk",
#     port=5432,
#     user="tereza_sivakova",
#     password="tereza_sivakovaUTIA123",
#     database="tereza_sivakova"
# )


# def save_data_to_database(data):
#     # Insert data into the database
#     cursor = conn.cursor()
#     insert_query = "INSERT INTO my_table (gender, age_category, rate_graph, rate_app, rate_theory, rate_usage, " \
#                    "comment, data, date_saved," \
#                    "email) VALUES (%s, %s,%s, %s,%s, %s,%s, %s,%s, %s)"
#     cursor.execute(insert_query, (data.get('gender', None), data.get('age', None), data.get('likability', None),
#                                   data.get('app_rate', None),
#                                   data.get('theory_rate', None), data.get('usage_rate', None),
#                                   data.get('comments', None),
#                                   data.get('data', None), data.get('today', None), data.get('email', None)
#                                   ))
#     conn.commit()
#     cursor.close()


layout = html.Div([
    html.Div(
        children=[
            dcc.Store(id='store-data', storage_type='memory'),
            # dcc.Location(id='url', refresh=False),
            dcc.Markdown('### These are the results with 500 time steps. You can see all states and actions that were '
                         'chosen during the decision-making process.'),
            html.Br(),
            html.Button("Show the final results", id='update-button', n_clicks=0),
            # html.Div(id='plots-container'),
            # dcc.Store(id='store-data2', storage_type='memory'),
            html.Br(),
            html.Div(
                dcc.Graph(id='bar_hist_all_states'),
                style={'width': '50%', 'display': 'inline-block'}),
            html.Div(
                dcc.Graph(id='bar_hist_all_actions'),
                style={'width': '50%', 'display': 'inline-block'}),
            # dcc.Store(id='store-data', storage_type='memory'),
            # html.Div([
            #     # Your existing layout components here
            #     dbc.Modal(
            #         [
            #             dbc.ModalHeader("Save Data"),
            #             dbc.ModalBody("Do you want to save the data before exiting?"),
            #             dbc.ModalFooter(
            #                 [
            #                     dbc.Button("Save", id="save-button", className="ml-auto", color="primary"),
            #                     dbc.Button("Discard", id="discard-button", className="mr-auto", color="danger"),
            #                 ]
            #             ),
            #         ],
            #         id="save-modal",
            #         centered=True,
            #         is_open=False,
            #     ),
            # ]),
            html.Div(
                children=[
                    html.H1("Questionnaire", style={'text-align': 'center'}),
                    html.P("Please fill in the following information:", style={'margin-bottom': '20px'}),
                    html.Div(
                        children=[
                            html.Label("Email (optional):", style={'display': 'inline-block', 'margin-right': '20px'}),
                            html.P("Attach your email, if you want to know the results of the survey or if we can "
                                   "contact you later to follow up on your answer", style={'font-size': 'small'}),
                            dcc.Input(
                                id="email",
                                type="email",
                                placeholder="Enter your email..",
                            ),
                        ],
                        style={'margin-bottom': '20px', 'text-align': 'center'}
                    ),
                    html.Div(
                        children=[
                            html.Label("Gender:", style={'display': 'inline-block', 'margin-right': '20px'}),
                            dbc.RadioItems(
                                options=[
                                    {'label': 'Male', 'value': 'male'},
                                    {'label': 'Female', 'value': 'female'}
                                ],
                                id='gender-radio',
                                inline=True,
                                style={'display': 'inline-block'}
                            ),
                        ],
                        style={'margin-bottom': '20px', 'text-align': 'center'}
                    ),
                    html.Div(
                        children=[
                            html.Label("Age: ", style={'display': 'inline-block', 'margin-right': '20px'}),
                            dbc.Checklist(
                                options=[
                                    {'label': 'under 20', 'value': 'under 20'},
                                    {'label': '20-30', 'value': '20-30'},
                                    {'label': '30-40', 'value': '30-40'},
                                    {'label': '40-50', 'value': '40-50'},
                                    {'label': '50-60', 'value': '50-60'},
                                    {'label': '60+', 'value': '60+'}
                                ],
                                id='age-checkbox',
                                inline=True,
                                style={'display': 'inline-block', 'vertical-align': 'middle'}
                            ),
                        ],
                        style={'margin-bottom': '20px', 'text-align': 'center'}
                    ),
                    html.Div(
                        style={'width': '80%', 'margin': '0 auto'},
                        children=[
                            html.Label('Rate how much you liked the finale results:'),
                            dcc.Slider(1, 5, 1,
                                       value=None,
                                       id='final_marks-slider'
                                       )
                        ]),
                    html.Div(
                        style={'width': '80%', 'margin': '0 auto'},
                        children=[
                            html.Label("Rate how good was your user experiences with the app:"),
                            dcc.Slider(1, 5, 1,
                                       value=None,
                                       id='app-rating-slider'
                                       )
                        ]),
                    html.Div(
                        style={'width': '80%', 'margin': '0 auto'},
                        children=[
                            html.Label("Rate how you liked how the app improved your results :"),
                            dcc.Slider(1, 5, 1,
                                       value=None,
                                       id='theory-rating-slider'
                                       )
                        ]),
                    html.Div(
                        style={'width': '80%', 'margin': '0 auto'},
                        children=[
                            html.Label("Rate how likely you would use it for some decision-making recommendation "
                                       "problem:"),
                            dcc.Slider(1, 5, 1,
                                       value=None,
                                       id='usage-rating-slider'
                                       )
                        ]),

                    # html.Div(
                    #     children=[
                    #         html.Label("Rate how much you liked the finale graphs:", style={'display': 'inline-block', 'margin-right': '20px'}),
                    #         dcc.Checklist(
                    #             options=[
                    #                 {'label': '1', 'value': '1'},
                    #                 {'label': '2', 'value': '2'},
                    #                 {'label': '3', 'value': '3'},
                    #                 {'label': '4', 'value': '4'},
                    #                 {'label': '5', 'value': '5'}
                    #             ],
                    #             id='graph-rating-checkbox',
                    #             inline=True,
                    #             style={
                    #                 'display': 'inline-block',
                    #                 'vertical-align': 'middle',
                    #                 'margin-right': '0 10px',
                    #                 #'margin-bottom': '20px'
                    #                 #'transform': 'scale(1.2)'
                    #             }
                    #         ),
                    #     ],
                    #     style={'margin-bottom': '20px', 'text-align': 'center'}
                    # ),
                    html.Div(
                        children=[
                            html.Label("Comments :", style={'margin-bottom': '10px'}),
                            html.P("What specifically did you like/dislike. Do you have an example from your life "
                                   "where this theory would apply?"),
                            html.Div(
                                children=[
                                    dcc.Textarea(
                                        placeholder="Enter your comments...",
                                        id='comments-textarea',
                                        style={'width': '80%', 'height': '100px'}
                                    )
                                ],
                                style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}
                            )
                        ],
                        style={'margin-bottom': '20px', 'text-align': 'center'}
                    ),
                    html.Div(
                        style={'display': 'flex', 'justify-content': 'center'},
                        children=[
                            html.Button("Submit", id='submit-button', n_clicks=0,
                                        style={'display': 'block', 'margin': 'center'}),
                        ]
                    ),
                    html.Div(id='output-div',
                             children=[
                                 html.P("The data are saved.")])
                ])
        ],
        style={'margin-left': '1cm'}
    ),
]
)


@callback(
    [Output('output-div', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('store-data', 'data'),
     State('age-checkbox', 'value'),
     State('gender-radio', 'value'),
     State('email', 'value'),
     State('final_marks-slider', 'value'),
     State('comments-textarea', 'value'),
     State('app-rating-slider', 'value'),
     State('theory-rating-slider', 'value'),
     State('usage-rating-slider', 'value')
     ]
)
def store_personal_info(n_clicks, data1, age, gender, email, likability, comments, app_rate, theory_rate,
                        usage_rate):
    if n_clicks > 0 and n_clicks is not None:
        today = date.today()
        data2 = {
            'data': data1,
            'age': age,
            'gender': gender,
            'email': email,
            'likability': likability,
            'comments': comments,
            'app_rate': app_rate,
            'theory_rate': theory_rate,
            'usage_rate': usage_rate,
            'today_date': today
        }
        # save_data_to_database(data2)
        print(data1)
        return 'Data saved to database'
    else:
        raise PreventUpdate
    # try:
    #     # Assuming you have a table named 'data' with appropriate columns in your database
    #     cursor.execute("INSERT INTO data (column1, column2, ...) VALUES (?, ?, ...)",
    #                    (data['key1'], data['key2'], ...))
    #     conn.commit()
    #     return 'Data saved successfully to the database.'
    # except Exception as e:
    #     return f'Error saving data to the database: {str(e)}'

    # return data
    # else:
    #     raise PreventUpdate


# @callback(
#     dash.dependencies.Output('submit-button', 'style'),
#     dash.dependencies.Input('submit-button', 'n_clicks')
# )
# def handle_submission(n_clicks):
#     if n_clicks > 0:
#         # Perform actions with the submitted data
#         # For example, save the data to a database or perform calculations
#         #print("Form submitted!")
#         # Reset the button to prevent multiple submissions
#         return {'display': 'none'}
#     else:
#         return {'display': 'block'}


@callback(
    [Output(component_id='bar_hist_all_states', component_property='figure'),
     Output(component_id='bar_hist_all_actions', component_property='figure')
     ],
    [
        Input('update-button', 'n_clicks'),
        Input('store-data', 'data')
    ]
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
        fig_states = px.bar(df_s, x='States', y='Number of states')
        fig_actions = px.bar(df_a, x='Actions', y='Number of actions')

        fig_states.update_layout(transition_duration=500)
        fig_actions.update_layout(transition_duration=500)

        # bar_plot1 = dcc.Graph(figure=bar_hist_states)
        # bar_plot2 = dcc.Graph(figure=bar_hist_actions)

        return [fig_states, fig_actions]

    else:
        raise PreventUpdate

