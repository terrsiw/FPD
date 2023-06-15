import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# app = dash.Dash(__name__, use_pages=True)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    use_pages=True,
    external_stylesheets=[dbc.themes.SPACELAB]
)

app.title = "Dynamic decision-making "

app.layout = html.Div(
    style={"height": "100%"},
    children=[
        # main app framework
        html.Div(
            [html.H2(
                "Decision-making app",
                id="title",
                className="eight columns",
                style={"margin-left": "3%"},
            )
            ],
            className="banner row",
        ),
        html.Div([
            dcc.Link(page['name'] + "  |  ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),
        html.Div(
            style={'position': 'fixed', 'bottom': '10px', 'right': '10px'},
            children=[
                html.Div('© 2023 Tereza Sivakova'),
                html.Div('Email: sivakter@cvut.cz')
            ]
        ),
        html.Div(
            id="acknowledgment",
            children=[
                html.P("Special thanks to Marko Ruman for helping with uploading the app on the server."),
            ],
            style={
                "position": "fixed",
                "bottom": "10px",
                "left": "10px",
                "text-align": "left",
                "font-size": "12px",
                "color": "gray",
            },
        ),

        # content of each page
        dash.page_container
    ]
)

if __name__ == "__main__":
    app.run(debug=True)

#import dash
# from dash import html, dcc
# import plotly.express as px
# import pandas as pd
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# # from pages import page1, page2, page3
# from Second import *
# from dash.exceptions import PreventUpdate
#
# app = dash.Dash(
#     __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
#     use_pages=True,
#     external_stylesheets=[dbc.themes.SPACELAB]
# )
#
# app.title = "Dynamic decision-making "
#
# app.layout = html.Div(
#     style={"height": "100%"},
#     children=[
#         # main app framework
#         html.Div(
#             [html.H2(
#                 "Decision-making app",
#                 id="title",
#                 className="eight columns",
#                 style={"margin-left": "3%"},
#             )
#             ],
#             className="banner row",
#         ),
#         # html.Div([
#         #     dcc.Link(page['name'] + "  |  ", href=page['path'])
#         #     for page in dash.page_registry.values()
#         # ]),
#         html.Hr(),
#         dcc.Location(id='url', refresh=False),
#         html.Div(id='page-content'),
#         dcc.Store(id='store-data', storage_type='memory'),
#         # content of each page
#         dash.page_container
#     ]
# )
#
# # @app.callback(
# #     Output('page-content', 'children'),
# #     [Input('url', 'pathname')]
# # )
# # def render_page(pathname):
# #     if pathname == '/':
# #         return page1.layout
# #     elif pathname == '/page2':
# #         return page2.layout
# #     elif pathname == '/page3':
# #         return page3.layout
# #     else:
# #         return "404 - Page not found"
#
#
# # @app.callback(
# #     Output('store-data', 'data'),
# #     [Input('submit-val', 'n_clicks'),
# #      Input('store-data', 'data'),
# #      Input('my_slider', 'value'),
# #      Input('mark_button', 'n_clicks')  # Input('url', 'pathname')
# #      ]
# # )
# # def store_data(n_clicks, data, value, mark_clicks):
# #     ctx = dash.callback_context
# #     triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]
# #     if n_clicks is not None and n_clicks != 0 and not data and value is None:
# #         var_init = initialization(500, "FALSE", 20)
# #         system = var_init[0]
# #         agent = var_init[1]
# #         data2 = var_init[2]
# #         user = var_init[3]
# #         data1 = var_init[4]
# #
# #         if data1.t <= data2.length_sim / data1.length_sim:
# #             user.calculate_alfa()
# #             s1 = data1.states[data1.t]
# #             a = dnoise(user.r[:, s1])
# #             data1.actions.append(a)
# #             data1, data2, agent = system.small_loop(agent, data2, data1)
# #
# #         w = agent.w
# #         # s0 = data2.states[data2.t]
# #         nu = agent.nu
# #         gam = agent.gam
# #         model = agent.model
# #         mi = agent.mi
# #         ri = agent.ri
# #         r = agent.r
# #         V = agent.V
# #         data_states = data2.states
# #         data_actions = data2.actions
# #         data_t = data2.t
# #         data1_states = data1.states
# #         data1_actions = data1.actions
# #         data1_marks = data1.marks
# #         data1_t = data1.t
# #         user_gam = user.gam
# #         user_model = user.model
# #         user_mi = user.mi
# #         user_ri = user.ri
# #         user_r = user.r
# #         user_V = user.V
# #
# #         obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
# #                           data1_actions,
# #                           data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V]
# #
# #         return obj_store_data
# #
# #     elif n_clicks is not None and n_clicks != 0 and data and value is not None:
# #         if triggered_component == 'my_slider' or (triggered_component == 'mark_button' and mark_clicks > 0):
# #             var_init = initialization(500, "FALSE", 20)
# #             system = var_init[0]
# #             agent = var_init[1]
# #             data2 = var_init[2]
# #             user = var_init[3]
# #             data1 = var_init[4]
# #
# #             agent.w = data[0]
# #             agent.nu = data[1]
# #             agent.gam = np.array(data[2])
# #             agent.model = np.array(data[3])
# #             agent.mi = np.array(data[4])
# #             agent.ri = np.array(data[5])
# #             agent.r = np.array(data[6])
# #             agent.V = np.array(data[7])
# #             data2.states = data[8]
# #             data2.actions = data[9]
# #             data2.t = data[10]
# #             data1.states = data[11]
# #             data1.actions = data[12]
# #             data1.marks = data[13]
# #             data1.t = data[14]
# #             user.gam = np.array(data[15])
# #             user.model = np.array(data[16])
# #             user.mi = np.array(data[17])
# #             user.ri = np.array(data[18])
# #             user.r = np.array(data[19])
# #             user.V = np.array(data[20])
# #
# #             m = value
# #             mm = 0
# #             if m == 0:
# #                 m = data1.marks[data1.t]
# #
# #             k = int(data1.marks[data1.t])
# #             if m - int(k) > 0:
# #                 mm = 2
# #             if m - int(k) < 0:
# #                 mm = 0
# #             if m - int(k) == 0:
# #                 mm = 1
# #
# #             data1.marks.append(m)
# #             data1.states.append(mm)
# #
# #             data1.t = data1.t + 1
# #             user.learn(data1)
# #
# #             if data1.t <= data2.length_sim / data1.length_sim:
# #                 user.calculate_alfa()
# #                 s1 = data1.states[data1.t]
# #                 a = dnoise(user.r[:, s1])
# #                 data1.actions.append(a)
# #                 data1, data2, agent = system.small_loop(agent, data2, data1)
# #
# #             w = agent.w
# #             # s0 = data2.states[data2.t]
# #             nu = agent.nu
# #             gam = agent.gam
# #             model = agent.model
# #             mi = agent.mi
# #             ri = agent.ri
# #             r = agent.r
# #             V = agent.V
# #             data_states = data2.states
# #             data_actions = data2.actions
# #             data_t = data2.t
# #             data1_states = data1.states
# #             data1_actions = data1.actions
# #             data1_marks = data1.marks
# #             data1_t = data1.t
# #             user_gam = user.gam
# #             user_model = user.model
# #             user_mi = user.mi
# #             user_ri = user.ri
# #             user_r = user.r
# #             user_V = user.V
# #
# #             obj_store_data = [w, nu, gam, model, mi, ri, r, V, data_states, data_actions, data_t, data1_states,
# #                               data1_actions,
# #                               data1_marks, data1_t, user_gam, user_model, user_mi, user_ri, user_r, user_V]
# #             return obj_store_data
# #     else:
# #         raise PreventUpdate
# #     return {}
#
#
# # Callback to retrieve stored data in Page 3
# # @app.callback(
# #     [
# #         Output('plots-container', 'children')
# #         # Output(component_id='bar_hist_all_states', component_property='figure'),
# #         # Output(component_id='bar_hist_all_actions', component_property='figure'),
# #     ],
# #     # Input('url', 'pathname'),
# #     [Input("update-button", "n_clicks")],
# #     State('store-data', 'data')
# # )
# # def display_graph(n_clicks, data):
# #     # if pathname == '/page3':
# #     if n_clicks is not None:
# #         if data is not None:
# #             # data = json.loads(data)
# #
# #             var_init = initialization(500, "FALSE", 20)
# #
# #             agent = var_init[1]
# #             data2 = var_init[2]
# #             data1 = var_init[4]
# #
# #             data2.states = data[8]
# #             data2.actions = data[9]
# #             data2.t = data[10]
# #             data1.states = data[11]
# #             data1.actions = data[12]
# #             data1.marks = data[13]
# #             data1.t = data[14]
# #
# #             data_states = data2.states
# #             data_actions = data2.actions
# #
# #             number_of_s = np.zeros(agent.ss)
# #             number_of_a = np.zeros(agent.aa)
# #
# #             data_states = np.array(data_states)
# #             data_actions = np.array(data_actions)
# #
# #             for j in range(agent.ss):
# #                 number_of_s[j] = np.sum(data_states[:] == j)
# #
# #             for k in range(agent.aa):
# #                 number_of_a[k] = np.sum(data_actions[:] == k)
# #
# #             data_state = {'States': list(np.arange(0, agent.ss)),
# #                           'Number of states': number_of_s}
# #
# #             data_action = {'Actions': list(np.arange(0, agent.aa)),
# #                            'Number of actions': number_of_a}
# #
# #             df_s = pd.DataFrame(data_state)
# #             # print(df_s)
# #             df_a = pd.DataFrame(data_action)
# #             # df_s = load_object("data_states")
# #             bar_hist_states = px.bar(df_s, x='States', y='Number of states')
# #             bar_hist_actions = px.bar(df_a, x='Actions', y='Number of actions')
# #
# #             bar_hist_states.update_layout(transition_duration=500)
# #             bar_hist_actions.update_layout(transition_duration=500)
# #
# #             bar_plot1 = dcc.Graph(figure=bar_hist_states)
# #             bar_plot2 = dcc.Graph(figure=bar_hist_actions)
# #
# #             return html.Div(html.Div(bar_plot1,
# #                                      style={'width': '50%', 'display': 'inline-block'}),
# #                             html.Div(bar_plot2,
# #                                      style={'width': '50%', 'display': 'inline-block'}))
# #         else:
# #             return {}
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
# # import dash
# # from dash import html, dcc
# # import dash_bootstrap_components as dbc
# # from dash.dependencies import Input, Output, State
# #
# #
# # app = dash.Dash(
# #     __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
# #     use_pages=True,
# #     external_stylesheets=[dbc.themes.SPACELAB]
# # )
# #
# #
# # app.title = "Dynamic decision-making "
# #
# # app.layout = html.Div(
# #     style={"height": "100%"},
# #     children=[
# #         # main app framework
# #         html.Div(
# #             [html.H2(
# #                 "Decision-making app",
# #                 id="title",
# #                 className="eight columns",
# #                 style={"margin-left": "3%"},
# #             )
# #             ],
# #             className="banner row",
# #         ),
# #         # dcc.Store(id='store-data', data=[], storage_type='memory'),
# #         html.Div([
# #             dcc.Link(page['name'] + "  |  ", href=page['path'])
# #             for page in dash.page_registry.values()
# #         ]),
# #         # html.Div(id='page-content'),
# #         html.Hr(),
# #         # dcc.Location(id='url', refresh=False),
# #         # dcc.Store(id='store-data', data=[], storage_type='memory'),
# #         # html.Div(id='page-content'),
# #         # content of each page
# #         dash.page_container
# #     ]
# # )
# #
# # if __name__ == "__main__":
# #     app.run(debug=True)
#
# # @app.callback(
# #     Output('page-content', 'children'),
# #     [Input('url', 'pathname')]
# #     [State('store-data', 'data')]
# # )
# # def render_page(pathname):
# #     if pathname == '/':
# #         # Display page 1 layout
# #         return page1_layout
# #     elif pathname == '/page2':
# #         # Display page 2 layout
# #         return page2_layout
# #     elif pathname == '/page2':
# #         # Display page 2 layout
# #         return page2_layout
# #     else:
# #         # Page not found
# #         return '404 - Page not found'
#
# # Define the callback to switch between pages
# # @app.callback(
# #     Output('page-content', 'children'),
# #     [Input('url', 'pathname')],
# #     [State('store-data', 'data')]
# # )
# # def render_page(pathname, stored_data):
# #     for page_layout in pages_registry.values():
# #         if page_layout.path == pathname:
# #             # Display the page layout
# #             return page_layout.layout(data)
# #
# #     # Page not found
# #     return '404 - Page not found'
#
# # Define the callback to switch between pages and update the stored data
# # @app.callback(
# #     Output('page-content', 'children'),
# #     [Input('url', 'pathname')],
# #     [State('store-data', 'data')]
# # )
# # def render_page(pathname, data):
# #     if pathname == '/':
# #         # Display the default page
# #         return page1.page1_layout(data)
# #     else:
# #         page_file = os.path.join('pages', pathname[1:] + '.py')
# #         if os.path.isfile(page_file):
# #             # Load the module dynamically
# #             spec = importlib.util.spec_from_file_location(pathname[1:], page_file)
# #             module = importlib.util.module_from_spec(spec)
# #             spec.loader.exec_module(module)
# #
# #             # Check if the page_layout variable exists in the module
# #             if hasattr(module, 'page_layout'):
# #                 return module.page_layout(data)
# #
# #     return '404 - Page not found'
#
#
# # import dash
# # from dash import html, dcc
# # import dash_bootstrap_components as dbc
# # from dash.dependencies import Input, Output, State
# #
# # import pages.page2
# # import pages.page3
# #
# # # app = dash.Dash(__name__, use_pages=True)
# #
# # app = dash.Dash(
# #     __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
# #     use_pages=True,
# #     external_stylesheets=[dbc.themes.SPACELAB]
# # )
# #
# # app.title = "Dynamic decision-making "
# #
# # app.layout = html.Div(
# #     style={"height": "100%"},
# #     children=[
# #         # main app framework
# #         html.Div(
# #             [html.H2(
# #                 "Decision-making app",
# #                 id="title",
# #                 className="eight columns",
# #                 style={"margin-left": "3%"},
# #             )
# #             ],
# #             className="banner row",
# #         ),
# #         # dcc.Store(id='store-data', data=[], storage_type='memory'),
# #         # html.Div([
# #         #     dcc.Link(page['name'] + "  |  ", href=page['path'])
# #         #     for page in dash.page_registry.values()
# #         # ]),
# #         # html.Div(id='page-content'),
# #         html.Hr(),
# #         dcc.Location(id='url', refresh=False),
# #         # content of each page
# #         dash.page_container
# #     ]
# # )
# #
# # page_registry = {
# #     '/page2': pages.page2.page2_layout,
# #     '/page3': pages.page3.page3_layout
# # }
# #
# #
# # # Define the callback to switch between pages and update the stored data
# # @app.callback(
# #     Output('page-content', 'children'),
# #     [Input('url', 'pathname')],
# #     [State('store-data', 'data')]
# # )
# # def render_page(pathname, stored_data):
# #     page_layout = page_registry.get(pathname)
# #     if page_layout:
# #         if pathname == '/page3':
# #             return html.Div([
# #                 page_layout,
# #                 dcc.Store(id='store-data', data=stored_data)  # Reinitialize the store on page 3 with stored data
# #             ])
# #         return page_layout
# #     else:
# #         return '404 - Page not found'
# #
# #
# # if __name__ == "__main__":
# #     app.run(debug=True)
#
# # import dash
# # import dash_core_components as dcc
# # import dash_html_components as html
# # from dash.dependencies import Input, Output
# #
# # app = dash.Dash(__name__, use_pages=True)
# #
# # layouts = [page.layout for page in dash.page_registry.values()]
# # app.title = "Dynamic decision-making "
# #
# # app.layout = html.Div([
# #     dcc.Location(id='url', refresh=False),
# #     html.Div(id='page-content')
# # ])
# #
# #
# # @app.callback(
# #     Output('page-content', 'children'),
# #               [Input('url', 'pathname')]
# # )
# # def display_page(pathname):
# #     if pathname in dash.page_registry:
# #         # Retrieve the layout for the current page
# #         return dash.page_registry[pathname].layout
# #     else:
# #         # Default to the first layout if the page is not found
# #         return layouts[0]
# #
# #
# # if __name__ == '__main__':
# #     app.run_server(debug=True)
