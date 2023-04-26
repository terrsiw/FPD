import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# app = dash.Dash(__name__, use_pages=True)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    use_pages=True,
    external_stylesheets=[dbc.themes.LUX]
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

        # content of each page
        dash.page_container
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
    # from dash import Dash, html, dcc, callback, Output, Input
    # import plotly.express as px
    # import pandas as pd
    #
    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
    #
    # app = Dash(__name__)
    #
    # app.layout = html.Div([
    #     html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    #     dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    #     dcc.Graph(id='graph-content')
    # ])
    #
    #
    # @callback(
    #     Output('graph-content', 'figure'),
    #     Input('dropdown-selection', 'value')
    # )
    # def update_graph(value):
    #     dff = df[df.country == value]
    #     return px.line(dff, x='year', y='pop')
    #
    #
    # if __name__ == '__main__':
    #     app.run_server(debug=True)
