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

        # content of each page
        dash.page_container
    ]
)

if __name__ == "__main__":
    app.run(debug=True)

