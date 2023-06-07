import dash
from dash import dcc, html
import plotly.express as px

dash.register_page(__name__, path='/')

df = px.data.gapminder()

layout = html.Div([
    html.H1('Description of the application'),
    html.Div(
        dcc.Link('I understand and want to continue on page 2', href='/page2'),
        style={'text-align': 'right', 'margin-right': '20px'}
    ),
    html.Div([
        html.P('Hi everyone,'),
        html.P("this application tries to map the user's preferences. ..."),
        html.P("Please, give us a feedback to our results."),
    ]),

]
)

