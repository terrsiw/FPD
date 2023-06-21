import dash
from dash import dcc, html
import plotly.express as px
import os

dash.register_page(__name__, path='/')

df = px.data.gapminder()
# current_directory = os.path.dirname(os.path.abspath(__file__))
# subdirectory = os.path.join(current_directory, 'subdirectory')
# image_path = os.path.join(subdirectory, 'fig.jpg')

layout = html.Div([
    html.Div(
        children=[
            html.H3('Description of the application'),
            html.Div(
                dcc.Link('I am ready to rate the graphs', href='/page2'),
                style={'text-align': 'right', 'margin-right': '20px'}
            ),
            html.Div([
                html.P('Hi everyone,'),
                html.P('my name is Tereza Sivakova and this web application is a part of my doctoral thesis.'),
                html.P('I am a PhD student at CTU FNSPE and my field is dynamic decision-making (DM), especially '
                       'qualification of user \'s preferences in DM.'),
                html.P('Because the average user does not understand the theory of DM, '
                       'we try to get some prior estimate of the user preference and then find out more information durion the DM. '
                       'I try to quantify user preferences especially when they have contradictory preferences.'
                       'Actually, I try to estimate to which of these contradictory preferences he will lean towards.'),
                html.P('I tried to use the theory on few school examples and now I would like to try it on you.')]),

            html.H5('What you will do:'),
            html.Div([
                html.P("This application tries to map the user's preferences. We would like to see, how "
                       "different people think of the same problem."),
                html.P("Imagine it is winter and you want to have 22 degrees in your living-room, but on the other "
                       "hand you want to spend as less money as possible."
                       "Now it's up to you which of these preferences you lean towards."),
                # html.Img(src='./fig.jpg'),
                html.P("We will show you results of states (degrees) and actions (heating/coaling) in a room in every "
                       "20 times steps and we want you to rate the"
                       "results as you liked them as at school with marks 1 to 5."),
                html.P("You should like the state 7 (which represents the 22 degrees) "
                       "and action 4 (which means that you will not heat or turn on air conditioner)."),
                html.H5('If you do not want to rate all 25 results'),
                html.P('There is the possibility that you will like the results before the 25th result. '
                       'Then, you can click on the button "I like the results", and continue on the next page with '
                       'final questionare. '),
                html.P('Or it can happen that you will be bored with rating, then you can click on "I am bored", '
                       'and also continue on the final page.'),
                html.H5('Final questioner:'),
                html.P(
                    "On the final page, you will find the final results and a short questioner."
                    "Please, fill the questionnaire. We want to then make some statistical conclusion based on "
                    "your responses. That is why we would like to know your age, gender ect."
                    "And, please, try to rate to the best of your knowledge and conscience. "),
                html.P(
                    "If you would have any questions, please, do not hesitate to contact me via email."),
                html.Br(),
                html.Br(),
                html.Br()
            ]),

            # html.Div(
            #     style={'position': 'fixed', 'bottom': '10px', 'right': '10px'},
            #     children=[
            #         html.Div('© 2023 Tereza Sivakova'),
            #         html.Div('Email: sivakter@cvut.cz')
            #     ]
            # ),
            # html.Div(
            #     id="acknowledgment",
            #     children=[
            #         html.P("Special thanks to Marko Ruman for helping with uploading the app on the server."),
            #     ],
            #     style={
            #         "position": "fixed",
            #         "bottom": "10px",
            #         "right": "10px",
            #         "text-align": "left",
            #         "font-size": "12px",
            #         "color": "gray",
            #     },
            # ),
        ],
        style={'margin-left': '1cm', "margin-bottom": "2 cm"}
    ),
]
)
