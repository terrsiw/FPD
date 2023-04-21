from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    dbc.Alert("Hello Bootstrap!", color="success"),
    className="p-5",
)

if __name__ == "__main__":
    app.run_server()

# from src.components.layout import create_layout
#
#
# def main() -> None:
#     app = Dash(external_stylesheets=[BOOTSTRAP])
#     app.title = "Medal dashboard"
#     app.layout = create_layout(app)
#     app.run()

#
# if __name__ == "__main__":
#     main()
