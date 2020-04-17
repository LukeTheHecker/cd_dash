import dash
import dash_bootstrap_components as dbc

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO], url_base_pathname='/')#
server = app.server
app.config.suppress_callback_exceptions = True
# import dash_auth
# # Keep this out of source code repository - save in a file or a database
# VALID_USERNAME_PASSWORD_PAIRS = [
#     ['[username]', '[password']
# ]
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )