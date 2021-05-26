import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd

f = pd.read_csv('data.csv')

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Checklist(
        id="checklist",
        options=[{"label": x, "value": x}
                 for x in f.columns],
        value=[f.columns[1]],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="line-chart"),
])


@app.callback(
    Output("line-chart", "figure"),
    [Input("checklist", "value")])
def update_line_chart(variables):
    if len(variables) == 0:
        return px.line(f, x="frame", y=f.columns[1])
    return px.line(f, x="frame", y=variables)


if __name__ == "__main__":
    app.run_server(debug=True)
