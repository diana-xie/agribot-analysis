import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

# COSTS_ALL = pd.read_pickle('costs_all.pkl')
COSTS_ALL = pd.read_pickle('costs_all.pkl')


# get best costs of gain-max
def best_cost_cost(costs: pd.DataFrame):
    df_best_cost = pd.DataFrame(
        costs.groupby(
            ['iou_threshold']
        )['cost_score'].max()
    ).reset_index()  # best # cost of gain-max
    return df_best_cost


# get best costs of mAP-max
def best_cost_map(costs: pd.DataFrame):
    best_map = pd.DataFrame(
        costs.groupby(
            ['iou_threshold']
        )['m_ap'].max()).reset_index()  # best cost of mAP-max
    best_map = list(best_map.set_index(['iou_threshold', 'm_ap']).index)
    df = costs.set_index(['iou_threshold', 'm_ap']).loc[best_map]
    df_best_map_avg = pd.DataFrame(df.groupby(['iou_threshold'])['cost_score'].mean()).reset_index()
    return df_best_map_avg


def Model(gain_tp_weed: float = 0.5, cost_fp_weed: float = 0.3):
    """
    The model, which will be plotted in the dashboard
    :param gain_tp_weed:
    :param cost_fp_weed:
    :return:
    """

    costs = COSTS_ALL[(COSTS_ALL['gain_tp_weed'] == gain_tp_weed) & (COSTS_ALL['cost_fp_weed'] == cost_fp_weed)]

    # get best cost results for each cost-type x IoU threshold
    df_best_cost = best_cost_cost(costs=costs)  # get best results for gain-max condition
    df_best_map = best_cost_map(costs=costs)  # get best results for map-max condition

    iou_thresholds = list(df_best_cost['iou_threshold'])
    gains_costmax = list(df_best_cost['cost_score'])
    gains_mapmax = list(df_best_map['cost_score'])

    return iou_thresholds, gains_costmax, gains_mapmax


"""
############################################ the dash app layout ################################################
"""

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Object detector gains ($)"

# these are the controls where the parameters can be tuned.
# They are not placed on the screen here, we just define them.
# Each separate input (e.g. a slider for the fatality rate) is placed
# in its own "dbc.FormGroup" and gets a "dbc.Label" where we put its name.
# The sliders use the predefined "dcc.Slider"-class, the numeric inputs
# use "dbc.Input", etc., so we don't have to tweak anything ourselves.
# The controls are wrappen in a "dbc.Card" so they look nice.
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Gain of correctly identifying a weed as weed ($)'),
                html.Br(),
                dcc.Slider(
                    id='gain_tp_weed',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,
                    tooltip={'always_visible': True, "placement": "bottom"}
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Cost of falsely identifying a corn as weed ($)'),
                dcc.Slider(
                    id='cost_fp_weed',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.3,
                    tooltip={'always_visible': True, "placement": "bottom"}
                ),
            ]
        ),
        dbc.Button("Apply", id="submit-button-state",
                   color="primary", block=True)
    ],
    body=True,
)

# layout for the whole page
app.layout = dbc.Container(
    [
        # first, a jumbotron for the description and title
        dbc.Jumbotron(
            [
                dbc.Container(
                    [
                        html.H1("Object detector gains ($)", className="display-3"),
                        html.P(
                            "Interactively simulate different cost-gain scenarios. ",
                            className="lead",
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''

                            You can freely tune the cost of incorrect and correct identifications.
                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),
        # now onto the main page, i.e. the controls on the left
        # and the graphs on the right.
        dbc.Row(
            [
                # here we place the controls we just defined,
                # and tell them to use up the left 3/12ths of the page.
                dbc.Col(controls, md=3),
                # now we place the graphs on the page, taking up
                # the right 9/12ths.
                dbc.Col(
                    [
                        # the main graph that displays coronavirus over time.
                        dcc.Graph(id='main_graph'),
                        # the graph displaying the R values the user inputs over time.
                    ],
                    md=9
                ),
            ],
            align="top",
        ),
    ],
    # fluid is set to true so that the page reacts nicely to different sizes etc.
    fluid=True,
)


############################################ the dash app callbacks ################################################


@app.callback(
    [dash.dependencies.Output('main_graph', 'figure')
     ],

    [dash.dependencies.Input('submit-button-state', 'n_clicks')],

    [
     dash.dependencies.State('gain_tp_weed', 'value'),
     dash.dependencies.State('cost_fp_weed', 'value'),
     ]
)
def update_graph(_, gain_tp_weed, cost_fp_weed):

    iou_thresholds, gains_costmax, gains_mapmax = Model(gain_tp_weed=gain_tp_weed, cost_fp_weed=cost_fp_weed)

    return [{
               # return graph for compartments, graph for fatality rates, graph for reproduction rate
               'data': [
                   {'x': iou_thresholds, 'y': gains_costmax, 'type': 'line', 'name': 'gain-max'},
                   {'x': iou_thresholds, 'y': gains_mapmax, 'type': 'line', 'name': 'mAP-max'}
               ],
               'layout': {
                   'title': 'Gains across different IoU thresholds',
                   'xaxis': {
                       'title': 'IoU threshold'
                   },
                   'yaxis': {
                       'title': 'Gains ($)'
                   }
               }
           }]


if __name__ == '__main__':
    app.run_server(debug=True)
