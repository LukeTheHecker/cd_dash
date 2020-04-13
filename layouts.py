import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
from components import Header
from components.functions import simulate_source, predict_source, make_fig_objects
from datetime import datetime as dt
from datetime import date, timedelta
import pandas as pd
from app import app

import numpy as np
from plotly.tools import mpl_to_plotly
import dash_core_components as dcc
import matplotlib.pyplot as plt

from dash_obj_in_3dmesh import geometry_tools, wav_obj

def example_plot():
    x = np.random.randn(100)
    fig = plt.figure()
    plt.hist(x)
    plotly_fig = mpl_to_plotly(fig)
    # graph = dcc.Graph(id='myGraph', fig=plotly_fig)
    return plotly_fig



######################## START ConvDip Layout ########################
layout_convdip_page =  html.Div([
    html.Div([
        # CC Header
        Header(),
        # Header Bar
        html.Div([
          html.Center(html.H1(["ConvDip"], className="gs-header gs-text-header padded",style={'marginTop': 15}))
          ]),
        # Abstract
        dbc.Card(
            dbc.CardBody([
                html.P('EEG and MEG are well-established non-invasive methods in neuroscientific research and clinical diagnostics. Both methods provide a high temporal but low spatial resolution of brain activity. In order to gain insight about the spatial dynamics of the EEG one has to solve the inverse problem, which means that more than one configuration of neural sources can evoke one and the same distribution of EEG activity on the scalp. A large number of approaches have been developed in the past to handle the inverse problem by creating more accurate and reliable solutions. Artificial neural networks have been previously used successfully to find either one or two dipoles sources. These approaches, however, have never solved the inverse problem in a distributed dipole model with more than two dipole sources. We present ConvDip, a novel convolutional neural network (CNN) architecture that solves the EEG inverse problem in a distributed dipole model based on simulated EEG data in a semi-supervised approach. We show that (1) ConvDip learned to produce inverse solutions from a single time point of EEG data and (2) outperforms state-of-the-art methods (eLORETA and LCMV beamforming) on all focused performance measures. (3) It is more flexible when dealing with varying number of sources, produces less ghost sources and misses less real sources than the comparison methods. (4) It produces plausible inverse solutions for real-world EEG recordings and needs less than 40 ms for a single forward pass. Our results qualify ConvDip as an efficient and easy-to-apply novel method for source localization in EEG an MEG data, with high relevance for clinical applications, e.g. in epileptology and real time applications.', className="card-text")
            ]
            ), style={'display': 'inline-block', 'width': 700, 'margin': '10px'}
        )
        ,
        # Image
        html.Div([html.Img(src='/assets/architecture.png', height=300)], style={'display': 'inline-block', 'margin': '10px'}),

        dbc.CardGroup([
            # Simulation Panel
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6(["Advanced Options"], style={'marginTop':15}),
                        html.Br(),
                        html.Label('Signal to Noise Ratio (in dB):'),
                        dcc.Input(id='noise_level_input', value=6),
                        html.Br(),
                        ]),
                    html.Div([
                        html.Br(),
                        html.Label('Number of sources:'),
                        dcc.Input(id='number_of_sources_input', value=3),
                        html.Br(),
                    ]),
                html.Div([
                    html.Br(),
                    html.Label('Size of sources (mm):'),
                    dcc.Input(id='size_of_source_input', value=35),
                    html.Br(),
                    ]),
                html.Div([
                    html.Br(),
                    dbc.Button('Simulate Sample', id='sim_button', color="primary"),
                    dbc.Spinner(html.Div(id="loading-output-simulation")),
                    ]),
                html.Div([
                    html.Div(id='output_container_button',
                    children='Enter the values and press button')
                    ]),
                ])
            ], style={"width": "200px"}), # end of settings card

            # Simulation Figures
            dbc.Card(
                dbc.CardBody([
                html.Br(),
                dcc.Graph(
                    id='sim_scalp_plot',
                    config={
                        'displayModeBar': False
                        },
                    figure={
                        'data': [],
                        }
                    ),
                    ])
                ),
            # Simulation Canvas 2
            dbc.Card(
                dbc.CardBody([
                    html.Br(),
                    dcc.Graph(
                        id="sim_source_plot",
                        figure={
                            "data": [],
                            },
                        ),
                    ])
                )
    ]),  # end of simulation group

        # Prediction Group
        dbc.CardGroup([
            dbc.Card(
                dbc.CardBody([
                    dbc.RadioItems(
                        id='model_selection',
                        options=[
                            {'label': 'ConvDip Paper Version', 'value': 'paper'},
                            {'label': 'ConvDip for low SNR', 'value': 'lowsnr'},
                        ],
                        value='paper'
                        ),
                    html.Br(),
                    dbc.Button('Predict Source', id='predict_button', color="primary"),
                    dbc.Spinner(html.Div(id="loading-output-prediction")),
                    ])
                ),
        # Prediction Figures
        dbc.Card(
            dbc.CardBody([
                html.Br(),
                dcc.Graph(
                    id='pred_scalp_plot',
                    config={
                        'displayModeBar': False
                        },
                    figure={
                        'data': [],
                        }
                    )
                ])
            ),


        # Simulation Canvas 2
        dbc.Card(
            dbc.CardBody([
                html.Br(),
                dcc.Graph(
                    id="pred_source_plot",
                    figure={
                        "data": [],
                        },
                    )
                ]),
            ),
        ]) # End of Prediction Group
        ], className="subpage"), 
    ], className="page")

######################## START ConvDip Callbacks ########################

# Callback for the Simulate-button
@app.callback(
        [Output('loading-output-simulation', 'children'),
        Output('sim_scalp_plot', 'figure'),
        Output('sim_source_plot', 'figure')], 
        [Input('sim_button', 'n_clicks')],
        [State('noise_level_input', 'value'), 
        State('number_of_sources_input', 'value'),
        State('size_of_source_input', 'value'),
        State('sim_source_plot', 'figure')
        ])

def simulate_sample(*params):
    print('simulating')
    settings = [i for i in params]
    if np.any(settings==None):
        return
    print(settings[3])
    print(settings[3])
    print(settings[3])
    print(settings[3])
    y, x_img = simulate_source(settings[1], settings[2], settings[3], 1)
    fig_y, fig_x = make_fig_objects(y, x_img)
    # fig_y, fig_x = simulate_source(settings[1], settings[2], settings[3])

    spinner_output = 'Simulation is Ready'
    return spinner_output, fig_x, fig_y

# Callback for the Predict button
@app.callback(
        [Output('loading-output-prediction', 'children'),
        Output('pred_scalp_plot', 'figure'),
        Output('pred_source_plot', 'figure')], 
        [Input('predict_button', 'n_clicks')],
        [State('sim_scalp_plot', 'figure'),
        State('model_selection', 'value')]
        ) 

def predict_sample(*params):
    inputs = [i for i in params]
    if inputs[1]['data'] == []:
        print('No sample simulated or at least its not plotted')
        spinner_output = 'No simulation available.'
        return spinner_output, None, None
    else:
        if inputs[2] == 'paper':
            pth_model = '\\model_paper\\'
        elif inputs[2] == 'lowsnr':
            pth_model = '\\model_lowsnr\\'

        data = inputs[1]['data'][0]['z']
        data = np.asarray(data)
        y, x_img = predict_source(data, pth_model)
        fig_y, fig_x = make_fig_objects(y, x_img)
        spinner_output = 'Prediction ready!'
        return spinner_output, fig_x, fig_y
