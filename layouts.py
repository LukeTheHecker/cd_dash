import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
from plotly.tools import mpl_to_plotly
import dash_core_components as dcc

import numpy as np
from datetime import datetime as dt
from datetime import date, timedelta
import pandas as pd
from app import app
import mne
import os
import pickle as pkl
import time

from components import Header
from components.functions import simulate_source, predict_source, make_fig_objects, load_model, inverse_solution

print('Loading Some Variables')
# Load Global variables
pth_modeling = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'assets\\modeling'))
# Load Triangles through some inverse operator created previously
fname_inv = pth_modeling + '\\inverse-inv.fif'
tris = mne.minimum_norm.read_inverse_operator(fname_inv)['src'][0]['use_tris']
# Load Forward Model for other inverse solutions:
fwd = mne.read_forward_solution(pth_modeling + '\\fsaverage-fwd.fif')
# Load generic Epochs structure for other inverse solutions:
epochs = mne.read_epochs(pth_modeling + '\\epochs-epo.fif')
## Leadfield
with open(pth_modeling +'\\leadfield.pkl', 'rb') as f:
    leadfield = pkl.load(f)[0]
## Positions
with open(pth_modeling +'\\pos.pkl', 'rb') as f:
    pos = pkl.load(f)[0]

print('Preloading Models')
model_paper = load_model(pth_modeling + '\\model_paper\\')
model_flex = load_model(pth_modeling + '\\model_flex\\')
model_lowsnr = load_model(pth_modeling + '\\model_lowsnr\\')



######################## START ConvDip Layout ########################
layout_convdip_page =  html.Div([
    html.Div([
        # CC Header
        Header(),
        # Header Bar
        html.Div([
          html.Center(html.H1(["ConvDip"], className="gs-header gs-text-header padded",style={'marginTop': 15}))
          ]),
        # Hidden divs: 
        # stores the simulated source vector y
        html.Div(id='current_y', style={'display': 'none'}),
        # stores the selected SNR in dB
        html.Div(id='current_snr', style={'display': 'none'}),

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

        dbc.Row(
            # Simulation Panel
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H4(["Advanced Options"], style={'marginTop':15}),
                        html.Br(),
                        html.Label('Signal to Noise Ratio (in dB):'),
                        dcc.Markdown('''*e.g. single value: 6 or range of values: 6, 9 (comma separated)*'''),
                        dbc.Input(id='noise_level_input', value=6),
                        html.Br(),
                        ]),
                    html.Div([
                        html.Br(),
                        html.Label('Number of sources:'),
                        dcc.Markdown('''*e.g. single value: 3 or range of values: 1, 5 (comma separated)*'''),
                        dbc.Input(id='number_of_sources_input', value=3),
                        html.Br(),
                    ]),
                html.Div([
                    html.Br(),
                    html.Label('Size of sources (diameter of sphere in mm):'),
                    dcc.Markdown('''*e.g. single value: 35 or range of values: 25, 35 (comma separated)*'''),
                    dbc.Input(id='size_of_source_input', value=35),
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
    ),  # end of simulation row

        # Prediction Group
        dbc.CardGroup([
            dbc.Card(
                dbc.CardBody([
                    dbc.Select(
                        id='model_selection',
                        options=[
                            {'label': 'ConvDip Flexible: Trained on a wide range of SNR from 0 to 8 dB with 1 to 5 sources each between 25 and 35 mm spheric diameter.', 'value': 'flex'},
                            {'label': 'ConvDip Paper Version: Trained on a narrow range of SNR from 6 to 9 dB with 1 to 5 sources each between 25 and 35 mm spheric diameter.', 'value': 'paper'},
                            {'label': 'ConvDip for low SNR: Trained on a narrow range of SNR from 3 to 6 dB with 1 to 5 sources each between 25 and 35 mm spheric diameter.', 'value': 'lowsnr'},
                            {'label': 'eLORETA', 'value': 'eLORETA'},
                            {'label': 'LCMV Beamforming', 'value': 'lcmv'},
                            {'label': 'Minimum Norm Estimate', 'value': 'MNE'},
                            {'label': 'dSPM', 'value': 'dSPM'},
                        ],
                        value='flex'
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
        Output('sim_source_plot', 'figure'),
        Output('current_y', 'children'),
        Output('current_snr', 'children')], 
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
    start = time.time()
    y, x_img, db_choice = simulate_source(settings[1], settings[2], settings[3], 1, leadfield, pos)
    end_1 = time.time()
    fig_y, fig_x = make_fig_objects(y, x_img, tris, pos)
    end_2 = time.time()
    print(f'Simulation: {end_1-start}, simulation+plotting: {end_2-start}')
    # fig_y, fig_x = simulate_source(settings[1], settings[2], settings[3])

    spinner_output = 'Simulation is Ready'
    return spinner_output, fig_x, fig_y, y, db_choice

# Callback for the Predict button
@app.callback(
        [Output('loading-output-prediction', 'children'),
        Output('pred_scalp_plot', 'figure'),
        Output('pred_source_plot', 'figure')], 
        [Input('predict_button', 'n_clicks')],
        [State('sim_scalp_plot', 'figure'),
        State('model_selection', 'value'),
        State('current_y', 'children'),
        State('current_snr', 'children')]
        ) 

def predict_sample(*params):
    inputs = [i for i in params]
    # Check if sample was simulated already
    if inputs[1]['data'] == []:
        print('No sample simulated or at least its not plotted')
        spinner_output = 'No simulation available.'
        return spinner_output, None, None
    
    # Check if hidden html.Div has source value
    try:
        y = np.asarray(inputs[3])
        db = np.asarray(inputs[4])[0]
        print(f'y.shape={y.shape}\ndb.shape={db.shape}')
        print(f'db={db}')
    except:
        spinner_output = 'No simulation available.'
        return spinner_output, None, None

    data = inputs[1]['data'][0]['z']
    data = np.asarray(data)

    if inputs[2] == 'paper':
        model = model_paper
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'lowsnr':
        model = model_lowsnr
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'flex':
        model = model_flex
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'eLORETA' or inputs[2] == 'lcmv' or inputs[2] == 'MNE' or inputs[2] == 'dSPM' or inputs[2] == 'mxne':
        x = np.sum(y * leadfield, axis=1)
        y, x_img = inverse_solution(x, db, epochs, fwd, leadfield, inputs[2])

    

    fig_y, fig_x = make_fig_objects(y, x_img, tris, pos)

    spinner_output = 'Prediction ready!'
    return spinner_output, fig_x, fig_y
