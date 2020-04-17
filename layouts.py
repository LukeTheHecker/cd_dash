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
model_gaussian = load_model(pth_modeling + '\\model_gaussian\\')



######################## START ConvDip Layout ########################
layout_convdip_page =  html.Div([
    html.Div([
        # CC Header
        Header(),
        # Header Bar
        html.Div([
          html.Center(html.H1(["ConvDip: A convolutional neural network for better M/EEG Source Imaging"], style={'marginTop': 15})),  #, className="gs-header gs-text-header padded",style={'marginTop': 15}));
          html.Br(),
          html.Center(html.H4(["Hecker, Lukas; Rupprecht, Rebekka; Tebartz van Elst, Ludger; Kornmeier, Juergen"], style={'marginTop': 15})),
          html.Center(html.H4(["2020"], style={'marginTop': 15}))
          ]),
        # Hidden divs: 
        # stores the simulated source vector y
        html.Div(id='current_y', style={'display': 'none'}),
        # stores the selected SNR in dB
        html.Div(id='current_snr', style={'display': 'none'}),

        # Abstract
        dbc.Card(
            dbc.CardBody([
                html.P('ConvDip is a convolutional neural network that finds solutions for the inverse problem of the EEG.', className="card-text"),
                html.A("Link to the paper", href='https://www.biorxiv.org/content/10.1101/2020.04.09.033506v1', target="_blank")
            ]
            ), style={'margin': '20px'}
        )
        ,
        # Image
        html.Div([html.Img(src='/assets/architecture.png', height=300)], style={'display': 'inline-block', 'margin': '20px'}),

        dbc.Row([
            # Simulation Panel
            dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H4(["Advanced Options"], style={'marginTop':15}),
                        html.Br(),
                        html.H5('Signal to Noise Ratio (in dB):'),
                        dcc.Markdown('''###### *e.g. single value: 6 or range of values: 6, 9*'''),
                        dbc.Input(id='noise_level_input', value=6),
                        html.Br(),
                        ]),
                    html.Div([
                        html.Br(),
                        html.H5('Number of sources:'),
                        dcc.Markdown('''###### *e.g. single value: 3 or range of values: 1, 5*'''),
                        dbc.Input(id='number_of_sources_input', value=3),
                        html.Br(),
                    ]),
                    html.Div([
                        html.Br(),
                        html.H5('Diameter of sources (in mm):'),
                        dcc.Markdown('''###### *e.g. single value: 35 or range of values: 25, 35*'''),
                        dbc.Input(id='size_of_source_input', value=35),
                        html.Br(),
                        ]),
                    html.Div([
                        html.Br(),
                        html.H5('Shape of sources:'),
                        dcc.Markdown('''###### *If gaussian is selected the diameter of source becomes the full width at half maximum (FWHM)*'''),
                        dbc.RadioItems(
                            id='source_shape',
                            options=[
                                {'label': 'Flat', 'value': 'flat'},
                                {'label': 'Gaussian', 'value': 'gaussian'},
                            ],
                            value='flat'
                        ),
                        html.Br()
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
                ], style={'margin': '20px'}), # end of settings card
            width=3),

            # Simulation: Scalp Map
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                    html.H4("Simulated brain-electric activity", className="card-title"),
                    html.Br(),
                    dcc.Graph(
                        id='sim_source_plot',
                        config={
                            'displayModeBar': False
                            },
                        figure={
                            'data': [],
                            },
                        style={'width': 720},
                        ),
                    html.Br(),
                    html.P("This plot depicts the cortical surface of the brain with its gyri and sulci. Colored in orange you see the simulated electric activity.", className="card-title"),

                        ]),
                    ),
                    width=5),
                
            # Simulation Canvas 2
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Resulting Scalp map of the brain-electric activity", className="card-title"),
                        html.Br(),
                        dcc.Graph(
                            id="sim_scalp_plot",
                            figure={
                                "data": [],
                                },
                            ),
                        html.Br(),
                        html.P("This is a low-resolution representation of the EEG (aerial view). This map is generated by projecting the simulated brain activity to the electrodes of a simulated set of EEG electrodes. This is possible through a forward model which describes the conductive and geometric properties of tissues of the head (e.g. brain, dura, skull). The task of an inverse solution is to infer the activity shown on the left (brain plot) from the scalp map shown.", className="card-title"),
                        ]),
                    ),
                width=4)
                
            ]),  # end of simulation row
        html.Br(),
        html.Br(),
        # Prediction Group
        dbc.Row([
            dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dbc.Select(
                        id='model_selection',
                        options=[
                            {'label': 'ConvDip for gaussian sources', 'value': 'gaussian'},
                            {'label': 'ConvDip Flexible', 'value': 'flex'},
                            {'label': 'ConvDip Paper', 'value': 'paper'},
                            {'label': 'ConvDip for low SNR', 'value': 'lowsnr'},
                            {'label': 'eLORETA', 'value': 'eLORETA'},
                            {'label': 'LCMV Beamforming', 'value': 'lcmv'},
                            {'label': 'Minimum Norm Estimate', 'value': 'MNE'},
                            {'label': 'dSPM', 'value': 'dSPM'},
                        ],
                        value='gaussian'
                        ),
                    html.Br(),
                    html.Br(),
                    dbc.Button('Predict Source', id='predict_button', color="primary"),
                    dbc.Spinner(html.Div(id="loading-output-prediction")),
                    html.Br(),
                    html.Div(id='inv_description'),
                    ])
                , style={'margin': '20px'}),
                width=3),
        # Prediction Figures
        dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H4("Predicted brain activity (i.e. the inverse solution)", className="card-title"),
                html.Br(),
                dcc.Graph(
                    id='pred_source_plot',
                    config={
                        'displayModeBar': False
                        },
                    figure={
                        'data': [],
                        },
                    style={'width': '700px'}
                    ),
                html.Br(),
                html.P("This plot depicts the inverse solution. In the best case this brain activity depicts exactly the simulated activity of the figure above. However, the inverse problem does not have a unique solution, wherefore we can only expect rough approximations.", className="card-title"),
                ]),
            ),
            width=5),


        # Simulation Canvas 2
        dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H4("Resulting Scalp Map of the prediction.", className="card-title"),
                html.Br(),
                dcc.Graph(
                    id="pred_scalp_plot",
                    figure={
                        "data": [],
                        },
                    ),
                html.Br(),
                html.P("This is a low-resolution representation of the EEG (aerial view). This map is generated by projecting the predicted brain activity (figure to the left) to the electrodes of a simulated set of EEG electrodes. This is possible through a forward that describes the conductive and geometric properties of tissues of the head(e.g. brain, dura, skull).", className="card-title"),

                ]),
        ),
        width=4)
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
        State('source_shape', 'value'),
        State('sim_source_plot', 'figure')
        ])

def simulate_sample(*params):
    print('simulating')
    settings = [i for i in params]
    if np.any(settings==None):
        print(f'FIRST CALL RETURNS NOTHING')
        return

    snr = settings[1]
    n_sources = settings[2]
    size = settings[3]
    source_shape = settings[4]

    start = time.time()
    y, x_img, db_choice = simulate_source(snr, n_sources, size, 1, leadfield, pos, source_shape=source_shape)
    end_1 = time.time()
    fig_y, fig_x = make_fig_objects(y, x_img, tris, pos)
    end_2 = time.time()
    print(f'Simulation: {end_1-start}, simulation+plotting: {end_2-start}')
    # fig_y, fig_x = simulate_source(settings[1], settings[2], settings[3])

    spinner_output = 'Simulation is Ready'
    return spinner_output, fig_x, fig_y, y, db_choice

# Callback for the Inverse Solution Dropdown

@app.callback(
        Output('inv_description', 'children'),
        [Input('model_selection', 'value')])

def retrieve_description(value):
    if value == 'flex':
        desc = ['ConvDip trained on a large range of SNRs.', html.Br(), html.Br(), 'SNR: 0 to 8 dB.', html.Br(), 'Source diameters: 25-35', html.Br(), 'Number of Sources: 1 to 5']
    elif value == 'gaussian':
        desc = ['ConvDip trained on gaussian sources.', html.Br(), html.Br(), 'SNR: 3 to 8 dB.', html.Br(), 'Source FWHM: 25-35', html.Br(), 'Number of Sources: 1 to 5']
    elif value == 'paper':
        desc = 'ConvDip as described in the paper.', html.Br(), html.Br(), 'SNR: 6 to 9 dB.', html.Br(), 'Source diameters: 25-35', html.Br(), 'Number of Sources: 1 to 5'
    elif value == 'lowsnr':
        desc = 'ConvDip as trained on low SNR data.', html.Br(), html.Br(), 'SNR: 3 to 6 dB.', html.Br(), 'Source diameters: 25-35', html.Br(), 'Number of Sources: 1 to 5'
    elif value == 'eLORETA':
        desc = 'Exact low-resolution tomography, a commonly used inverse solution.', html.Br(), 'Developed by Pascual-Marqui (2007) and implemented in MNE python.', html.Br(), 'Tip:', html.Br(), 'eLORETA assumes large sources with gaussian distribution. Try to simulate a sample with gaussian shape and 50 mm FWHM and reconstruct it with eLORETA.'
    elif value == 'lcmv':
        desc = 'Linear constraint minimum variance (LCMV) beamformer, a commonly used inverse solution.', html.Br(), 'Developed by Van Veen et al. (1997) and implemented in MNE python.'
    elif value == 'MNE':
        desc = 'Minimum norm estimates (MNE), a commonly used inverse solution.', html.Br(), 'Developed by Hämäläinen & Ilmoniemi (1994) and implemented in MNE python.', html.Br(), 'Tip:', html.Br(), 'MNE produces large sources with gaussian distribution. Try to simulate a sample with gaussian shape and 50 mm FWHM and reconstruct it with MNE.'
    elif value == 'dSPM':
        desc = 'Dynamic statistical parametric mapping (dSPM).', html.Br(), 'Developed by Dale et al. (2000) and implemented in MNE python.', html.Br(), 'Tip:', html.Br(), 'dSPM produces large sources with gaussian distribution. Try to simulate a sample with gaussian shape and 50 mm FWHM and reconstruct it with dSPM.'
    
    return desc


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
    elif inputs[2] == 'gaussian':
        model = model_gaussian
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
