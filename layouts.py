import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
from components import Header
from components.functions import simulate_source
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

############################### REMOVE ME LATER MAYBE
axis_template = {
    "showbackground": False,
    "visible" : False
}

plot_layout = {
    "title": "",
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": False,
    'uirevision':'same_all_the_time', #this keeps camera position etc the same when data changes.
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectmode" : "data",
        "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
        "annotations": [],
    },
}
###############################


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
        # Simulation Panel
        dbc.Card(
            dbc.CardBody([
            html.Div([
            html.H6(["Advanced Options"], style={'marginTop':15}),
            html.Label('Signal to Noise Ratio (in dB):'),
            dcc.Input(id='noise_level_input', value=6),
            html.Br(),
            ]),
            html.Div([
            html.Label('Number of Sources:'),
            dcc.Slider(
                id='number_of_sources_input',
                min=1,
                max=5,
                step=1,
                value=3,
                marks= {
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5'
                    }
                ),
            html.Br(),
        ], style={'width': '12%'}),
        html.Div([
            html.Label('Size of sources (mm):'),
            dcc.Input(id='size_of_source_input', value=35),
            html.Br(),
        ]),
        html.Div([
            html.Br(),
            html.Button('Simulate Sample', id='sim_button'),
            ]),
        html.Div([
            html.Div(id='output_container_button',
             children='Enter the values and press button')
        ])]
            ), style={'width':'20%', 'margin': '10px'}
        )
       ,
        # Simulation Canvas
        html.Div([
            html.Br(),
            dcc.Graph(
                id='sim_scalp_plot',
                config={
                    'displayModeBar': False
                },
                figure={
                    'data': [],
                }
            )

        ]),
        # Simulation Canvas 2
        html.Div([
            html.Br(),
            dcc.Graph(
                id="sim_source_plot",
                figure={
                    "data": [],
                },
            )

        ])
        # Example Graph
        # dcc.Graph(
        # id='basic-interactions',
        # figure=example_plot()
        # )
        ], className="subpage"), 
    ], className="page")

######################## START ConvDip Callbacks ########################
## Simulation Callback:
# Output('output_container_button', 'children'),
# Output('sim_plot_graphs', 'figure'), 
@app.callback(
        [Output('sim_scalp_plot', 'figure'),
        Output('sim_source_plot', 'figure')], 
        [Input('sim_button', 'n_clicks')],
        [State('noise_level_input', 'value'), 
        State('number_of_sources_input', 'value'),
        State('size_of_source_input', 'value'),
        State('sim_source_plot', 'figure')
        ])

def simulate_sample(*params):
    settings = [i for i in params]
    if np.any(settings==None):
        return
    # y, x = simulate_source(*settings[1:])
    fig_y, fig_x = simulate_source(settings[1], settings[2], settings[3])
    # print(f'fig={settings[4]}')
    # if settings[4]['data']  == []:
    #     return fig_x, fig_y
    # else:
    return fig_x, fig_y

    # return f'settings: {[param for param in params]}'
     