import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from skimage.restoration import inpaint
from plotly.tools import mpl_to_plotly
import plotly.express as px
import mne
import matplotlib
import plotly.figure_factory as FF
FF.create_trisurf

def simulate_source(snr, n_sources, size):
    ''' This function takes the simulation settings and simulates a pseudo-random sample in brain and sensor space.
    settings keys: ['snr', 'n_sources', 'size']
    '''

    matplotlib.use('Agg')

    settings = {'amplitude': (10, 100), # amplitude range, not too important
                'snr': float(snr),
                'n_sources': int(n_sources),
                'size': int(size)}
    # print(f'simulating {n_sources} sources of {size} mm diameter and an snr of {snr}')
    
    # Load basic Files
    pth = "C:\\Users\\Lukas\\Documents\\cd_dash\\assets\\modeling\\"
    ## Leadfield
    with open(pth+'leadfield.pkl', 'rb') as f:
        leadfield = pkl.load(f)[0]
    ## Positions
    with open(pth+'pos.pkl', 'rb') as f:
        pos = pkl.load(f)[0]
    ## Inverse operator, needed to get the triangle-information in the plotting
    fname_inv = pth + 'inverse-inv.fif'
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)
    tris = inverse_operator['src'][0]['use_tris']

    # Generate a source configuration based on settings
    y = np.zeros((leadfield.shape[1],))

    if not isinstance(settings['n_sources'], int) and not isinstance(settings['n_sources'], float):
        srange = np.arange(settings["n_sources"][0], settings["n_sources"][1]+1)
        n_sources = np.random.choice(srange)
    else:
        n_sources = settings['n_sources']

    #print(n_sources)
    src_centers = np.random.choice(np.arange(0, pos.shape[0]), n_sources,
                                   replace=False)
    if not isinstance(settings['size'], int) and not isinstance(settings['size'], float):
        src_diams = (settings["size"][1]-settings["size"][0]) * np.random.random_sample(n_sources) + settings["size"][0]
        src_amps = (settings["amplitude"][1]-settings["amplitude"][0]) * np.random.random_sample(n_sources) + settings["amplitude"][0]   
    else:
        src_diams = np.repeat(settings['size'], n_sources)
        src_amps = (settings["amplitude"][1]-settings["amplitude"][0]) * np.random.random_sample(n_sources) + settings["amplitude"][0]   
    # Smoothing and amplitude assignment
    d = {}
    for i in range(len(src_centers)):
        dists = np.sqrt(np.sum((pos - pos[src_centers[i], :])**2, axis=1))
        d[i] = np.where(dists<src_diams[i]/2)
        y[d[i]] = src_amps[i]

    # Project target
    x = np.sum(y * leadfield, axis=1)
    # Add noise
    if np.sum(settings['snr']) == 0:
        x_noise = x
    else:
        x_noise, snr = addNoise(x, settings['snr'])
    # CAR
    x_noise -= np.mean(x_noise)
    x_img = vec_to_sevelev_newlayout(x_noise)
    # Scale img
    x_img /= np.max(np.abs(x_img))
    
    # fig_x = plt.figure()
    # plt.imshow(x_img)
    # fig_x = mpl_to_plotly(fig_x)

    fig_x = px.imshow(x_img)

    fig_y, _ = brain_plotly(tris, pos, y)
    # print(f'simulating {n_sources} sources of {src_diams} mm diameter and an snr of {snr}')
    # print(f'source center(s): {pos[src_centers, :]}')

    return fig_y, fig_x
    # return stc

def vec_to_sevelev_newlayout(x):
    ''' convert a vector consisting of 32 electrodes to a 7x11 matrix using 
    inpainting '''
    x = np.squeeze(x)
    w = 11
    h = 7
    elcpos = np.empty((h, w))
    elcpos[:] = np.nan
    elcpos[0, 4] = x[0]
    elcpos[1, 3] = x[1]
    elcpos[1, 2] = x[2]
    elcpos[2, 0] = x[3]
    elcpos[2, 2] = x[4]
    elcpos[2, 4] = x[5]
    elcpos[3, 3] = x[6]
    elcpos[3, 1] = x[7]
    elcpos[4, 0] = x[8]
    elcpos[4, 2] = x[9]
    elcpos[4, 4] = x[10]
    
    elcpos[5, 5] = x[11]
    elcpos[5, 3] = x[12]
    elcpos[5, 2] = x[13]
    elcpos[6, 4] = x[14]
    elcpos[6, 5] = x[15]
    elcpos[6, 6] = x[16]

    elcpos[5, 7] = x[17]
    elcpos[5, 8] = x[18]
    elcpos[4, 10] = x[19]
    elcpos[4, 8] = x[20]
    elcpos[4, 6] = x[21]
    elcpos[3, 5] = x[22]
    elcpos[3, 7] = x[23]
    elcpos[3, 9] = x[24]
    
    elcpos[2, 10] = x[25] # FT10
    elcpos[2, 8] = x[26]
    elcpos[2, 6] = x[27]
    elcpos[1, 7] = x[28]
    elcpos[1, 8] = x[29]
    elcpos[0, 6] = x[30]
    # elcpos[1, 5] = 5 Fz was reference
    # elcpos[6, 2] = 28 PO9 deleted
    # elcpos[6, 8] = 32 PO10 deleted

    mask = np.zeros((elcpos.shape))
    mask[np.isnan(elcpos)] = 1
        
    
    return inpaint.inpaint_biharmonic(elcpos, mask, multichannel=False)

def addNoise(x, db):
    if len(x.shape) > 2:
        n_samples = x.shape[0]
    else:
        n_samples = 1
        x = np.expand_dims(x, axis=0)

    x_out = np.zeros_like(x)
    if not isinstance(db, int) and not isinstance(db, float):
        db_choice = (db[1] - db[0]) * np.random.random_sample(n_samples) + db[0]                
    else:
        db_choice = np.repeat(db, n_samples)

    for i in range(n_samples):
        relnoise = (10**(db_choice[i] / 10))**-1
        ss = np.sum(x[i]**2)
        sc = x.shape[1]
        rms = np.sqrt(ss/sc)
        noise = np.random.randn(x.shape[1]) * relnoise * rms
        x_out[i] = x[i] + noise
    
    
    return x_out, db_choice

def brain_plotly(tris, pos, y):
    ''' takes triangulated mesh, list of coordinates and a vector of brain activity and plots a plotly triangulated surface '''

    # Concatenate tris so that it covers the whole brain
    tmp = tris + int(pos.shape[0]/2)
    new_tris = np.concatenate([tris, tmp], axis=0)
    # Calculate the true value for each triangle (which is the mean of the triangle's vertices)
    colors = []
    for tri in new_tris:
        positions = pos[tri, :]
        indices = []
        for j in positions:
            indices.append(np.where((pos == j).all(axis=1))[0][0])
        colors.append(np.mean(y[indices]))


    # scale y
    y /= np.max(y)

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    ## Plot
    fig1 = FF.create_trisurf(x=x, y=y, z=z,
                            simplices=new_tris,
                            title="Simulated brain activity",
                            color_func=colors,
                            aspectratio=dict(x=1, y=1, z=1)
                            )
    return fig1, colors


def quickplot(data, pth_res, fig, del_below=0.0, title='ConvDip', background='white', views = ['lat']):
    ''' quickly plot a source: not used for the convdip webapp '''
    backend = 'mayavi'
    if backend=='mayavi':
        fig = [None, None]
    fleft = []
    fright = []
    data = np.squeeze(data)
    mask_below_thr = data < (np.max(data) * del_below)
    data[mask_below_thr] = 0
    # Read some dummy object to assign the voxel values to
    if len(data) == 20484:
        try:
            a = mne.read_source_estimate(pth_res + "\\sourcetemplate-lh.stc")
        except:
            a = mne.read_source_estimate(pth_res + "/sourcetemplate-lh.stc")
    else:
        try:
            a = mne.read_source_estimate(pth_res + "\\ResSourceEstimate-lh.stc")
        except:
            a = mne.read_source_estimate(pth_res + "/ResSourceEstimate-lh.stc")

    # assign precomputed voxel values
    for i in range(a.data.shape[1]):
        a.data[:, i] = data

    # its a template average
    a.subject = "fsaverage"
    # Use stc-object plot function
    clim = {'kind': 'percent',
            'lims': (20, 50, 100)
            }
    
    for view in views:
        fleft.append(a.plot(hemi='lh', initial_time=0.5, surface="white", backend=backend, title=title+'_lh' , clim=clim, transparent=True, views=view, background=background, foreground='black', verbose=False, colorbar=False))
        fright.append(a.plot(hemi='rh', initial_time=0.5, surface="white", backend=backend, title=title+'_rh' , clim=clim, transparent=True, views=view, background=background, foreground='black', verbose=False, colorbar=False))

    # return a
    return fleft, fright