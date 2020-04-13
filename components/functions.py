import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from skimage.restoration import inpaint
from plotly.tools import mpl_to_plotly
import plotly.express as px
import mne
import matplotlib
import plotly.figure_factory as FF
import time
import os
from numba import njit
from joblib import Parallel, delayed
import ast


def simulate_source(snr, n_sources, size, n):
    ''' This function takes the simulation settings and simulates a pseudo-random sample in brain and sensor space.
    settings keys: ['snr', 'n_sources', 'size']
    '''
    # Check Inputs
    amps = (10, 100)
    if type(snr) == str:
        snr = str2num(snr)
    if type(n_sources) == str:
        n_sources = str2num(n_sources)
        if type(n_sources) == tuple or type(n_sources) == list:
            n_sources = [int(i) for i in n_sources]
        else:
            n_sources = int(n_sources)
    if type(size) == str:
        size = str2num(size)

    print(f'snr={snr}, n_sources={n_sources}, size={size}, n={n}')
    print(f'snr={type(snr)}, n_sources={type(n_sources)}, size={type(size)}, n={type(n)}')

    # Load basic Files
    pth = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'assets\\modeling'))
    ## Leadfield
    with open(pth+'\\leadfield.pkl', 'rb') as f:
        leadfield = pkl.load(f)[0]
    ## Positions
    with open(pth+'\\pos.pkl', 'rb') as f:
        pos = pkl.load(f)[0]
    

    # Generate a source configuration based on settings
    y = np.zeros((n, leadfield.shape[1]))
    x_img = np.zeros((n, 7, 11))
    for s in range(n):
        if type(n_sources) == list or type(n_sources) == tuple:
            srange = np.arange(n_sources[0], n_sources[1]+1)
            n_sources_tmp = np.random.choice(srange)
        else:
            n_sources_tmp = n_sources

        src_centers = np.random.choice(np.arange(0, pos.shape[0]), n_sources_tmp,
                                    replace=False)
        if type(size) == list or type(size) == tuple:
            src_diams = (size[1]-size[0]) * np.random.random_sample(n_sources_tmp) + size[0]
        else:
            src_diams = np.repeat(size, n_sources_tmp)

        src_amps = (amps[1]-amps[0]) * np.random.random_sample(n_sources_tmp) + amps[0]   
        # Smoothing and amplitude assignment
        
        d = {}
        for i in range(src_centers.shape[0]):
            dists = np.sqrt(np.sum((pos - pos[src_centers[i], :])**2, axis=1))
            d[i] = np.where(dists<src_diams[i]/2)
            y[s, d[i]] = src_amps[i]
        
    # Noise
    if type(snr) == tuple or type(snr) == list: # pick noise in some range
        db_choice = (snr[1]-snr[0]) * np.random.random_sample(n) + snr[0]
    else:  # pick definite noise
        if n == 1:
            db_choice = [snr]
        else:
            db_choice = np.repeat(snr, n)

    x_noise = source_to_ximg(y, leadfield, n, db_choice)


    x_img = np.stack(Parallel(n_jobs = -1, backend = 'loky')(delayed(vec_to_sevelev_newlayout)(i) for i in x_noise))

    # x_img = np.stack([vec_to_sevelev_newlayout(i) for i in x_noise], axis=0)
        
    return np.squeeze(y), np.squeeze(x_img)

# @njit(nopython=True, fastmath=False, parallel=True)
def source_to_ximg(y, leadfield, n, db_choice):
    x_noise = np.zeros((n, leadfield.shape[0]))
    

    for s in range(n): # loop simulations
        # Project target
        x = np.sum(y[s, :] * leadfield, axis=1)
        # Add noise
        
        
        relnoise = (10**(db_choice[s] / 10))**-1
        ss = np.sum(x**2)
        sc = len(x)
        rms = np.sqrt(ss/sc)
        noise = np.random.randn(len(x)) * relnoise * rms
        x_noise[s,] = x + noise

        # CAR
        x_noise[s,] = x_noise[s,] - (np.sum(x_noise[s,]) / len(x_noise[s,]))
    return x_noise

def make_fig_objects(y, x_img):
    # Scale vectors
    x_img /= np.max(np.abs(x_img))
    y /= np.max(np.abs(y))

    fig_x = px.imshow(x_img)
    fig_y, _ = brain_plotly(y)
    return fig_y, fig_x

def predict_source(x, pth_model='\\model_paper\\'):
    pth = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'assets\\modeling\\'))
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    # load some stuff
    ## Leadfield
    with open(pth+'\\leadfield.pkl', 'rb') as f:
        leadfield = pkl.load(f)[0]
    ## Positions
    with open(pth+'\\pos.pkl', 'rb') as f:
        pos = pkl.load(f)[0]
    ## Inverse operator, needed to get the triangle-information in the plotting
    fname_inv = pth + '\\inverse-inv.fif'
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)
    tris = inverse_operator['src'][0]['use_tris']


    # Load Model
    print('###LOADING MODEL###')
    print('###LOADING MODEL###')
    print('###LOADING MODEL###')

    json_file = open(pth + pth_model + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(pth + pth_model + "model.h5")
    print("Loaded model from disk")

    # Predict
    if len(x.shape) == 2:
        x = np.expand_dims(np.expand_dims(x, axis=2), axis=0)

    y = np.squeeze(model.predict(x))
 
    # Create forward projection
    # Project target
    x = np.sum(y * leadfield, axis=1)
    # CAR
    x -= np.mean(x)
    x_img = vec_to_sevelev_newlayout(x)

    return y, x_img

def str2num(mystr):
    if len(mystr) == 1:
        return float(mystr)
    else:
        arr = []
        
        return ast.literal_eval(mystr)
        

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

def sevelev_to_vec_newlayout(x):
    # x = np.squeeze(x)
    if len(x.shape) == 2:
        # x_out = np.zeros((1, 31))
        # x = np.expand_dims(x, axis=0)
        x_out = x[[0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0], [4, 3, 2, 0, 2, 4, 3, 1, 0, 2, 4, 5, 3, 2, 4, 5, 6, 7, 8, 10, 8, 6, 5, 7, 9, 10, 8, 6, 7, 8, 6]]
    else:
        x_out = np.zeros(shape=(x.shape[0], 31))
        for i in range(x.shape[0]):
            tmp = np.squeeze(x[i, :])
            # tmp = x[i, ]
            # print(tmp.shape)
            x_out[i,:] = tmp[[0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0], [4, 3, 2, 0, 2, 4, 3, 1, 0, 2, 4, 5, 3, 2, 4, 5, 6, 7, 8, 10, 8, 6, 5, 7, 9, 10, 8, 6, 7, 8, 6]]
    
    return x_out

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

def brain_plotly(y):
    ''' takes triangulated mesh, list of coordinates and a vector of brain activity and plots a plotly triangulated surface '''
    pth = "C:\\Users\\Lukas\\Documents\\cd_dash\\assets\\modeling\\"
    ## Positions
    with open(pth+'pos.pkl', 'rb') as f:
        pos = pkl.load(f)[0]

    ## Inverse operator, needed to get the triangle-information in the plotting
    fname_inv = pth + 'inverse-inv.fif'
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)
    tris = inverse_operator['src'][0]['use_tris']
    

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


    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    ## Plot
    fig1 = FF.create_trisurf(x=x, y=y, z=z,
                            simplices=new_tris,
                            title="Simulated brain activity",
                            color_func=colors,
                            aspectratio=dict(x=1, y=1, z=1),
                            )
    return fig1, colors

def split_data(x, y, key):
    ''' split input and target data based on key'''
    # shuffle first:
    idx = np.arange(0, y.shape[0])
    np.random.shuffle(idx)
    x = x[idx,]
    y = y[idx,]

    tr_range = np.arange(0, np.floor(x.shape[0]*key[0]), dtype='int16')

    if len(key) == 2:
        val_range = np.arange(np.floor(x.shape[0]*key[0]), np.floor(x.shape[0]*key[1]), dtype='int16')
        tst_range = np.arange(np.floor(x.shape[0]*key[1]), x.shape[0], dtype='int16')

        return x[tr_range], x[val_range], x[tst_range], y[tr_range], y[val_range], y[tst_range]

    elif len(key) == 1:
        val_range = np.arange(np.floor(x.shape[0]*key[0]), x.shape[0], dtype='int16')
        
        return x[tr_range], x[val_range], y[tr_range], y[val_range]

def scale_data(x):
    return x / np.max([np.max(np.abs(x)), K.epsilon()])

def semi_supervised_loss(leadfield, batch_size):
    ''' this function gets input X and targets y and calculates multiple metrics. 
    If y appears to be empty (unlabeled data) the function will then calculate only 
    the unsupervised metrics '''
    
    def get_error_sparsity(data, y_pred):
        ''' Sparsity Error - the more zeros the smaller the error '''
        error_sparsity = 0
        for i in range(batch_size):
            flat = K.flatten(y_pred[i, :])
            flat = flat / K.max(flat) # scale to maximum = 1
            n_not_near_zero = tf.shape(tf.where(flat > 0.10))[0]
            n_total = tf.shape(flat)
            error_sparsity += n_not_near_zero / n_total

        error_sparsity /= batch_size
        return error_sparsity

    def get_fwd_corr(data, y_pred):
        ''' Forward Projection Error: correlation between input 
        map and forward-projection-of-prediction-map'''

        x = data[:, y_pred.shape[1]:]
        
        @tf.function
        def project(a, b):
            return K.sum(a * b, axis = 1) 
        
        n = 31
        error_fwd = 0
        for i in range(batch_size):
            pred_fwd = project(y_pred[i, :], leadfield)
            # print(f"shape1: {pred_fwd.shape} shape2: {x[i, :].shape}")
            cov = (K.sum((x[i, :] - K.mean(x[i, :])) * (pred_fwd - K.mean(pred_fwd)))) * (1./(n))
            r = cov / (K.std(x[i, :]) * K.std(pred_fwd))
            error_fwd += 2 - (r + 1)  # set range of correlation from 0 (highly negatively correlated) to 2 highly positively correlated
        error_fwd /= batch_size
        return error_fwd

    def source_corr(data, y_pred):

        print(f'batch_size={batch_size}')
        source_corr_loss = 0
        zero_cnt = 0  # count variable that checks occurences of all-zero targets
        n = y_pred.shape[1]
        for i in range(batch_size):
            y_pred_tmp = y_pred[i,]
            y_true_tmp = data[i, 0:y_pred_tmp.shape[0]]
            if K.sum(K.abs(y_true_tmp)) == 0:
                # print('all zeros, skipping')
                zero_cnt += 1
                continue
            ''' pearson correlation between predicted and true source'''
            cov = (K.sum((y_true_tmp - K.mean(y_true_tmp)) * (y_pred_tmp - K.mean(y_pred_tmp)))) / (n-1)
            r = cov / (K.std(y_true_tmp) * K.std(y_pred_tmp))

            source_corr_loss += 1-K.square(r)

        return source_corr_loss / (batch_size - zero_cnt)

    def source_loss(data, y_pred):
        ''' normalized mean squared error with ignoring of empty targets (unsupervised samples)'''
        non_zero_samples = tf.dtypes.cast(batch_size, tf.float32)
        error = 0
        for i in range(batch_size):
            targ = data[i, :y_pred.shape[1]]
            multi = tf.dtypes.cast(K.clip(tf.math.count_nonzero(targ), 0, 1), tf.float32)  # Multi: 0 if all zero, 1 if not all zero
            y_pred_tmp = y_pred[i, ] / K.max([K.max(K.abs(y_pred[i, ])), K.epsilon()])  # norm sample
            targ = targ / K.max([K.max(K.abs(targ)), K.epsilon()]) # norm target, avoid division by zero with K.epsilon()
            error += K.mean(K.square(targ - y_pred_tmp)) * multi # calc mean squared error but nullify it if target is all zero
            non_zero_samples = K.switch(multi==0, non_zero_samples-1, non_zero_samples) # if target was all zero diminish the divider
        
        return error / K.clip(non_zero_samples, 1, batch_size)

    def weighted_loss(data, y_pred, w):
        data = tf.dtypes.cast(tf.squeeze(data), tf.float32)
        y_pred = tf.dtypes.cast(tf.squeeze(y_pred), tf.float32)
        def weighted(data, y_pred):
            ''' normalized mean squared error with ignoring of empty targets (unsupervised samples) and weighting of false positive errors'''
            non_zero_samples = tf.dtypes.cast(batch_size, tf.float32)
            error = 0
            for i in range(batch_size):
                targ = data[i, :y_pred.shape[1]]
                multi = tf.dtypes.cast(K.clip(tf.math.count_nonzero(targ), 0, 1), tf.float32)  # Multi: 0 if all zero, 1 if not all zero
                # Normalize by to max=1
                # y_pred_tmp = y_pred[i, ] / K.max([K.max(K.abs(y_pred[i, ])), K.epsilon()])  # norm sample
                y_pred_tmp = K.switch(tf.equal(tf.reduce_sum(y_pred[i, ]), 0), y_pred[i, ], y_pred[i, ] / K.max(y_pred[i, ]))
                targ = targ / K.max([K.max(K.abs(targ)), K.epsilon()]) # norm target, avoid division by zero with K.epsilon()
                # Error
                tmp_error = K.square(targ - y_pred_tmp)
                # Weight false positive errors
                tmp_error = K.switch(K.equal(targ, 0), w * tmp_error , tmp_error)
                tmp_error = tf.dtypes.cast(tmp_error, tf.float32)
                # Nullify
                error += K.mean(tmp_error) * multi # calc mean squared error but nullify it if target is all zero
                non_zero_samples = K.switch(multi==0, non_zero_samples-1, non_zero_samples) # if target was all zero diminish the divider

            # y_true = data[:, 0:y_pred.shape[1]]
            # y_true = y_true / K.max(y_true)
            # y_pred = y_pred / K.max(y_pred)
            
            # error = K.square(y_true - y_pred)
            # error = K.switch(K.equal(y_true, 0), w * error , error)

            return error / K.clip(non_zero_samples, 1, batch_size) 

        return weighted(data, y_pred)

    def combined_loss(data, y_pred):

        error_fwd = get_fwd_corr(data, y_pred)

        # error_sparsity = get_error_sparsity(data, y_pred)

        # error_source_corr = source_corr(data, y_pred)

        # error_source_loss = source_loss(data, y_pred) * 10.0

        error_wsource_loss = weighted_loss(data, y_pred, 10.0) * 50.0

        loss = tf.dtypes.cast(error_wsource_loss, tf.float32)  + tf.dtypes.cast(error_fwd, tf.float32)  # + tf.dtypes.cast(error_sparsity, tf.float16) + tf.dtypes.cast(error_source_loss, tf.float16)  + 
        return loss

    return combined_loss

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