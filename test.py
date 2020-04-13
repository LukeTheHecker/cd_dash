from plotly.tools import mpl_to_plotly
import dash_core_components as dcc
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import importlib
from mpld3 import fig_to_html, fig_to_dict, save_html
import components
importlib.reload(components)
from components.functions import simulate_source
from plotly.offline import plot_mpl
from mayavi import mlab
import nibabel
import mne
import plotly.graph_objects as go
from scipy import sparse

def mesh_edges(faces):
    """Returns sparse matrix with edges as an adjacency matrix
    Parameters
    ----------
    faces : array of shape [n_triangles x 3]
        The mesh faces
    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    npoints = np.max(faces) + 1
    nfaces = len(faces)
    a, b, c = faces.T
    edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                              shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                      shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                      shape=(npoints, npoints))
    edges = edges + edges.T
    edges = edges.tocoo()
    return edges

def plotly_triangular_mesh(vertices, faces, intensities=None, colorscale="Viridis",
                           flatshading=False, showscale=False, reversescale=False, plot_edges=False):
    ''' vertices = a numpy array of shape (n_vertices, 3)
        faces = a numpy array of shape (n_faces, 3)
        intensities can be either a function of (x,y,z) or a list of values '''

    x, y, z = vertices.T
    I, J, K = faces.T

    mesh = dict(
        type='mesh3d',
        hoverinfo='none',
        x=x, y=y, z=z,
        colorscale=colorscale,
        intensity=intensities,
        flatshading=flatshading,
        i=I, j=J, k=K,
        name='',
        showscale=showscale
    )

    mesh.update(lighting=dict(ambient=0.8,
                              diffuse=1,
                              fresnel=0.1,
                              specular=1,
                              roughness=0.1,
                              facenormalsepsilon=1e-6,
                              vertexnormalsepsilon=1e-12))

    mesh.update(lightposition=dict(x=100,
                                   y=200,
                                   z=0))

    if showscale is True:
        mesh.update(colorbar=dict(thickness=20, ticklen=4, len=0.75))

    if plot_edges is False:  # the triangle sides are not plotted
        return [mesh]
    else:  # plot edges
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        tri_vertices = vertices[faces]
        Xe = []
        Ye = []
        Ze = []
        for T in tri_vertices:
            Xe += [T[k % 3][0] for k in range(4)] + [None]
            Ye += [T[k % 3][1] for k in range(4)] + [None]
            Ze += [T[k % 3][2] for k in range(4)] + [None]
        # define the lines to be plotted
        lines = dict(type='scatter3d',
                     x=Xe,
                     y=Ye,
                     z=Ze,
                     mode='lines',
                     name='',
                     line=dict(color='rgb(70,70,70)', width=1)
                     )
        return [mesh, lines]


pth = "C:\\Users\\Lukas\\Documents\\cd_dash\\assets\\modeling\\"
# stc = simulate_source(6, 3, 33)

stc = mne.read_source_estimate(pth + "\\ResSourceEstimate-lh.stc")

fname_inv = pth + 'inverse-inv.fif'
inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)
src = inverse_operator['src']
DEFAULT_COLORSCALE = [[0, 'rgb(12,51,131)'], [0.25, 'rgb(10,136,186)'],
                      [0.5, 'rgb(242,211,56)'], [0.75, 'rgb(242,143,56)'], [1, 'rgb(217,30,30)']]
lh = nibabel.freesurfer.io.read_geometry(pth + "lh.white")[0]
rh = nibabel.freesurfer.io.read_geometry(pth + "rh.white")[0]
rh[:, 0] = rh[:, 0] + 85
lh_points = lh
rh_points = rh
points = np.r_[lh, rh]
# points = pos
points *= 170
vertices = np.r_[src[0]['vertno'], points.shape[0] + src[1]['vertno']]
use_faces = np.r_[src[0]['tris'], points.shape[0] + src[1]['tris']]
adj_mat = mesh_edges(use_faces)

data = plotly_triangular_mesh(points, use_faces, stc.data[:, 0],
                              flatshading=False, colorscale=DEFAULT_COLORSCALE,
                              showscale=False, reversescale=False, plot_edges=False)



x = data[0]['x']
y = data[0]['y']
z = data[0]['z']
i = data[0]['i']
j = data[0]['j']
k = data[0]['k']
print(data[0])

color = data[0]['intensity']
# color[np.argwhere(color==0)[0][0]] = np.nan
# fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, vertexcolor=color, opacity=0.50, colorscale=data[0]['colorscale'])])
fig = go.Figure(data=data)

fig.show()

input("Press Enter to continue...")