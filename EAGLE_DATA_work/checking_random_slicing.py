import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot_line(theta,X,C):
    m = np.tan(theta)
    return m*X + C

'''
def get_rotation_matrix(vec1,vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    v = np.cross(vec1, vec2)

    c = np.dot(vec1, vec2)

    V_x = np.array([[0.0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])

    R = np.eye(3, 3) + V_x + np.matmul(V_x, V_x) * (1.0 / (1.0 + c))

    return R
'''

def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [0.0, 0.0, 1.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )




def get_slice(origin_shift,normal,Data_array,rad):

    a,b,c = normal[0],normal[1],normal[2]

    origin_shiftx,origin_shifty,origin_shiftz = origin_shift[0],origin_shift[1],origin_shift[2]



    R = get_rotation_matrix(normal)
    # create x,y
    xx, yy = np.meshgrid([float(i) for i in range(10)],[float(i) for i in range(10)])
    zl = (1.0/c)*(a*origin_shiftx+b*origin_shifty+c*origin_shiftz - rad*(np.sqrt(a**2+b**2+c**2)) - a*xx -b*yy)
    zu = (-1.0/c)*(-a*origin_shiftx-b*origin_shifty-c*origin_shiftz - rad*(np.sqrt(a**2+b**2+c**2)) + a*xx +b*yy)

    trans_matrix = np.zeros_like(Data_array)

    print origin_shiftx,origin_shifty,origin_shiftz

    #SHIFT ORIGIN
    trans_matrix[0,:] = Data_array[0,:]-origin_shiftx
    trans_matrix[1,:] = Data_array[1,:]-origin_shifty
    trans_matrix[2,:] = Data_array[2,:]-origin_shiftz

    print "DIsttnaces in translated"

    for i in range(trans_matrix.shape[1]):

        print np.sqrt((trans_matrix[0,i])**2 + (trans_matrix[1,i])**2 + (trans_matrix[2,i])**2)


    print "DISTANCES in P_rot"
    P_rot = np.matmul(R, trans_matrix)

    for i in range(trans_matrix.shape[1]):
        print np.sqrt((P_rot[0, i]) ** 2 + (P_rot[1, i]) ** 2 + (P_rot[2, i]) ** 2)
        print "X,Y,Z",P_rot[0, i],P_rot[1, i],P_rot[2, i]

    print "P_rot shape",P_rot.shape

    #SELECTING THE POINTS
    d = {'x1': P_rot[0, :], 'y1': P_rot[1, :], 'z1': P_rot[2, :]}
    B = pd.DataFrame(d)

    print "B shape", B.shape

    #clipped = np.zeros((B.loc[np.abs(B['z1']) <= rad].shape[1], B.loc[np.abs(B['z1']) <= rad].shape[0]))

    clippedx=[]
    clippedy=[]
    clippedz=[]

    for ii in range(B.shape[0]):
        #print "Z value",B['z1'].loc[ii]

        if(np.abs(B['z1'].loc[ii])<=rad):
            #print np.abs(B['z1'].loc[ii])
            clippedx.append(B['x1'].loc[ii])
            clippedy.append(B['y1'].loc[ii])
            clippedz.append(B['z1'].loc[ii])

    clipped = np.array([clippedx,clippedy,clippedz])

    print "clipped shape", clipped.shape



    #INVERTING THE MATRIX

    #R_inv = np.linalg.inv(R)   #np.array([[np.cos(-theta),-np.sin(-theta),0],[np.sin(-theta),np.cos(-theta),0],[0,0,1]])

    back_array  = np.matmul(R.T,clipped)

    print "Shape: ",back_array.shape

    back_array[0,:] = back_array[0,:]+origin_shiftx
    back_array[1,:] = back_array[1,:]+origin_shifty
    back_array[2,:] = back_array[2,:]+origin_shiftz

    return back_array,zl,zu,P_rot



