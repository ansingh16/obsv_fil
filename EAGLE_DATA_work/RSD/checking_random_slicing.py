import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from STEP2 import Environment_classifier
from STEP1 import RUN_DISPERSE


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



def transform_back(Ar,R,Ox,Oy,Oz):

    Dat = np.matmul(R.T, Ar)
    Dat[0, :] = Dat[0, :] + Ox
    Dat[1, :] = Dat[1, :] + Oy
    Dat[2, :] = Dat[2, :] + Oz
    return Dat.T

def get_slice(origin_shift,normal,Data_array,rad,k):


    print "Data_shape", Data_array.shape
    a,b,c = normal[0],normal[1],normal[2]

    origin_shiftx,origin_shifty,origin_shiftz = origin_shift[0],origin_shift[1],origin_shift[2]


    z_dir=np.array([0.0,0.0,1.0])

    R = get_rotation_matrix(normal)
    # create x,y
    xx, yy = np.meshgrid([float(i) for i in range(10)],[float(i) for i in range(10)])
    zl = (1.0/c)*(a*origin_shiftx+b*origin_shifty+c*origin_shiftz - rad*(np.sqrt(a**2+b**2+c**2)) - a*xx -b*yy)
    zu = (-1.0/c)*(-a*origin_shiftx-b*origin_shifty-c*origin_shiftz - rad*(np.sqrt(a**2+b**2+c**2)) + a*xx +b*yy)

    trans_matrix = np.zeros_like(Data_array)


    #SHIFT ORIGIN
    trans_matrix[0,:] = Data_array[0,:]-origin_shiftx
    trans_matrix[1,:] = Data_array[1,:]-origin_shifty
    trans_matrix[2,:] = Data_array[2,:]-origin_shiftz



    #print "DISTANCES in P_rot"
    P_rot = np.matmul(R, trans_matrix)


    #print "P_rot shape",P_rot.shape

    #SELECTING THE POINTS
    d = {'x1': P_rot[0, :], 'y1': P_rot[1, :], 'z1': P_rot[2, :]}
    B = pd.DataFrame(d)

    #print "B shape", B.shape

    #clipped = np.zeros((B.loc[np.abs(B['z1']) <= rad].shape[1], B.loc[np.abs(B['z1']) <= rad].shape[0]))

    clippedx=[]
    clippedy=[]
    clippedz=[]

    for ii in range(B.shape[0]):
        #print "Z value",B['z1'].loc[ii]

        if(np.abs(B['z1'].loc[ii])<=rad):
            clippedx.append(B['x1'].loc[ii])
            clippedy.append(B['y1'].loc[ii])
            clippedz.append(B['z1'].loc[ii])

    clipped = np.array([clippedx,clippedy,clippedz])

    np.savetxt('Clipped.csv',clipped[0:2,:].T,delimiter=',',header='# px py')

    print "clipped shape", clipped.shape

    #INVERTING THE MATRIX

    #R_inv = np.linalg.inv(R)   #np.array([[np.cos(-theta),-np.sin(-theta),0],[np.sin(-theta),np.cos(-theta),0],[0,0,1]])

    RUN_DISPERSE('Clipped.csv')

    data_groups, data_filament, data_field, tot_data = Environment_classifier(clipped.T,k)

    #PUT CODE HERE TO INDIVIDUALLY TRANSFORM FIELD,FILAMENT AND CLUSTER GALAXIES.
    #DG_T = np.zeros(data_groups.shape[1],data_groups.shape[0])
    #DG_T[0,:] = data_groups[:,0]
    #DG_T[1, :] = data_groups[:, 1]
    #DG_T[2, :] = data_groups[:, 2]

    DG_new = transform_back(data_groups.T,R,origin_shiftx,origin_shifty,origin_shiftz)

    data_filament[:,0:3] = transform_back(data_filament[:,0:3].T,R,origin_shiftx,origin_shifty,origin_shiftz)
    DF_new = data_filament


    DFi_new = transform_back(data_field.T,R,origin_shiftx,origin_shifty,origin_shiftz)
    tot_data_clipped = transform_back(tot_data.T,R,origin_shiftx,origin_shifty,origin_shiftz)

    '''
    back_array  = np.matmul(R.T,clipped)

    print "Shape: ",back_array.shape

    back_array[0,:] = back_array[0,:]+origin_shiftx
    back_array[1,:] = back_array[1,:]+origin_shifty
    back_array[2,:] = back_array[2,:]+origin_shiftz
    '''

    return DG_new,DF_new,DFi_new,tot_data_clipped,zl,zu





