import numpy as np
import pandas as pd
from progressbar import *  # just a simple progress bar


def cross_matching(Data,back_array,k):

    print "In cross match",Data.shape,back_array.shape


    d = {'x': Data[:, 0], 'y': Data[:, 1], 'z': Data[:, 2], 'SM': Data[:, 3], 'Metal': Data[:, 4],
         'u_minus_r': Data[:, 5], 'g_minus_r': Data[:, 6]}

    DF = pd.DataFrame(d)

    SLICED_DATA = pd.DataFrame(columns=DF.columns)

    widgets = ['Processing: ' + str(k) + ' Cluster', Percentage(), ' ', Bar(marker='0', left='[', right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]

    pbar = ProgressBar(widgets=widgets, maxval=back_array.shape[0])
    pbar.start()


    for m in range(back_array.shape[0]):

        for n in range(DF.shape[0]):

            if ((np.abs(DF['x'].loc[n] - back_array[m,0]) < 0.000001) & (
                        np.abs(DF['y'].loc[n] - back_array[m,1]) < 0.000001) & (
                        np.abs(DF['z'].loc[n] - back_array[m,2]) < 0.000001)):
                #print "M here"
                SLICED_DATA = SLICED_DATA.append(DF.loc[n])

        pbar.update(m)

    pbar.finish()

    print "TOTAL SLICED DATA ", SLICED_DATA.shape, back_array.shape
    return SLICED_DATA
