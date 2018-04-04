import numpy as np
import matplotlib.pyplot as plt
from halotools.mock_observables import return_xyz_formatted_array




import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import pandas as pd



window=500

data_Filament = pd.read_csv('Filament_output.csv',header=1,names=['x','y','z','d_per','d_long','length_fil','SM','Metal','u_minus_r','g_minus_r'])
data_Fil = data_Filament[data_Filament['d_per']<=5.0]
#sort_data = data_Fil.sort_values('d_per')


data_Fil.set_index('d_per',inplace=True)

A = pd.DataFrame()

A['u_minus_r']= data_Fil['u_minus_r'].sort_index()


A['median']= pd.rolling_median(A['u_minus_r'], 600)

print A['median']

plt.scatter(A.index, A['u_minus_r'],s=1)
plt.plot(A.index,A['median'],'r')

plt.show()
