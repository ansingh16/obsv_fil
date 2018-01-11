import pandas as pd
import numpy as np


with open('SLICE_DATA23.dat.NDnet_s3.up.NDskl.S010.a.NDskl') as infile, open('testing_extraction.dat', 'w') as outfile:
    copy = False
    for line in infile:
        if line.strip() == "[FILAMENTS]":
            copy = True
        elif line.strip() == "[CRITICAL POINTS DATA]":
            copy = False
        elif copy:
            outfile.write(line)



outfile.close()

fil_file = open('testing_extraction.dat','r')

lines = fil_file.readlines()

print len(lines)

total_filaments = lines[0]


count=1
k=0
for i in range(len(lines)):
    if lines[i].startswith(' '):
       k=k+1
    else:   
       count=count+1

print count

file_list = []
for i in range(count-1):
    file_list.append('filament'+str(i))
    file_list[i] = open('filament' + str(i),'w')



l=0
for i in range(1,len(lines)):
    if lines[i].startswith(' '):
       #print l,i,len(file_list),len(lines)
       file_list[l].write(lines[i])
    else:   
       l=l+1


for i in range(count-1):
    file_list[i].close()




d={}
for i in range(1,count-1):
    dat = np.genfromtxt('filament'+str(i))
    #print dat.shape
    d['filament'+str(i)+'x'] = dat[:,0]
    d['filament'+str(i)+'y'] = dat[:,1]
    X = 'filament'+str(i)+'x'
    Y = 'filament'+str(i)+'y'
    d[X] = dat[:,0]
    d[Y] = dat[:,1]

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.iteritems() ]))
#df = pd.DataFrame(d)

print df.shape

df.to_csv('FINAL_TABEL.csv', header=True, index=False, sep='\t', mode='a')
