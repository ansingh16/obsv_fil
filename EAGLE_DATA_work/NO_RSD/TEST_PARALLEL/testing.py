from multiprocessing import Pool
import numpy as np


def doubler(number):
    b = []
    b.append(2*number)
    b.append(4*number)

    c = []
    c.append(6*number)
    c.append(8*number)


    d = []
    d.append(10*number)
    d.append(12*number)

    return b,c,d
 

if __name__ == '__main__':
    result1=[]
    numbers = [5, 3, 7,8,9,11]
    pool = Pool(processes=3)
    result = pool.map(doubler, numbers)
    
    x1=[]
    y1=[]
    z1=[]
    for r in range(len(result)):
        x,y,z=result[r]
        x1+=x
        y1+=y
        z1+=z
        
    print x1,y1,z1
