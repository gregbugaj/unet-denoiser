import random
import string
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':   
    randomlist = []
    letters = string.digits
    # print ( ''.join(random.choice(letters) for i in range(10)) )
    choice = np.random.choice([0,1], p =[0.3, 0.7])
    _file = open('digits.txt', 'w') 

    for i in tqdm(range(100000)):
        k = random.randint(1, 8)
        n = np.random.choice(['', '$', '$ ','-'], p =[0.5, 0.2, 0.2, .1]) 
        n += (''.join(random.choice(letters) for i in range(k)))
        if np.random.choice([0,1], p =[0.3, 0.7]) == 1:
            n += '.'
            n += ''.join(random.choice(letters) for i in range(k//2))

        _file.write(n)
        _file.write('\n')

    _file.close() 
    letters = '0123456789.-/+-%#:'
