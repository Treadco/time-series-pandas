#!/usr/bin/python3

import pandas as pd
import numpy as np

import sys

data = pd.read_csv( sys.stdin)
just_data = data.to_numpy()

print( '"time","value"')
which = 0
for i in range(0, just_data.shape[0]):
    for j in range( 1, just_data.shape[1]):
        print( str(which)+','+str( just_data[i,j]) )
        which = which + 1

