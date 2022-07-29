#!/usr/bin/python3
#
#
import numpy as np
import pandas as pd

class data_module:
    def __init__(me, file_name, time_series_column=0):
        me.panda = pd.read_csv(file_name)
        # this needs to be converted to an explict conversion.
        # to_numpy does not fail gracefully.
#        me.data = me.panda.to_numpy(dtype="float32")
        temp = me.panda.to_numpy()
        me.data = np.zeros( temp.data.shape)
        for i in range(0, temp.shape[0]):
            for j in range(0,temp.shape[1]):
                try:
                    me.data[i,j] = float(temp[i,j])
                except:
                    me.data[i,j] = float(i)
        me.new_data = np.zeros( me.data.shape) 
        me.time_series_column = time_series_column
        me.nrows = me.data.shape[0]
        me.ncolumns = me.data.shape[1]
        me.which = 0
# scale parameters
        me.maxes = np.zeros(me.ncolumns)
        me.mins = np.zeros(me.ncolumns)
        me.means = np.zeros(me.ncolumns)
        for i in range(0,me.ncolumns):
           me.maxes[i] = max( me.data[0:,i])
           me.mins[i] = min( me.data[0:,i])
           me.means[i] = me.data[0:,i].mean()
# normalize by column.
        for i in range(0,me.ncolumns):
            if i == me.time_series_column:
                continue
            s = 1./(me.maxes[i] - me.mins[i])
            m = me.means[i]
            for j in range(0, me.nrows):
                me.data[j,i] = (me.data[j,i]-m)*s

    def reset_data_index(me,where = 0):
        me.which = where
        
    def a_data_point(me, window):
        if me.which + window +1 >= me.nrows:
            me.reset_data_index()
        dependent = []
        independent = []
        for i in range(0, me.ncolumns):
            if i == me.time_series_column:
                continue
            for j in range(me.which, me.which+window):
                dependent.append( me.data[j,i])
            independent.append(me.data[me.which+window,i])
        me.which += 1
        return dependent,independent

    def a_specific_data_point(me,which, window):
        if which + window +1 >= me.nrows:
 #           return ([],[])  # causes errors
             which = me.nrows-1-window
        dependent = []
        independent = []
        for i in range(0, me.ncolumns):
            if i == me.time_series_column:
                continue
            for j in range(which, which+window):
                dependent.append( me.data[j,i])
            independent.append(me.data[which+window,i])
        return (dependent,independent)

    def set_up_prediction(me, window):
        me.which = 0
#        for j in range( 0, window):
#            for i in range(0, me.ncolumns):
#                me.new_data[j,i] = me.data[j,i]
#        for j in range(0, me.nrows):
#            me.new_data[j,me.time_series_column] = me.data[j, time_series_column] 
        np.copyto( me.new_data, me.data)


    def set_up_average_prediction(me):
        me.average_data = np.zeros( me.data.shape)
        me.in_average = 0

            
    def a_prediction_data_point(me, window):
        if me.which + window  >= me.nrows:
            me.reset_data_index()
        dependent = []
        independent = []
        for i in range(0, me.ncolumns):
            if i == me.time_series_column:
                continue
            for j in range(me.which, me.which+window):
                dependent.append( me.new_data[j,i])
            independent.append(0.)
        me.which += 1
        return dependent,independent


    def add_average_prediction(me):
        me.average_data = me.average_data + me.new_data
        me.in_average += 1

    def add_prediction(me, what, where):
        i = 0
        for j in range(0,me.ncolumns):
# this skips the time column
            if j == me.time_series_column:
                continue
            me.new_data[where,j] = what[i]
            i = i + 1


    def get_prediction(me):
# first denormalize the new data
# denormalize by column.
        for i in range(0,me.ncolumns):
            if i == me.time_series_column:
                continue
            s = (me.maxes[i] - me.mins[i])
            m = me.means[i]
            for j in range(0, me.nrows):
                me.new_data[j,i] = me.new_data[j,i]*s+m
        me.prediction = pd.DataFrame( me.new_data, me.panda.index, me.panda.columns)
        return me.prediction
# design decision here. We could pass a file name, but then we'd limit what the user can do with the 
# prediction.  So we'll return the prediction.

    def get_average_prediction(me):
# first denormalize the new data
# denormalize by column.
        for i in range(0,me.ncolumns):
            if i == me.time_series_column:
                continue
            s = (me.maxes[i] - me.mins[i])
            m = me.means[i]
            for j in range(0, me.nrows):
                me.average_data[j,i] = (me.average_data[j,i]/me.in_average)*s+m
        me.prediction = pd.DataFrame( me.average_data, me.panda.index, me.panda.columns)
        return me.prediction
# design decision here. We could pass a file name, but then we'd limit what the user can do with the 
# prediction.  So we'll return the prediction.

def main():
   q = data_module("macrodata.csv")
   b = q.a_data_point(8)
   print(b)
   b = q.a_data_point(8)
   print(b)
   b = q.a_data_point(8)
   print(b)
   b = q.a_data_point(8)
   print(b)
   b = q.a_data_point(8)
   print(b)
   s = pd.DataFrame( q.data, q.panda.index, q.panda.columns)
   print(s)
   q.set_up_prediction(8)
   x = q.a_prediction_data_point(8)
   print( x )
   print( len(x[0])/8)
   print( len(x[1]) )

if __name__ == "__main__":
    main()
