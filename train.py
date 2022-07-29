#!/usr/bin/python3
#
import numpy as np
import math
import random
import rbm
import sys
import pickle
import loading_module as ld

if len(sys.argv) < 5:
    print("usage train data pickle width nrbm")
    sys.exit(-1)

training_data = ld.data_module(sys.argv[1])
number_of_depends = training_data.ncolumns -1
number = training_data.nrows
print(number_of_depends)
print(number)
width = int(sys.argv[3])
n_rbm = int(sys.argv[4])

print(width)
print(n_rbm)

#n_rbm = 3 
#n_rbm = 5 
#n_rbm = 7 
#n_rbm = 11
#n_rbm = 21
#n_rbm = width+1

#the_rbm = rbm.rbm( width-number_of_depends, n_rbm)  #guessed parameters for the number of rbms
# change is due to the new form of data manager.
# width is the number of data points. Each datapoint is number_of_depends wide
the_rbm = rbm.rbm( width*number_of_depends, n_rbm)  #guessed parameters for the number of rbms
the_rbm.its_symmetric()
the_rbm.add_multi_fuzzy(number_of_depends,-2., 2., 101)

print("training starts")
# initialize to more than the  first n points
for i in range(0,n_rbm):
#    j = int(i * number/n_rbm)
    j = int(number*random.random()+0.5)
    point = training_data.a_specific_data_point(j,width)
    print(i, point)
    the_rbm.initial_values( i, point[0])
#    p = np.zeros(width-1)
#    for j in range(0, width-1):
#        p[j] = random.random()-0.5
#    the_rbm.initial_values( i, p)

for j in range(0,10):   
  training_data.reset_data_index()
  for i in range(0,number-width):
    point = training_data.a_data_point(width)
    the_rbm.train_multi_fuzzy(point[0],0.1, 1.0, point[1])
  for i in range(0, n_rbm):  # check that I've been trained
    if the_rbm.seen_multi_data(i) == 0:
        k = int(number*random.random()+0.5)
        point = training_data.a_specific_data_point(k,width)
        the_rbm.initial_values( i, point[0])

the_rbm.make_multi_cumulative()
print("training stops")

save_set = open( sys.argv[2], 'wb')
x = pickle.dumps( the_rbm)
save_set.write(x)
save_set.close()


# testing
for i in range(0, number):
    point = training_data.a_data_point(width)
    x = the_rbm.random_multi_choice( point[0], random.random())
    print( str(point[0])+' '+str(point[1])+ ' ' + str(x))

training_data.set_up_prediction(width)
for i in range(width, number):
    point = training_data.a_prediction_data_point(width)
#    x = the_rbm.random_multi_choice( point[0], random.random())
    x = the_rbm.random_multi_independent_choice( point[0], random.random)
    training_data.add_prediction(x,i)

x = training_data.get_prediction()
x.to_csv("a.csv")

"""
# trial run
run = np.zeros(width-number_of_depends)
dataset.reset()
point = training_data.a_data_point()
for i in range(0,width-number_of_depends):
    run[i] = (point[0])[i]
#    run[i] = 1.
    print(run[i])
window_size = int((width - number_of_depends)/number_of_depends)
for i in range(0,number):
    point = dataset.a_point()
    x = the_rbm.random_multi_choice( run, random.random())
    print(str( (point[0])[0]) + ' , '+ str(x))
    for k in range(0, number_of_depends):
      for j in range(k*window_size+1,k*window_size+window_size):
        run[j-1] = run[j]
      run[k*window_size + window_size -1] = x[k]
#    print(run)
"""
