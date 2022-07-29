#!/usr/bin/python3
#
#  multi-fuzzy
#  (C) 2022 Treadco Software
#
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# python version of the fuzzy rbm
# supports the non-fuzzy version.
#
# 
license =''' 
Copyright (c) 2017  Treadco LLC, Amelia Treader, Robert W Harrison

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import sys,os
from math import exp as exp
from math import sqrt as sqrt

import fuzzy


#
# use pickle.dump(instance,file)
#  and pickle.load(file)
#
# to save and restore data. file is a Python FILE object so 
# it's opened first.
#
#

class rbm:  #the basic rbm
  def __init__(me, number_visible, number_hidden):
    me.nvis = number_visible 
    me.nhid = number_hidden
    me.layers = []
    me.energies = []
    me.hidden = []
    me.symmetric_encoding = False
    me.valid = []
# initialize the space
# making essentially empty lists means that we can avoid using append etc
    me.scratch = np.full(number_visible,0.0)
    for i in range(0,number_hidden):
        me.layers.append(np.float32(np.zeros(number_visible)))
        me.hidden.append(0)
        me.energies.append(0.) 
        me.valid.append(True)

  def initial_values(me, who, what):
      who = who %me.nhid
      x = me.layers[who]
      print(x)
#      me.layers[who] = np.add( me.layers[who], what)
      for i in range(0,len(x)):
          x[i] = what[i]
      print(x)

  def add_fuzzy(me, thmin,thmax,thenumber):
    me.fuzz = []
    for i in range(0, me.nhid):
        me.fuzz.append( fuzzy.fuzzy(thmin,thmax,thenumber))

  def add_multi_fuzzy(me, how_many, thmin,thmax,thenumber):
    me.mfuzz = []
    me.how_many_fuzzy = how_many

    for j in range(0, me.how_many_fuzzy):
      me.mfuzz.append( [])
      for i in range(0, me.nhid):
        me.mfuzz[j].append( fuzzy.fuzzy(thmin,thmax,thenumber))

  def reinitialize_fuzzy(me):
     for i in range(0, me.nhid):
        me.fuzz[i].initialize_counts()

  def reinitialize_multi_fuzzy(me):
    for j in range(0, me.how_many_fuzzy):
     for i in range(0, me.nhid):
        me.mfuzz[j][i].initialize_counts()

  def seen_data(me,i):
      return  me.fuzz[i].counts.sum()

  def seen_multi_data(me,i):
      x = 0
      for j in range(0,me.how_many_fuzzy):
          x += me.mfuzz[j][i].counts.sum()
      return x

  def make_cumulative(me):
     for i in range(0, me.nhid):
        me.valid[i] = me.fuzz[i].make_cumulative_distribution()

  def make_multi_cumulative(me):
    for j in range(0, me.how_many_fuzzy):
     for i in range(0, me.nhid):
        me.valid[i] = me.mfuzz[j][i].make_cumulative_distribution()

  def random_choice( me, data, a_random_float):
    the_layer = me.the_best_layer( data,True)
    ib = the_layer[0]
    return me.fuzz[ib].random_choice(a_random_float)

  def random_multi_choice( me, data, a_random_float):
    the_layer = me.the_best_layer( data,True)
    ib = the_layer[0]
    retval = []
    for j in range(0, me.how_many_fuzzy):
       retval.append( me.mfuzz[j][ib].random_choice(a_random_float) )
    return retval

  def random_multi_independent_choice( me, data, an_rng):
    the_layer = me.the_best_layer( data,True)
    ib = the_layer[0]
    retval = []
    for j in range(0, me.how_many_fuzzy):
       retval.append( me.mfuzz[j][ib].random_choice(an_rng()) )
    return retval

  def reconstruct_raw(me, data, use_best = True):
    the_layer = me.the_best_layer(data,use_best) 
    ib = the_layer[0]
    a = me.layers[ib]
    sign = 1.
    if np.dot(a,data) < 0.:
       sign = -1.
    for i in range(0,me.nvis):
        me.scratch[i] = a[i]*sign
    num = np.dot(data,data)
    denom = np.dot(me.scratch,me.scratch)
    return me.scratch.__mul__( sqrt(num/denom))

 
  def reconstruct(me, data, use_best = True):
    the_layer = me.the_best_layer(data,use_best) 
    ib = the_layer[0]
    a = me.layers[ib]
    sign = 1.
    if me.hidden[ib] < 0.:
       sign = -1.
#
# there may be a clever numpy solution for this loop
#
    for i in range(0,me.nvis):
       me.scratch[i] = -1.
       if a[i] < 0.:
           me.scratch[i] =  1.
    return me.scratch.__mul__(sign) 

  def its_symmetric(me):
      me.symmetric_encoding = True

  def the_best_layer(me, data, use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
    ib = np.argmin(me.energies)
    eb = me.energies[ib]
#    ib = 0
#    eb = me.energies[0]
#    for i in range(1,me.nhid):
#       if me.energies[i] <= eb:
#          ib = i
#          eb = me.energies[i]
    return ib,eb

  def the_best_built_layer(me, data, use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
#    ib = 0
#    eb = 10.e10
#    for i in range(0,me.nhid):
#       if me.energies[i] < -1.0 and use_best:
#          me.energies[i] = 10.e10
#       if me.energies[i] <= eb:
#          ib = i
#          eb = me.energies[i]
    ib = np.argmin(me.energies)
    eb = me.energies[ib]
    while use_best and eb < -1.:
           me.energies[ib] = 10.e10
           ib = np.argmin(me.energies)
           eb = me.energies[ib]
    return ib,eb

  def estimate_EV( me, data, use_best = True):
    ib = me.the_best_layer(data,use_best)[0]
    return me.fuzz[ib].expected_value()

  def estimate_multi_EV( me, data, use_best = True):
    ib = me.the_best_layer(data,use_best)[0]
    retval = []
    for i in range(0, me.how_many_fuzzy):
       retval.append( me.mfuzz[i][ib].expected_value() )
    return retval 


  def assign_hidden_and_reconstruction_energy(me, data):
    for i in range(0, me.nhid):
       eraw = np.dot( data, me.layers[i])
       if not me.valid[i]:
           eraw = 10.e10
# some weird data type problem
#       ebest = np.dot( data.__abs__(), (me.layers[i]).__abs__())
       ebest = 0.
       for j in range(0, len(data)):
           ebest +=  abs(data[j])*abs( (me.layers[i])[j])
       if ebest == 0.0 :
#          ebest = 1.0
# this forces the RBM to train this layer.
          me.energies[i] = -10.e10
#         me.energies[i] = 0.0
          me.hidden[i] =  1.0
       elif  me.symmetric_encoding:
            me.hidden[i] = 1.0
            me.energies[i] = eraw/ebest
       else:
         if eraw > 0.:
            me.hidden[i] = -1.0
            me.energies[i] = -eraw/ebest
         else:
            me.hidden[i] = 1.0
            me.energies[i] = eraw/ebest


  def assign_hidden_and_energy(me, data):
    for i in range(0, me.nhid):
       eraw = np.dot( data, me.layers[i])
       if not me.valid[i]:
           eraw = 10.e10
       if me.symmetric_encoding:
            me.hidden[i] = 1.0
            me.energies[i] = eraw
       else:
         if eraw > 0.:
            me.hidden[i] = -1.0
            me.energies[i] = -eraw
         else:
            me.hidden[i] = 1.0
            me.energies[i] = eraw

  def trainOmatic(me,data,beta,learning_rate,use_best = True):
      me.train(data,beta,learning_rate,use_best)
      me.antitrain(data,beta,learning_rate*0.1,use_best)


# this is the online one pass algorithm
  def train_fuzzy(me,data,beta,learning_rate,dependent_value, use_best = True):
    if len(me.fuzz) == 0:
       print("You Must define fuzzy first to use this")
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
# select the row to train.
    imin = 0
    emin = me.energies[0]
    for i in range(1,me.nhid):
  #    print( emin, me.energies[i])
      if emin >= me.energies[i] :
         imin = i
         emin = me.energies[i]
#
# emin,imin now point to the best row
#    
    hsign = me.hidden[imin]
    alayer = me.layers[imin]
#    print(me.fuzz[imin].counts)
#    fdamp = me.fuzz[imin].damp(dependent_value)
#    if fdamp > 0:
    me.fuzz[imin].add(dependent_value)
#    print(me.fuzz[imin].counts)
#    sys.stdout.flush()
# the products with hsign keep the +- straight.
# for the gradients that is.
#    learning_rate = learning_rate*fdamp
    for i in range(0, me.nvis): # over the row
      ef = alayer[i]*hsign*data[i]
      ep = ef*beta*hsign
      em = -ep
      fp = exp(-ep)
      fm = exp(-em)
      damp = (fp-fm)/(fp+fm)  *hsign  *data[i]
      hv = hsign *data[i]
      alayer[i] += learning_rate*( -hv + damp)
    return emin

# this is the online one pass algorithm
  def train_multi_fuzzy(me,data,beta,learning_rate,dependent_values, use_best = True):
    if len(me.mfuzz) == 0:
       print("You Must define multi fuzzy first to use this")
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
# select the row to train.
    imin = 0
    emin = me.energies[0]
    for i in range(1,me.nhid):
  #    print( emin, me.energies[i])
      if emin >= me.energies[i] :
         imin = i
         emin = me.energies[i]
#
# emin,imin now point to the best row
#    
    hsign = me.hidden[imin]
    alayer = me.layers[imin]
#    print(me.fuzz[imin].counts)
#    fdamp = me.fuzz[imin].damp(dependent_value)
#    if fdamp > 0:
    for i in range(0, me.how_many_fuzzy):
       me.mfuzz[i][imin].add(dependent_values[i])
#    print(me.fuzz[imin].counts)
#    sys.stdout.flush()
# the products with hsign keep the +- straight.
# for the gradients that is.
#    learning_rate = learning_rate*fdamp
    for i in range(0, me.nvis): # over the row
      ef = alayer[i]*hsign*data[i]
      ep = ef*beta*hsign
      em = -ep
      fp = exp(-ep)
      fm = exp(-em)
      damp = (fp-fm)/(fp+fm)  *hsign  *data[i]
      hv = hsign *data[i]
      alayer[i] += learning_rate*( -hv + damp)
    return emin

  def train(me,data,beta,learning_rate, use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
# select the row to train.
    imin = 0
    emin = me.energies[0]
    for i in range(1,me.nhid):
      if emin >= me.energies[i] :
         imin = i
         emin = me.energies[i]
#    print(emin)
#    sys.stdout.flush()
#
# emin,imin now point to the best row
#    
    hsign = me.hidden[imin]
    alayer = me.layers[imin]
# the products with hsign keep the +- straight.
# for the gradients that is.
    for i in range(0, me.nvis): # over the row
      ef = alayer[i]*hsign*data[i]
      ep = ef*beta*hsign
      em = -ep
      fp = exp(-ep)
      fm = exp(-em)
      damp = (fp-fm)/(fp+fm)  *hsign  *data[i]
      hv = hsign *data[i]
      alayer[i] += learning_rate*( -hv + damp)
    return emin


  def antitrain(me,data,beta,learning_rate,use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
# select the row to train.
    imax = 0
    emax = me.energies[0]
    for i in range(1,me.nhid):
      if emax <= me.energies[i] :
         imax = i
         emax = me.energies[i]
#
# emin,imin now point to the best row
#    
    hsign = me.hidden[imax]
    alayer = me.layers[imax]
# the products with hsign keep the +- straight.
# for the gradients that is.
    for i in range(0, me.nvis): # over the row
      ef = alayer[i]*hsign*data[i]
      ep = ef*beta*hsign
      em = -ep
      fp = exp(-ep)
      fm = exp(-em)
      damp = (fp-fm)/(fp+fm)  *hsign  *data[i]
      hv = hsign *data[i]
      alayer[i] -= learning_rate*( -hv + damp)

  def normalize(me, data, depend):
# data is a np array - to be used for training.
      m = np.max(data)
      mi = np.min(data)
#      if m < 1. and mi > -1.:
#          return (data, depend, 1., 0.)
      sh = np.mean(data)
      mi -= sh
      m -= sh
      if mi == m:  #behave sensibly
          m = mi + 1.
      retv = np.zeros(len(data))
      for i in range(0, len(data)):
          retv[i] = (data[i]-sh)/(m - mi)  
      return (retv, (depend -sh)/(m-mi), 1./(m -mi), sh)

  def denormalize(me, predict, scale, shift):
      return predict/scale+shift
      



def main():
   print("this is the main routine, set up for testing")
   my_rbm = rbm(2,2)
   print(my_rbm.layers)   
   d = np.full(2,1.)
   d[0] = -1.0
#   my_rbm.train(d, 0.1, 0.1)
#   print(my_rbm.layers)   
   for i in range(1,10):
     d[0] = 1.; d[1] = -1.
     my_rbm.train(d, 0.1, 0.1)
     d[0] = 1.; d[1] = 1.
     my_rbm.train(d, 0.1, 0.1)
     print(str(i)+' '+str(my_rbm.layers)  ) 

   d[0] = 0.
   print(my_rbm.reconstruct(d))
   d[0] = 1.
   d[1] = 0.
   print(my_rbm.reconstruct(d))

   my_rbm.layers[0] = np.array([-1.,1.])
   my_rbm.layers[1] = np.array([1.,1.])
   my_rbm.add_fuzzy(-1., 1., 20)
   print(my_rbm.layers)   
   d[0] = 1.; d[1] = -1.
   my_rbm.train_fuzzy(d, 0.1, 0.1, 0.4)
   d[0] = 1.; d[1] = 1.
   my_rbm.train_fuzzy(d, 0.1, 0.1, -0.4)
   print(my_rbm.layers)   
   print(d,my_rbm.estimate_EV(d))
   d[0] = 1.; d[1] = -1.
   print(d,my_rbm.estimate_EV(d))
#main()
# more tests
   n_rbm = rbm(2,3)
   print(n_rbm.layers)   
   n_rbm = rbm(3,2)
   print(n_rbm.layers)   
   d = np.full(3,1.)
   d[0] = -1.0
   d[1] =  2.0
   d[2] =  3.0
   for i in range(1,10):
     d[0] = -1.0
     d[1] =  2.0
     d[2] =  3.0
     n_rbm.train(d, 0.1, 0.1)
     d[0] =  1.0
     d[1] = -2.0
     d[2] =  3.0
     n_rbm.train(d, 0.1, 0.1)
     print(str(i)+' '+str(n_rbm.layers)  ) 
     print(str(i)+' '+str(n_rbm.hidden)  ) 
   d[0] = 0
   print(n_rbm.reconstruct(d))
   print(n_rbm.reconstruct_raw(d))
   x = n_rbm.reconstruct_raw(d)
   print( np.dot( x,d)/sqrt(np.dot(d,d)*np.dot(x,x)))
   print(d)
   d[0] = -1.0
   d[1] =  0.0
   d[2] =  3.0
   print(n_rbm.reconstruct_raw(d))
   x = n_rbm.reconstruct_raw(d)
   print( np.dot( x,d)/sqrt(np.dot(d,d)*np.dot(x,x)))
   print(d)
   d[0] = -1.0
   d[1] =  2.0
   d[2] =  3.0
   print(n_rbm.reconstruct_raw(d))
   x = n_rbm.reconstruct_raw(d)
   print( np.dot( x,d)/sqrt(np.dot(d,d)*np.dot(x,x)))
   print(d)
   q = n_rbm.normalize(d,1.)
   print( q)
   print( n_rbm.denormalize( q[1], q[2],q[3]))

if __name__ == "__main__":
    main()
