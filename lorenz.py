#!/usr/bin/python3
#
# simple program to make a lorenz function
#

import sys
# parameters are fixed
# other than the number of steps.

sigma = 10.
beta = 8./3
rho = 28.
# lorenz's own values

if len(sys.argv) < 2:
    nstep = 1000
else:
    nstep = int(sys.argv[1])

x = 1.
y = 0.
z = 0.
dt = 0.01
for i in range(0,10000):
    dxdt = sigma*(y -x)
    dydt = x*(rho -z) -y
    dzdt = x*y - beta *z
    x  += dxdt*dt
    y  += dydt*dt
    z  += dzdt*dt
#    sys.stdout.write( str(i)+','+str(x)+','+str(y)+','+str(z)+'\n')


for i in range(0,nstep):
    dxdt = sigma*(y -x)
    dydt = x*(rho -z) -y
    dzdt = x*y - beta *z
    x  += dxdt*dt
    y  += dydt*dt
    z  += dzdt*dt
    sys.stdout.write( str(i)+','+str(x)+','+str(y)+','+str(z)+'\n')


