import numpy as np
import matplotlib.pyplot as plt
import random

data = np.genfromtxt( 'notas_andes.dat' )
subject = [ "Nota Final Física 1",  "Nota Final Física 2",  "Nota Final Algebra Lineal",  "Nota Final Cálculo Diferencial",  "Promedio Final PGA"]

Y = data[:,4]
X = np.transpose( data[:,:4] )

def likelihood(y_obs, y_model):
    chi_squared = (1.0/2.0)*sum((y_obs-y_model)**2)
    return np.exp(-chi_squared)

def linear_model(x_arr, beta_arr):
    x1, x2, x3, x4 = x_arr
    b0, b1, b2, b3, b4 = beta_arr
    return b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4

beta_walk = []
l_walk = []


beta_walk.append( np.random.rand(5) )
y_init = linear_model(X, beta_walk[0])
l_walk.append(likelihood(Y, y_init))

n_iterations = 20000 #this is the number of iterations I want to make

for i in range(n_iterations):
    beta_prime = np.random.normal(loc=beta_walk[i] , scale = 0.1 )
    
    y_init = linear_model(X, beta_walk[i])
    y_prime = linear_model(X, beta_prime)
    
    l_prime = likelihood(Y, y_prime)
    l_init = likelihood(Y, y_init)
    
    if l_init !=0:
        alpha = l_prime/l_init
        if(alpha>=1.0):
            beta_walk.append(beta_prime)
            l_walk.append(l_prime)
        else:
            beta = random.random()
            if(beta<=alpha):
                beta_walk.append(beta_prime)
                l_walk.append(l_prime)
            else:
                beta_walk.append(beta_walk[i])
                l_walk.append(l_walk[i])
                
beta_sample = np.array(beta_walk)[10000:]

plt.figure(figsize=(15,10))


for b in range(5):
    plt.subplot(2,3,b+1)
    plt.hist( np.array(beta_walk)[1000:,b] , bins = 20 , density = True )
    plt.title( r"$\beta_{}= {:.2f}\pm {:.2f}$".format(b, np.mean(beta_sample[:,b]), np.std( beta_sample[:,b] ) ))

plt.subplots_adjust(hspace=.5) # ajustar el espacio horizontal entre ellas
plt.savefig( 'ajuste_bayes_mcmc.png' )