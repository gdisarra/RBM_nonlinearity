import numpy as np
from scipy.special import erf
from scipy.special import erfinv
import math as m
import pdb

def sampling_functions(activation_function):
    if (str(activation_function)=='linear'):
        mean_zgivenv = lambda x: x
        sample_z = lambda x: x + np.random.normal(size=x.shape)
    elif (str(activation_function)=='ReLU'):
        mean_zgivenv = lambda x: mean_zgivenv_relu(x)
        sample_z = lambda x: sample_zgivenv_relu(x)
    elif (str(activation_function)=='step'):
        mean_zgivenv = lambda x: 1./(1+np.exp(-x))
        sample_z = lambda x: np.random.uniform(size=x.shape) < 1./(1+np.exp(-x))
    elif (str(activation_function)=='exponential'):
        mean_zgivenv = lambda x: np.exp(x)
        sample_z = lambda x: np.random.poisson( np.exp(x), size=x.shape)
    elif (str(activation_function)=='relu2'):
        num = lambda x: m.sqrt(2/m.pi)+np.multiply(np.multiply(x,np.exp(np.square(x)/2)),1+erf(x/m.sqrt(2)))
        den = lambda x: 1+ np.multiply(np.exp(np.square(x)/2),1+erf(x/m.sqrt(2)))
        mean_zgivenv = lambda x: np.divide(num(x),den(x))
        
        phi = lambda u,x: x + m.sqrt(2/m.pi)*erfinv(np.multiply(1+erf(x/m.sqrt(2))+np.exp(-np.square(x)/2),u)-erf(x/m.sqrt(2))-np.exp(-np.square(x)/2))
        p0 = lambda x: 1./(1+np.multiply(np.exp(np.square(x)/2),1+erf(x/m.sqrt(2))))
        l = lambda x: np.random.rand(x.shape[0],x.shape[1]) > p0(x)
        sample_z = lambda x: np.multiply(l(x),phi(p0(x)+np.multiply(1-p0(x),np.random.rand(x.shape[0],x.shape[1])),x)) 
        
    return sample_z, mean_zgivenv

def mean_zgivenv_relu(x):
    if not isinstance(x,np.ndarray):
        x=np.array([x])
    result = x + np.true_divide(m.sqrt(2/m.pi)*np.exp(-np.square(x)/2.),(1.+erf(x/m.sqrt(2))))
    a = x < -7
    result[a] = np.true_divide(-1.,x[a]) + np.true_divide(2.,np.power(x[a],3))
    return result

def cumulant_relu0(x, c):
    #pdb.set_trace()
    if x.size==1:
        x=np.array([x])
    if c.size==1:
        c=np.array([c])
    else:
        x=x[0,:]
        c=c[0,:]    
    result = np.square(x)/2. - np.multiply(x,c) + np.log(1+erf((x-c)/m.sqrt(2)))
    a0 = x-c < -8
    
    c_rep=c
    #pdb.set_trace()
    result[a0] = -np.log(c_rep[a0]-x[a0]) + np.log(1-np.true_divide(1.,np.power(c_rep[a0]-x[a0],2)) +3*np.true_divide(1.,np.power(c_rep[a0]-x[a0],4))-15*np.true_divide(1,np.power(c_rep[a0]-x[a0],6)))-np.square(c_rep[a0])/2 - 0.5*m.log(m.pi/2)
    
    return result

def cumulant_relu(x,c):
        
    if not isinstance(x,np.ndarray) or x.size==1:
        x=np.array([x])
        c=np.array([c])
        
    elif isinstance(x,np.ndarray):
        x=x[0,:]
        c=c[0,:]   
        
    result = np.square(x)/2. - np.multiply(x,c) + np.log(1+erf((x-c)/m.sqrt(2)))
    a0 = x-c < -8
    c_rep = c
    result[a0] = -np.log(c_rep[a0]-x[a0]) + np.log(1-np.true_divide(1.,np.power(c_rep[a0]-x[a0],2)) +3*np.true_divide(1.,np.power(c_rep[a0]-x[a0],4))-15*np.true_divide(1,np.power(c_rep[a0]-x[a0],6)))-np.square(c_rep[a0])/2 - 0.5*m.log(m.pi/2)
    
    return result

def cumulant_relu2(x,c):
    #pdb.set_trace()
    result = np.square(x)/2. - np.multiply(x,c) + np.log(1+erf((x-c)/m.sqrt(2)))
    a0 = x-c < -8
    c_rep=np.repeat(c,x.shape[0],axis=0)
    result[a0] = -np.log(c_rep[a0]-x[a0]) + np.log(1-np.true_divide(1.,np.power(c_rep[a0]-x[a0],2)) +3*np.true_divide(1.,np.power(c_rep[a0]-x[a0],4))-15*np.true_divide(1,np.power(c_rep[a0]-x[a0],6)))-np.square(c_rep[a0])/2 - 0.5*m.log(m.pi/2)
    
    return result

def sample_zgivenv_relu(x):
    if not isinstance(x,np.ndarray):
        x=np.array([x])
    result = x + (m.sqrt(2) * erfinv(np.multiply(1+erf(x/m.sqrt(2)),np.random.uniform(size=x.shape))-erf(x/m.sqrt(2))))
    #exceptions here the approximation which applies generally for x<-6
    a = np.isinf(result)
    stand_dev_relu = lambda x: 1 - np.true_divide( np.multiply( 2-np.square(x) , 2-np.square(x)-np.power(x,4))  , np.power(x,6) )
    result[a] =  abs(mean_zgivenv_relu(x[a])) + np.multiply(stand_dev_relu(x[a]),np.random.normal(size=x[a].shape))
    return result
