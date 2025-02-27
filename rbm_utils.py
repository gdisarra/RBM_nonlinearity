#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:35:05 2024

@author: giovannidisarra
"""

import numpy as np
import matplotlib.pyplot as pl
import math
from scipy.special import erf
from sampling_functions import mean_zgivenv_relu

import warnings
warnings.filterwarnings("ignore")

def decomp(states,k,N):
    dec_s=states>0.
    s_= np.array([int(d) for d in np.binary_repr(k+1,width=N)])
    dec_s[dec_s!=0]=s_
    return dec_s


class Lattice_Gas:
    
    def __init__(self, v, N, h, J, T, if_plot=True, tit=''):
        
        """
        Initialize the ground truth model.
        
        Args:
            v (array): units configurations
            N (int): number of units
            h (float): fields of the ground truth model.
            J (float): couplings of the ground truth model.
            T (float): three body interactions of the ground truth model.
        """
        
        self.N = v.shape[1]
        
        Ns = [h, J, T]
        s = np.sum(v, axis = 1)
        
        op_truth = np.zeros(v.shape[0])
        
        for order in range(min(N,3)):
            op_truth[np.where(s == order+1)[0]] = np.random.normal(loc = Ns[order], scale = Ns[order]/2, size = len(np.where(s == order+1)[0]) )
            
        self.interactions = op_truth
        
        self.hs = op_truth[np.where(np.sum(v, axis = 1)==1)]
        self.Js = op_truth[np.where(np.sum(v, axis = 1)==2)]
        self.Ts = op_truth[np.where(np.sum(v, axis = 1)==3)]
        
        print(vars(self))
        
        if if_plot:
            
            orde_2m_truth = np.zeros((N))
            orde_truth = np.zeros((N))
            
            for s in range(N):
                orde_2m_truth[s] = np.sqrt(np.mean(np.square(op_truth[np.where(np.sum(v, axis=1) == s+1)])))
                orde_truth[s] = np.mean(op_truth[np.where(np.sum(v, axis=1) == s+1)])
            
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title(tit, fontsize = 60)
            ax.scatter(np.sum(v, axis=1), op_truth, s = 600)
            ax.scatter(np.arange(1,N+1, 1), orde_truth, s = 600, color ='red', label = r'$\overline{ I^{(s)}} $')
            ax.scatter(np.arange(1,N+1, 1), orde_2m_truth, s = 600, color ='green', label= r'$\sqrt{\overline{ I^{(s)^{2}}} }$')
            ax.set_xlabel('s', fontsize = 60)
            ax.set_ylabel(r'$I^{(s)}$', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            pl.legend(fontsize = 50)
            pl.show()
            
    def compute_Q(self, v):
        
        N = self.N
        
        orde_2m = np.zeros((N))
        orde = np.zeros((N))
        
        for s in range(N):
            orde_2m[s] = np.sqrt(np.mean(np.square(self.interactions[np.where(np.sum(v, axis=1) == s+1)])))
            orde[s] = np.mean(self.interactions[np.where(np.sum(v, axis=1) == s+1)])

        orde_2m = np.where(orde_2m < 1e-10, 1e-10, orde_2m)        

        Qs = np.array([ orde_2m[s+1]/orde_2m[s] for s in range(N-1)])
        Q0s = np.array([ orde[s+1]/orde[s] for s in range(N-1)])
        
        return Q0s, Qs
        
        
    def compute_Z_llh(self, v, if_plot = True, tit=''):
        
        '''
        Compute partition function and log-likelihood of the lattice gas model
        
        Parameters
        ----------
        v: visible configurations
    
        Returns
        -------
        Z_d : partition function of data distribution
        H_d : energy of the single states
        llh : log-likelihood <ln p_lg>_p_lg
        
        '''
        
        v_str = np.array([str(vi) for vi in v])
        H_d = np.zeros((v.shape[0],1))
        for i, vi in enumerate(v): 
            
            Ns = int(sum(vi))
            for k in range(int(2**Ns-1)):
                S_ = np.array(decomp(vi, k, Ns),  dtype = int)
                #S_ = decomp(vi, k, Ns)
                
                H_d[i] = H_d[i] - self.interactions[np.where(v_str == str(S_))[0][0]]
        
        Z_d = np.sum(np.exp(-H_d))
        
        llh = - np.sum(H_d * np.exp(-H_d))/Z_d - np.log(Z_d)
        
        if if_plot:
            
            x = np.exp(-H_d)/Z_d
            
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title(r''+tit+'\n $< \ln \; p_{lg} >_{p_{lg}}=$'+str(np.round(llh, decimals=3)), fontsize = 60)
            ax.plot(x)
            ax.set_xlabel('states', fontsize = 60)
            ax.set_ylabel('state probability', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            #pl.yscale('log')
            pl.show()
        
        return H_d, Z_d, llh
    
    
    
class RBM:
    
    def __init__(self, N, M, gt_std, activation_function, hidden_bias = True, init = True, noise = 0.):
        
        """
        Initialize the RBM with the given number of visible and hidden units.
        
        Args:
            N (int): Number of visible units.
            M (int): Number of hidden units.
            activation_function (str): activation function of the hidden nodes
            hidden_bias (bool): define c as the hidden biases. If False, hidden biases are set to 0 and excluded from training.
            init (bool): initialize the network. If False, the parameters can be arbitrarily defined.
            noise (float): standard deviation of the gaussian noise added to the ground truth initialization.
        """
        
        self.N = N
        self.M = M
        
        if init:
            std = 0.01
            
            # Initialize weights and biases
            self.W = np.random.normal(0, std, size=(N,M))
            self.b = np.random.normal(0, std, size=(1,N))
            self.c = np.random.normal(0, std, size=(1,M))
            
            if not hidden_bias:
                self.c = np.zeros((1,M))
                
        else:
            
            # Initialize weights and biases
            self.W = np.random.normal(0., gt_std, size=(N,M))
            self.b = np.random.normal(0., gt_std, size=(1,N))
            self.c = np.random.normal(0., gt_std, size=(1,M))
            
            if not hidden_bias:
                self.c = np.zeros((1,M))
            
        self.activation_function = activation_function
        
        print(vars(self))
        
        
    def perturb_ic(self, standard_initialization, rbm_gt, activation_function, hidden_bias=True, noise=0.):
        """
        Perturb the ground truth model with gaussian noise and set it as initial conditions for training.
        
        Args:
            rbm_gt (class): ground truth rbm.
            activation_function (str): activation function of the hidden nodes
            hidden_bias (bool): define c as the hidden biases. If False, hidden biases are set to 0 and excluded from training.
            noise (float): standard deviation of the gaussian noise added to the ground truth initialization.
        
        """
        
        self.N = rbm_gt.W.shape[0]
        self.M = rbm_gt.W.shape[1]
        
        self.W = rbm_gt.W + np.random.normal(0, noise, size = rbm_gt.W.shape)
        self.b = rbm_gt.b + np.random.normal(0, noise, size = rbm_gt.b.shape)
        self.c = rbm_gt.c + np.random.normal(0, noise, size = rbm_gt.c.shape)
        
        if not hidden_bias:
            self.c = np.zeros((1,self.M))
            
        self.activation_function = activation_function
        
        if standard_initialization:
            std = 0.01
            
            # Initialize weights and biases
            self.W = np.random.normal(0, std, size=(self.N,self.M))
            self.b = np.random.normal(0, std, size=(1,self.N))
            self.c = np.random.normal(0, std, size=(1,self.M))
            
            if not hidden_bias:
                self.c = np.zeros((1,self.M))
            
        
        print(vars(self))
        
            
            
            
    def compute_Z_llh(self, v, H_d, Z_d, rbm_gt, ground_truth = 'lg', if_plot = True, tit=''):
        
        '''
        Compute partition function and log-likelihood of the rbm with respect to lattice gas or generative rbm distribution.
        
        Parameters
        ----------
        v: visible configurations
        H_d : energy of the single states for the lattice gas model
        Z_d : partition function of the lattice gas model
        rbm_gt : ground truth rbm
        ground_truth: can be set to 'lg' for lattice gas and 'rbm' for generative rbm
        
        Returns
        -------
        S : energy of the single states for the rbm 
        Z_RBM : partition function of the rbm
        llh : log-likelihood <ln p_rbm>_p_gt
        
        '''
        
        W = self.W
        b = self.b
        c = self.c
        
        activation_function = self.activation_function
        
        if (str(activation_function)=='linear'):
            K = lambda x,c: np.square(x)/2 -np.multiply(x,c)
        elif (str(activation_function)=='relu'):
            #K = lambda x,c: cumulant_relu(x,c)
            K = lambda x,c: np.square(x)/2. - np.multiply(x,c) + np.log((1+erf((x-c)/math.sqrt(2)))/(1-erf(c/math.sqrt(2))))
        elif (str(activation_function)=='step'):
            K = lambda x,c: np.log(1.+np.exp(np.subtract(x,c)))-np.log(1+np.exp(-c))
        elif (str(activation_function)=='exponential'):
            K = lambda x,c: np.multiply(np.exp(-c),(np.exp(x)-1))
        
        bv = np.dot(b, v.T)
        i_v = np.dot(v, W)
        S = - bv - np.sum(K(i_v, c).T, axis = 0)

        Z_RBM = np.sum(np.exp(-S))

        if ground_truth == 'lg':
            
            #compute exact log-likelihood of the RBM parameters with respect to the data distribution
            llh = np.dot(np.exp(-H_d.T), -S.T)/Z_d - np.log(Z_RBM)
            
            if if_plot:
                
                y = np.exp(-S)/Z_RBM
                x = np.exp(-H_d)/Z_d
                
                
        elif ground_truth == 'rbm':
            
            W_gt = rbm_gt.W
            b_gt = rbm_gt.b
            c_gt = rbm_gt.c
            
            bv_gt = np.dot(b_gt, v.T)
            i_v_gt = np.dot(v, W_gt)
            S_gt = - bv_gt - np.sum(K(i_v_gt, c_gt).T, axis = 0)
            Z_gt = np.sum(np.exp(-S_gt))
            
            llh = np.dot(np.exp(-S_gt), -S.T)/Z_gt - np.log(Z_RBM)
            
            if if_plot:
                

                y = np.exp(-S)/Z_RBM
                x = np.exp(-S_gt)/Z_gt
        
        if if_plot:
            
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title(tit, fontsize = 60)
            ax.set_title(r''+tit+'\n $< \ln \; p_{rbm} >_{p_{gt}}=$'+str(np.round(llh[0,0], decimals=3)), fontsize = 60)
            ax.scatter(x, y, s = 600)
            
            #diag = np.arange(0, max(np.max(x),np.max(y))+max(np.max(x),np.max(y))/200, max(np.max(x),np.max(y))/200)
            #ax.plot(diag, diag)
            ax.set_ylabel('RBM state probability', fontsize = 60)
            ax.set_xlabel('data probability', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            pl.show()
            
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title(tit, fontsize = 60)
            ax.set_title(r''+tit+'\n $< \ln \; p_{rbm} >_{p_{gt}}=$'+str(np.round(llh[0,0], decimals=3)), fontsize = 60)
            ax.scatter(x, y, s = 20)
            #diag = np.arange(0, max(np.max(x),np.max(y))+max(np.max(x),np.max(y))/200, max(np.max(x),np.max(y))/200)
            #ax.plot(diag, diag)
            ax.set_ylabel('RBM state probability', fontsize = 60)
            ax.set_xlabel('data probability', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.yscale('log')
            pl.xscale('log')
            pl.grid(True)
            pl.show()
                    
            
        return S, Z_RBM, llh[0,0]
        

    def mapping(self, if_plot=True, tit=''):
        
        '''
        Implementation of exact mapping (eq. 3.1) from https://doi.org/10.1162/neco_a_01420

        Returns
        -------
        op : interactions of the mapped rbm

        '''
        
        N = self.N
        M = self.M
        
        W = self.W
        b = self.b
        c = self.c
        
        activation_function = self.activation_function

        if (str(activation_function)=='linear'):
            K = lambda x,c: np.square(x)/2 -np.multiply(x,c)
        elif (str(activation_function)=='relu'):
            K = lambda x,c: np.square(x)/2. - np.multiply(x,c) + np.log(1+erf((x-c)/math.sqrt(2)))
            #K = lambda x,c: cumulant_relu(x,c)
        elif (str(activation_function)=='step'):
            K = lambda x,c: np.log(1.+np.exp(np.subtract(x,c)))-np.log(1.+np.exp(-c))
        elif (str(activation_function)=='exponential'):
            K = lambda x,c: np.multiply(np.exp(-c),(np.exp(x)-1))
            
        #Implementing eq.13, 14
        op = np.zeros(2**N)
        Sop = np.zeros((2**N,N),dtype=int)

        for i in range(1,2**N):
            Sop[i,:]=np.array([bool(int(d)) for d in np.binary_repr(int(i),width=N)])
            Ns = int(sum(Sop[i,:]))
        
            for k in range(int(2**Ns-1)):
                #find primitive states from state Sop
                
                S_=decomp(Sop[i,:],k, Ns)
                
                #compute interaction term from RBM architecture for every state
                op[i] = op[i] + np.sum(K(np.sum(W[S_,:],axis=0).reshape(1,M),c))*(-1)**(Ns-sum(S_))
            
            if Ns==1:
                op[i] = op[i]+b[:,S_]
        
        #pdb.set_trace()
        
        if if_plot:
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title(tit, fontsize = 60)
            ax.scatter(np.sum(Sop, axis=1), op, s = 600)
            ax.set_xlabel('s', fontsize = 60)
            ax.set_ylabel(r'$I^{(s)}$', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            pl.show()
            
            s = np.sum(Sop, axis = 1)
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title('Interactions distribution', fontsize = 60)
            for order in range(N):
                ax.hist(op[np.where(s == order+1)[0]], bins = int(2*N /(order+1)), label = 's='+str(order+1), alpha = 0.5)
            ax.set_xlabel(r'$I^{(s)}$', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            pl.legend(fontsize = 50 )
            pl.show()
            
        #return Sop, op        
        return op
    
    
    def exact_train(self, maxepoch, v, l_rate, H_d, Z_d, rbm_gt, ground_truth, hidden_bias=True, if_plot = True):
        
        '''
        Train rbm with exact maximization of the log-likelihood with respect to a lattice gas distribution or a generative rbm distribution
        
        Args
            maxepoch (int) : maximum number of training steps.
            v (array) : visible configurations
            l_rate (float) : learning rate
            H_d (array) : energy of the single states for the lattice gas model
            Z_d (float) : partition function of the lattice gas model
            rbm_gt (class) : ground truth rbm
            ground_truth (str): can be set to 'lg' for lattice gas and 'rbm' for generative rbm
            hidden_bias (bool): if False, hidden biases are set to 0 and excluded from training.
            
        '''
        
        
        lh = []
        grads = {}
        epoch = 0
        
        activation_function = self.activation_function
        
        llh_gt = - np.sum(H_d * np.exp(-H_d))/Z_d - np.log(Z_d)
            
        if (str(activation_function)=='linear'):
            K = lambda x,c: np.square(x)/2 -np.multiply(x,c)
            K1 = lambda x: x
            K2 = lambda x: x*0+1
        elif (str(activation_function)=='relu'):
            #K = lambda x,c: cumulant_relu(x,c)
            K = lambda x,c: np.square(x)/2. - np.multiply(x,c) + np.log((1+erf((x-c)/math.sqrt(2)))/(1-erf(c/math.sqrt(2))))
            K1 = lambda x: x + math.sqrt(2/math.pi) * np.true_divide(np.exp(-x**2/2),(1+erf(x/math.sqrt(2))))
            K1 = lambda x: mean_zgivenv_relu(x)
            K2 = lambda x: 1 - np.multiply((K1(x)-x), K1(x))
        elif (str(activation_function)=='step'):
            K = lambda x,c: np.log(1.+np.exp(np.subtract(x,c)))-np.log(1+np.exp(-c))
            K1 = lambda x: np.exp(x)/(1+np.exp(x))
            K2 = lambda x: np.exp(x)/np.square(1+np.exp(x))
        elif (str(activation_function)=='exponential'):
            K = lambda x,c: np.multiply(np.exp(-c),(np.exp(x)-1))
            K1 = lambda x: np.exp(x)
            K2 = lambda x: np.exp(x)
        
        if if_plot:
            pl.ion()
            fig1, axs = pl.subplots(nrows=4,ncols=2,figsize=(24,24))
            
        while epoch <= maxepoch:

            W = self.W
            b = self.b
            c = self.c
            
            #compute positive term of the log-likelihood gradient
            bv = np.dot(b, v.T)
            
            i_v = np.dot(v, W)-c
            
            z = K1(i_v)
            
            if ground_truth == 'lg':
                pos_c = np.dot(np.exp(-H_d.T), z) / Z_d
                #print('pos_c', pos_c)
                
                pos_b = np.dot(np.exp(-H_d.T), v) / Z_d
                
                arg = v * np.array([np.sum(K1(i_v).T, axis=0)]).T
                
                pos_W = np.dot( np.exp(-H_d.T), arg).T /Z_d
                
            elif ground_truth == 'rbm':
                
                W_gt = rbm_gt.W
                b_gt = rbm_gt.b
                c_gt = rbm_gt.c
                
                bv_gt = np.dot(b_gt, v.T)
                
                i_v_gt = np.dot(v, W_gt)-c_gt
                
                z_gt = K1(i_v_gt)
                
                S_gt, Z_gt, llh_gt = rbm_gt.compute_Z_llh(v, H_d, Z_d, rbm_gt, ground_truth, if_plot = False)
                
                pos_c = np.dot(np.exp(-S_gt), z_gt) / Z_gt
                #print('pos_c', pos_c)
                
                pos_b = np.dot(np.exp(-S_gt), v) / Z_gt
                
                arg = v * np.array([np.sum(K1(i_v).T, axis=0)]).T
                
                pos_W = np.dot( np.exp(-S_gt), arg).T /Z_gt
            
            #partition function and negative term of log-likelihood gradient
            S = - bv - np.sum(K(i_v, c).T, axis = 0)
            Z = np.sum(np.exp(-S))
            
            neg_b = np.dot(np.exp(-S), v) / Z
            
            neg_c = np.dot(np.exp(-S), z) / Z
            #print('neg_c',neg_c)
            
            neg_W = np.dot( np.exp(-S), arg).T / Z
            
            #compute gradients
            grad_W = pos_W - neg_W
            grad_b = pos_b - neg_b
            grad_c = pos_c - neg_c
            
            if not hidden_bias:
                grad_c = grad_c*0.
            
            
            #print('grad_W', grad_W)
            #print('grad_b', grad_b)
            #print('grad_c', grad_c)
            
            #set the number of recorded steps based on length of training
            if (epoch % 10 +1):
                
                #self.mapping()
                
                _, Z_RBM, llh_scat = self.compute_Z_llh(v, H_d, Z_d, rbm_gt, ground_truth, if_plot =False)
                
                lh.append(llh_scat)
                
                if if_plot:
                    
                    #plot the parameters evolution during training for each training epoch
                    #axis0.scatter(np.sum(Sop,axis=1),op)
                    axs[0,0].set_title('parameters update', fontsize =40)
                    axs[0,1].set_title('parameters update', fontsize =40)
                    axs[0,0].scatter(np.zeros((b.shape)).flatten()+epoch, b.flatten(), s=20, color = 'black')
                    axs[0,0].set_ylabel('b', fontsize = 36)
                    axs[0,0].tick_params(labelsize=32)
                    axs[1,0].scatter(np.zeros((W.shape)).flatten()+epoch, W.flatten(), s=20, color = 'blue')
                    axs[1,0].set_ylabel('W', fontsize = 36)
                    axs[1,0].tick_params(labelsize=32)
                    axs[2,0].scatter(np.zeros((c.shape))+epoch, c.flatten(), s=20, color = 'red')
                    axs[2,0].set_ylabel('c', fontsize = 36)
                    axs[2,0].tick_params(labelsize=32)
                    axs[3,0].scatter(epoch, llh_scat, s=20, color = 'lime')
                    axs[3,0].hlines(llh_gt, 0, maxepoch, ls ='dashed', color = 'black')
                    axs[3,0].set_ylabel('log-likelihood', fontsize = 36)
                    axs[3,0].tick_params(labelsize=32)
                    axs[3,0].set_xlabel('epoch', fontsize = 36)
                    axs[0,1].scatter(np.zeros((grad_b.shape)).flatten()+epoch, grad_b.flatten(), s=20, color = 'black')
                    axs[0,1].hlines(0., 0,maxepoch, ls ='dashed', lw=3, color = 'black')
                    axs[0,1].set_ylabel(' grad b', fontsize = 36)
                    axs[0,1].tick_params(labelsize=32)
                    axs[1,1].scatter(np.zeros((grad_W.shape)).flatten()+epoch, grad_W.flatten(), s=20, color = 'blue')
                    axs[1,1].set_ylabel('grad W', fontsize = 36)
                    axs[1,1].tick_params(labelsize=32)
                    axs[1,1].hlines(0., 0,maxepoch, ls ='dashed', lw=3, color = 'black')
                    axs[2,1].scatter(np.zeros((grad_c.shape))+epoch, grad_c.flatten(), s=20, color = 'red')
                    axs[2,1].set_ylabel('grad c', fontsize = 36)
                    axs[2,1].tick_params(labelsize=32)
                    axs[2,1].hlines(0., 0,maxepoch, ls ='dashed', lw=3, color = 'black')
                    if epoch> int(maxepoch - maxepoch/10):
                        axs[3,1].scatter(epoch, llh_scat, s=20, color = 'lime')
                        axs[3,1].hlines(llh_gt, epoch, maxepoch, ls ='dashed', color = 'black')
                        axs[3,1].set_ylabel('log-likelihood', fontsize = 36)
                        axs[3,1].tick_params(labelsize=32)
                        axs[3,1].set_xlabel('epoch', fontsize = 36)
# =============================================================================
#                     axs[3,1].scatter(epoch, Z_RBM, s=20, color = 'lime')
#                     axs[3,1].set_ylabel('Z RBM', fontsize = 36)
#                     axs[3,1].tick_params(labelsize=32)
#                     axs[3,1].set_xlabel('epoch', fontsize = 36)
#                     axs[3,1].set_yscale('log')
# =============================================================================
                    pl.ioff()
                
            grads["DW"] = l_rate * grad_W
            grads["Db"] = l_rate * grad_b*0.1
            grads["Dc"] = l_rate * grad_c
            
            self.W = W + grads["DW"]
            self.b = b + grads["Db"]
            self.c = c + grads["Dc"]
                
            epoch += 1
        
        
        pl.show()
        
    def compute_Q(self, op, v):
        
        N = v.shape[1]
        
        orde_2m = np.zeros((N))
        orde = np.zeros((N))
        
        for s in range(N):
            orde_2m[s] = np.sqrt(np.mean(np.square(op[np.where(np.sum(v, axis=1) == s+1)])))
            orde[s] = np.mean(op[np.where(np.sum(v, axis=1) == s+1)])
        
        orde_2m = np.where(orde_2m < 1e-10, 1e-10, orde_2m)
        
        Qs = np.array([ orde_2m[s+1]/orde_2m[s] for s in range(N-1)])
        Q0s = np.array([ orde[s+1]/orde[s] for s in range(N-1)])
        
        return Q0s, Qs
    
    
    
def ground_truth_training(std_init, ground_truth, gt_std, h, J, T, noises, N, M, l_rate, maxepoch, activation_functions, cc):
    
    rbm = {}
    plots = False
    ic = False
    
    v = np.zeros((2**N, N), dtype =int)
    for i in range(2**N):
        v[i] = np.array([int(d) for d in np.binary_repr(int(i),width=N)], dtype =int).reshape(1,N)
    
    if ground_truth == 'lg':
        plots = True
        ic = True
    
    lg = Lattice_Gas(v, N, h, J, T, tit = 'Lattice gas model', if_plot = plots) 
    Q0s_gt, Qs_gt = lg.compute_Q(v)
    H_d, Z_d, llh_lg = lg.compute_Z_llh( v, tit = 'State probabilities', if_plot = plots)
    
    for activation in activation_functions:
        
        #define an RBM
        rbm[activation] = RBM( N, M, gt_std, activation, hidden_bias = cc, init = True)
        rbm_gt = RBM( N, M, gt_std, activation, hidden_bias = cc, init = ic)
        
        #define generative model (random rbm or lattice gas) and compute Z and max llh 
        if ground_truth == 'rbm':
            S_gt, Z_gt, llh_gt = rbm_gt.compute_Z_llh(v, H_d, Z_d, rbm_gt, ground_truth, if_plot = False)
            
            Q0s_gt, Qs_gt = rbm_gt.compute_Q(rbm_gt.mapping(tit='Ground truth RBM mapping'), v)
        
        for noise in noises:
            #initialize and compute Z and llh for initial rbm
            rbm[activation].perturb_ic(std_init, rbm_gt, activation, hidden_bias = cc, noise = noise)
            
            _, Z_init, llh_init = rbm[activation].compute_Z_llh(v, H_d, Z_d, rbm_gt, ground_truth, tit = 'Probabilities comparison: initial vs data')
        
            #compute interaction model of the initial RBM
            rbm[activation].mapping(tit = 'Initial RBM mapping')
            
            #train RBM with exact log-likelihood maximization
            rbm[activation].exact_train( maxepoch, v, l_rate, H_d, Z_d, rbm_gt, ground_truth, hidden_bias = cc)
            #map the trained RBM
            I = rbm[activation].mapping(tit = 'Trained RBM mapping')
            _, Z_final, llh_final = rbm[activation].compute_Z_llh(v, H_d, Z_d, rbm_gt, ground_truth, tit = 'Probabilities comparison: trained vs data')
            
            Q0s_final, Qs_final = rbm[activation].compute_Q(I,v)
            
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title('Decayness '+str(activation), fontsize = 60)
            ax.scatter(np.arange(1,N,1), Qs_final, s = 600)
            ax.set_ylabel(r'$\sqrt{\overline{ I^{(s+1)^{2}}} } / \sqrt{\overline{ I^{(s)^{2}}} }$', fontsize = 60)
            ax.set_xlabel('s', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            pl.show()
            
            fig, ax = pl.subplots(figsize=(16,16))
            ax.set_title('Decayness '+str(activation), fontsize = 60)
            ax.scatter(Qs_gt, Qs_final, s = 600)
            for i in range(N-1):
                ax.scatter(Qs_gt[i], Qs_final[i], s = 600, label = 's='+str(i+1))
            ax.plot(np.arange(0, np.nanmax(Qs_gt)+0.1, np.nanmax(Qs_gt)/100), np.arange(0, np.nanmax(Qs_gt)+0.1, np.nanmax(Qs_gt)/100))
            ax.set_ylabel(r'$\sqrt{\overline{ I^{(s+1)^{2}}} } / \sqrt{\overline{ I^{(s)^{2}}} }$', fontsize = 60)
            ax.set_xlabel(r'$\sqrt{\overline{ I^{(s+1)^{2}}} } / \sqrt{\overline{ I^{(s)^{2}}} }$ ground truth', fontsize = 60)
            ax.tick_params(labelsize=50)
            pl.grid(True)
            pl.legend(fontsize = 50)
            pl.show()
    
    if ground_truth == 'lg': 
        out = [lg, H_d, Z_d, llh_lg, Q0s_gt, Qs_gt, Z_init, llh_init, I, Z_final, llh_final, Q0s_final, Qs_final, rbm]
    if ground_truth == 'rbm': 
        out = [rbm_gt, S_gt, Z_gt, llh_gt, Q0s_gt, Qs_gt, Z_init, llh_init, I, Z_final, llh_final, Q0s_final, Qs_final, rbm]
        
    return out
