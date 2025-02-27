#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:33:55 2024

@author: giovannidisarra
"""

import matplotlib.pyplot as pl
import numpy as np
from rbm_utils import ground_truth_training

pl.rcParams['font.family'] = 'serif'
pl.rcParams['font.serif'] = ['Times New Roman']
pl.rcParams['mathtext.fontset'] = 'custom'
pl.rcParams['mathtext.rm'] = 'Times New Roman'
pl.rcParams['mathtext.it'] = 'Times New Roman:italic'
pl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

#rbm parameters

N = 4
M = 5

cc = False

activation_functions = ['linear','step', 'relu', 'exponential']
activation_functions = ['exponential']

std_init = False

#ground truth parameters

ground_truth = 'rbm'

#ground_truth = 'lg'

#rbm
gt_std = 0.1

#lg

h = 0.
J = 0.
T = 0.03 # 1.84 / np.sqrt(N) / ( N / 1.5 )
T = 0.

hs = np.arange(0.,10.,1.) + 1.
#Js = [0,...,0.5]
Js = np.arange(0., 0.5, 0.05) + 0.05
#Ts = [0,...,0.1]
Ts = np.arange(0.,0.1, 0.01 ) + 0.01

#triplets
# hs = [0., 1., 2., 3.]
# Js = [0., 0.05, 0.1, 0.15]
# Ts = [0., 0.01, 0.02, 0.03]

#training parameters

l_rate = 1e-3

maxepoch = int(1e3)

noises = [0., 0.1, 0.2, 0.5]


        
if __name__ == '__main__':
    
    for T in Ts:
        
        #standard training from lattice gas
        config_lg = [True, 'lg', 0., h, J, T, [0.], N, M, l_rate, maxepoch, activation_functions, cc]

        #standard training from random rbm
        config_rbm_gt = [True, 'rbm', gt_std, 0., 0., 0., [0.], N, M, l_rate, maxepoch, activation_functions, cc]

        #ground truth perturbation training 
        config_rbm_gt = [False, 'rbm', gt_std, 0., 0., 0., noises, N, M, l_rate, maxepoch, activation_functions, cc]

    
        std_init, ground_truth, gt_std, h, J, T, noises, N, M, l_rate, maxepoch, activation_functions, cc = config_rbm_gt
        
        
        out = ground_truth_training(std_init, ground_truth, gt_std, h, J, T, noises, N, M, l_rate, maxepoch, activation_functions, cc)
    
        gt_model, H_gt, Z_gt, llh_gt, Q0s_gt, Qs_gt, Z_init, llh_init, I, Z_final, llh_final, Q0s_final, Qs_final, rbm = out
    
    
