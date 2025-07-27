# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Last edition on on Sun Nov 21 19:27:54 2021
@author: Pablo Merchán-Rivera

Evaluate DREAM results 
"""


# %% Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dream_init import my_seed, numParams, samples, rep, flag, param_distros
import f_color as fc
from f_maps import params_names_all_s1, params_names_short_s1
from scipy.stats import skew, kurtosis


# %% Evaluate results

if __name__ == "__main__":  
    
    convEvals = 5000
        
    # sampling settings
    np.random.seed(my_seed) 
    
    # numSamples, numParams, samples, rep, flag = di.numSamples, di.numParams, di.samples, di.rep, di.flag
    
    # import DREAM results
    data_results = pd.read_csv('dream_FLOOD.csv')
    
    results_array = np.zeros((len(data_results), numParams))
    for i in range(numParams):
        results_array[:,i] = data_results.iloc[:,i+1]
        
    likelihood = np.array(data_results.iloc[:,0])

        
    # extract estimated parameters
    params_new_all = data_results.to_numpy()[:,1:numParams+1]
    params_new_con = params_new_all[-convEvals:len(params_new_all),:]
    
    # extract output of simulations
    sims_all = data_results.to_numpy()[:,numParams+1:-1].reshape(10000,4,300)
    
    
    # %% Compute statistics
    params_new_all_mean = np.mean(params_new_all, axis=0).round(4)
    params_new_all_std =  np.std(params_new_all, axis=0).round(4)
    params_new_all_skw =  skew(params_new_all, axis=0).round(4)
    params_new_all_kur =  kurtosis(params_new_all, axis=0).round(4)
    
    best_set_loc = data_results['like1'].idxmax(axis=0)
    best_set = data_results.iloc[best_set_loc,1:12].round(4)
    print(best_set)

    
    # %% Plot prior parameter uncertainty
    
    # plot parameters    
    bars = 15
    coord = np.linspace(1,rep,rep)
    alpha = 0.60  
    
    for i in range(numParams):
        plt.figure('histogram after convergence' +  params_names_short_s1[i], figsize=(3,3))
        plt.hist(samples[0:convEvals,i], bars, alpha=alpha+0.2, color=fc.orangecad, label="Prior distribution", 
                  range=(param_distros[i].minbound,param_distros[i].maxbound), density=False, stacked=True)
        plt.hist(params_new_con[:,i], bars, alpha=alpha, color=fc.bluesaph, label="Posterior distribution", 
                  range=(param_distros[i].minbound,param_distros[i].maxbound), density=False, stacked=True)
        plt.ylabel('Frequency', fontsize=12)
     
        plt.xlim(np.min(results_array[:,i]), np.max(results_array[:,i]))

        plt.title('Histogram - ' + params_names_all_s1[i])
        plt.legend(edgecolor='black', fancybox=False, fontsize=8.5,
                    borderpad=0.8, handletextpad=0.9, labelspacing=0.65)
        
        
        
    # %% Uncertainty
    
    # import observations
    observations = np.loadtxt('./obs/obs_append_300sp.txt').reshape(4,300)
    
    # import prior
    rule = 'random'
    sim_from = [0, 250, 500, 750]
    sim_to = [250, 500, 750, 1000]

    num_prior_sims = sim_to[-1]

    sims_prior = ['None'] * len(sim_from)
    
    for i in range(len(sim_from)):
        sims_prior[i] = np.load('./inputs/prior/stored_sim_prior_%s_from%d_to%d_seed1234.npy' % (rule, sim_from[i], sim_to[i]), allow_pickle=True)[:,1]
    
    jj = 0
    pro_prior = np.zeros((sim_to[-1], 1200))     
    for i in range(len(sim_from)):
        for j in range(250):
            for k in range(1200):
                pro_prior[jj,k] = sims_prior[i][j][k]
            jj = jj + 1
            
    pro_prior = pro_prior.reshape(sim_to[-1], 4, 300)    
    
    sp_coord = np.linspace(0,300,300)
    obs_wells = ['Alzpitz', 'B1', 'B3', 'B4']
    
    sims_all_exp = np.mean(sims_all, axis=0)
    sims_all_std = np.std(sims_all, axis=0)
      
    pro_prior_exp = np.mean(pro_prior, axis=0)
    pro_prior_std = np.std(pro_prior, axis=0)
    
    sims_all_exp_average = np.mean(sims_all_exp, axis=1).round(4)
    sims_all_std_average = np.mean(sims_all_std, axis=1).round(4)
    pro_prior_exp_average = np.mean(pro_prior_exp, axis=1).round(4)
    pro_prior_std_average = np.mean(pro_prior_std, axis=1).round(4)
    
    sims_all_std_average_alltimes = np.mean(sims_all_std_average).round(4)
    pro_prior_std_average_alltimes = np.mean(pro_prior_std_average).round(4)
    
    colors = ['black', 'deepskyblue', fc.redpermanent]
    
    for i in range(len(obs_wells)):
        plt.figure('%s' % (obs_wells[i]), figsize=[6,3])
        plt.title('%s' % (obs_wells[i]), loc='left', fontsize=15)
                

        plt.plot(sp_coord, sims_all_exp[i,:], color='blue', alpha=1, label=r'$\mu_o$')
        plt.fill_between(sp_coord, sims_all_exp[i,:], sims_all_exp[i,:] - sims_all_std[i,:], 
                          color=colors[1], alpha=0.55, edgecolor='None', label=r'[$\mu_o$ + $\sigma_o$, $\mu_o$ - $\sigma_o$]')
        plt.fill_between(sp_coord, sims_all_exp[i,:], sims_all_exp[i,:] + sims_all_std[i,:], 
                          color=colors[1], alpha=0.55, edgecolor='None')

        plt.plot(sp_coord, observations[i,:], 
                 linestyle=':', color=colors[0], alpha=0.95, linewidth=2.5, label='Observations')
        
        plt.ylabel('Groundwater head [m a.s.l]', fontsize=12)
        plt.xlabel('Stress period', fontsize=12)
        plt.xlim(0,300)
        plt.legend(edgecolor='black', fancybox=False, fontsize=8.5,
                    borderpad=0.8, handletextpad=0.9, labelspacing=0.65)
        plt.savefig('./figures/' + flag + '_uq-time_' + obs_wells[i] + '.png')
        

    phases = [86, 127, 145, 290]
    
    print('prior')
    for i in phases:
        print(pro_prior_exp[:,i].round(4))
        print(pro_prior_std[:,i].round(4))
    
    print('posterior')
    for i in phases:
        print(sims_all_exp[:,i].round(4))
        print(sims_all_std[:,i].round(4))



    # %% Save results
    
    # save estimated parameters
    np.save('./outputs/estimated_parameters_all', params_new_all)
    np.save('./outputs/estimated_parameters_con', params_new_con)
    
    
    # %% Create frame with prior predictive
    
    # pro_prior = pro_prior[900:]
    length_prior = len(pro_prior)
    
    txt_before = np.full(length_prior*4, 'Before the flood')
    txt_peak = np.full(length_prior*4, 'Peak-flow')
    txt_recession = np.full(length_prior*4, 'Recession phase')
    txt_after = np.full(length_prior*4, 'After the flood')
    
    txt_wells = np.hstack((np.full((length_prior), 'Alzpitz'),
                           np.full((length_prior), 'B1'),
                           np.full((length_prior), 'B3'),
                           np.full((length_prior), 'B4')))
    
    sims_valid_before =    pro_prior[:,:,86].reshape((length_prior*4), order='F')
    sims_valid_peak =      pro_prior[:,:,127].reshape((length_prior*4), order='F')
    sims_valid_recession = pro_prior[:,:,145].reshape((length_prior*4), order='F')
    sims_valid_after =     pro_prior[:,:,290].reshape((length_prior*4), order='F')
    
    stack_before =    np.vstack((sims_valid_before, txt_before, txt_wells)).T
    stack_peak =      np.vstack((sims_valid_peak, txt_peak, txt_wells)).T
    stack_recession = np.vstack((sims_valid_recession, txt_recession, txt_wells)).T
    stack_after =     np.vstack((sims_valid_after, txt_after, txt_wells)).T
   
    stack_all_prior = np.vstack((stack_before,
                                 stack_peak,
                                 stack_recession,
                                 stack_after))
    
    stack_all_prior = np.c_[stack_all_prior, np.full((len(stack_all_prior)), 'Prior')]
    
    stack_all_prior[:,0].astype('float')
    

    # %% Violin plot

    import seaborn as sns
    
    burn_in = int(0.6 * len(sims_all))
    cold = len(sims_all) - burn_in
    length_text = (len(sims_all) - burn_in) * 4
    
    txt_before = np.full(length_text, 'Before the flood')
    txt_peak = np.full(length_text, 'Peak-flow')
    txt_recession = np.full(length_text, 'Recession phase')
    txt_after = np.full(length_text, 'After the flood')
    
    txt_wells = np.hstack((np.full((len(sims_all) - burn_in), 'Alzpitz'),
                           np.full((len(sims_all) - burn_in), 'B1'),
                           np.full((len(sims_all) - burn_in), 'B3'),
                           np.full((len(sims_all) - burn_in), 'B4')))
    
    sims_valid_before =    sims_all[burn_in:,:,86].reshape((cold*4), order='F')
    sims_valid_peak =      sims_all[burn_in:,:,127].reshape((cold*4), order='F')
    sims_valid_recession = sims_all[burn_in:,:,145].reshape((cold*4), order='F')
    sims_valid_after =     sims_all[burn_in:,:,290].reshape((cold*4), order='F')
    
    stack_before =    np.vstack((sims_valid_before, txt_before, txt_wells)).T
    stack_peak =      np.vstack((sims_valid_peak, txt_peak, txt_wells)).T
    stack_recession = np.vstack((sims_valid_recession, txt_recession, txt_wells)).T
    stack_after =     np.vstack((sims_valid_after, txt_after, txt_wells)).T
   
    stack_all_post = np.vstack((stack_before,
                                stack_peak,
                                stack_recession,
                                stack_after))
    
    stack_all_post = np.c_[stack_all_post, np.full((len(stack_all_post)), 'Posterior')]
    
    stack_all = np.vstack((stack_all_prior, stack_all_post))

    stack_df = pd.DataFrame(data=stack_all, columns=['Groundwater heads [m a.s.l.]', 'Phase', 'Well', 'Problem'])
    
    stack_df['Groundwater heads [m a.s.l.]'] = pd.to_numeric(stack_df['Groundwater heads [m a.s.l.]'])
   
    sns.set_theme(style="whitegrid")
    
    outliers_min = [441, 441, 441, 441]
    outliers_max = [500, 500, 500, 500]
    fig_lo_lims = [460, 460.75, 459, 457]
    fig_up_lims = [466, 466, 467, 463]
    
    for i in range(len(obs_wells)):
        
        stack_df = stack_df[~(stack_df['Groundwater heads [m a.s.l.]'] < outliers_min[i])]
        stack_df = stack_df[~(stack_df['Groundwater heads [m a.s.l.]'] > outliers_max[i])]
        
        plt.figure('Violin' + obs_wells[i], figsize=(5,15))
        tips = stack_df[stack_df.Well == obs_wells[i]]
        plt.title(obs_wells[i], loc='left', fontsize=16)  
        ax = sns.violinplot(x="Phase", y='Groundwater heads [m a.s.l.]', hue='Problem', alpha=0.5,
                            data=tips, 
                            split=True, 
                            inner='quartile', 
                            scale="width", 
                            width=.90,
                            linewidth=0.2,
                            scale_hue=False,
                            palette=[fc.orangecad, fc.bluesaph])
        ax.set_ylim(fig_lo_lims[i], fig_up_lims[i])
    
    
