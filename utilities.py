import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def ar1(beta,sd,length):
    '''Creates a realization of a AR(1) process with autoregresssion
    coefficient 'beta', a jump standard deviation of 'sd' and is 'length'
    elements long.'''

    vector = np.zeros(length)
    vector[0] = np.random.randn() * sd
    for i in range(length-1):
        vector[i+1] = vector[i] * beta + np.random.randn() * sd
    return vector

def simulate_dataset(S,V,C,F,T,function_type,error_sd,error_type,autocorr=0.5,functional_covariate_type='normal'):
    '''Function for generating simulated data for a scalar-on-function
    regression with longitudinal measurements and scalar covariates.'

    Parameters
    ----------
    S : integer
        Number of individuals / subjects
    V : integer
        Number of longitudinal measurements per individual
    C : integer
        Number of scalar (i.e. non-functional) covariates
    F : integer
        Number of functional covariates
    T : integer
        Number of elements for each functional covariate measurement
    function_type : string
        One of 'logistic', 'sinusoid', or 'spike'. Defines the shape of the
        generated functional coefficient.
    error_sd : float
        The standard deviation of the residual error distribution.
    error_type : string
        One of 'normal','cauchy', or 'autocorrelated'. Determines the residual distribution.
        If 'autocorrelated' is selected, then the per-subject residuals are autocorrelated
        across longitudinal measurements.
    autocorr : float
        If error_type is set to 'autocorrelated', then this sets the autoregressive coefficient.
        Otherwise, this has no effect.
    functional_covariate_type : string
        One of 'normal' or 'autocorrelated'. The first sets the functional covariates to have a
        standard normal distribution. The second sets the functional covariates to be AR(1) with
        a regression coefficient of 0.5 and a jump standard deviation of 0.5.

    Returns
    -------
    functional_covariates : 4D Numpy array
        Array of functional covariates with shape [S,V,T,F]
    longitudinal_covariates : 3D Numpy array
        Array of per-visit/longitudinal measurement covariates with shape [S,V,C]
    response : 2D Numpy array
        Array of scalar response variable for each individual and each measurement.
        This has a shape of [S,V].
    functional_coefficients : 2D Numpy array
        Array of functional coefficients with shape [T,F]
    longitudinal_coefficients : 1D Numpy array
        Array of coefficients for the nonfunctional predictors with shape [C]
    random_effect : 1D Numpy array
        Per-subject intercepts in an array with shape [S]
    '''

    longitudinal_covariates  = np.random.randn(S,V,C)

    if functional_covariate_type == 'normal':
        functional_covariates    = np.random.randn(S,V,T,F)
    elif functional_covariate_type == 'autocorrelated':
        functional_covariates = np.zeros([S,V,T,F])
        for s,v,f in product(range(S),range(V),range(F)):
            functional_covariates[s,v,:,f] = ar1(0.5,0.5,T)
    else:
        raise ValueError('Covariate type not recognized.')

    random_effect            = np.random.randn(S)[:,np.newaxis].repeat(V,axis = 1)
    longitudinal_coefficient = np.random.randn(C)

    timesteps = np.arange(T)[:,np.newaxis].repeat(F,axis=1)

    if function_type == 'logistic':
        timesteps = np.linspace(-6,6,T)[:,np.newaxis].repeat(F,axis=1)
        functional_coefficients = 1. / (1. + np.exp(-timesteps))
    elif function_type == 'sinusoid':
        timesteps = np.linspace(-3.14*3,3.14*3,T)[:,np.newaxis].repeat(F,axis=1)
        functional_coefficients = np.sin(timesteps)
    elif function_type == 'spike':
        timesteps = np.linspace(-2,2,T)[:,np.newaxis].repeat(F,axis=1)
        functional_coefficients = np.exp(-(timesteps/0.05)**2)
    else:
        raise ValueError('Function type not recognized.')

    longitudinal_mean = np.einsum('ijk,k',longitudinal_covariates,longitudinal_coefficient)
    functional_mean   = np.einsum('ijkl,kl',functional_covariates,functional_coefficients)
    mean = random_effect + longitudinal_mean + functional_mean

    if error_type == 'normal':
        error = np.random.randn(S,V)

    elif error_type == 'cauchy':
        error = np.random.standard_t(10,size=[S,V])

    elif error_type == 'autocorrelated':
        error = np.zeros([S,V])
        for s in range(S):
            error[s,:] = autoregression(autocorr,0.5,V)
    else:
        raise ValueError('Error type not recognized.')

    response = mean + error * error_sd
    return functional_covariates,longitudinal_covariates,response,functional_coefficients,longitudinal_coefficient,random_effect


def coef_plot(samples,upper = 97.5,lower = 2.5):

    T = samples.shape[1]

    plt.figure(figsize = (5,3))
    plt.plot(np.arange(T),np.median(samples,axis = 0),color='k',label = 'Median')
    plt.plot(np.arange(T),np.percentile(samples[:,:],upper,axis = 0),linestyle='--',color='k')
    plt.plot(np.arange(T),np.percentile(samples,lower,axis = 0),linestyle='--',
             color='k',label = '{0}% CI'.format(upper-lower))
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('B(t)')
    plt.grid('on')

def multiple_coef_plot(samples,num_horizontal,num_vertical,titles,upper = 97.5,lower = 2.5,fig_kwargs = {'figsize':(8,6),'sharex':True},
                     xlabel='Timestep',ylabel='B(t)',true_coef = None):
    _,T,F = samples.shape
    figure,axes = plt.subplots(num_vertical,num_horizontal,**fig_kwargs)
    axes = axes.ravel()
    timesteps = np.arange(T)
    zeros = np.zeros_like(timesteps)
    for i in range(F):
        upper_percentile = np.percentile(samples[:,:,i],upper,axis=0)
        lower_percentile = np.percentile(samples[:,:,i],lower,axis=0)
        axes[i].plot(timesteps,np.median(samples[:,:,i],axis = 0),color='k',label = 'Median',linewidth = 2)
        axes[i].plot(timesteps,upper_percentile,color='k',linewidth = 1)
        axes[i].plot(timesteps,lower_percentile,color='k',linewidth = 1)
        axes[i].fill_between(timesteps,upper_percentile,lower_percentile,color='k',alpha = 0.1)
        axes[i].plot(timesteps,zeros,linewidth = 3, linestyle='--',alpha = 0.5,color='k')
        if true_coef is not None:
            axes[i].plot(timesteps,true_coef[i],linewidth = 3,alpha = 0.75,color='g')

        axes[i].set_title(titles[i])
    plt.tight_layout()

    return figure,axes



def get_data(response_col,functional_covariates,static_covariates,log_transform_response = False,T=336,standardize_inputs = False,
            filename = '/home/ubuntu/Dropbox/wfmm/intermediate/no_wavelet_dataframe_5_6.p'):
    '''Function for loading data from a specific data file for use in functional
    linear mixed model.'''
    df = pd.read_pickle(filename)
    P = df.id.unique().shape[0]
    V = df.visit.unique().shape[0]
    F = len(functional_covariates)
    C = len(static_covariates)

    D_func = np.zeros([P,V,T,F])
    D_static = np.zeros([P,V,C])
    Y = np.zeros([P,V])

    # This loop will fill in the design matrix / tensor (for functional data).

    # Iterate over each unique subject
    for p,unique_id in enumerate(df.id.unique()):

        # For each subject, iterate over all visits
        for v,unique_visit in enumerate(df.visit.unique()):

            # Fill in the response variable first.
            # We will overwrite these entries with NaNs if the observation is invalid.
            scalar_response = df[(df.id == unique_id) & (df.visit == unique_visit)][response_col].values
            if len(scalar_response) == 1:
                Y[p,v] = scalar_response

            # fill in the static covariates
            static_row = df[(df.id == unique_id) & (df.visit == unique_visit)][static_covariates]
            if len(static_row) > 0:
                D_static[p,v,:] = static_row
            if np.any(np.isnan(static_row)):

                Y[p,v] = np.nan


            # Fill in the functional covariates
            for f,column_name in enumerate(functional_covariates):
                per_func_cov_cols = [col for col in df.columns if column_name in col]

                # This picks out a T-long vector of values and puts it into the func. design array.
                func_row = df[(df.id == unique_id) & (df.visit == unique_visit)][per_func_cov_cols]
                if len(func_row) > 0:
                    D_func[p,v,:,f] = func_row

                # Again, if the covariate is missing then we want to remove this observation.
                else:
                    Y[p,v] = np.nan

                if np.any(np.isnan(func_row)):
                    Y[p,v] = np.nan

    if log_transform_response:
        Y = np.log(Y)

    is_bad = np.isnan(Y) + np.any(np.isnan(D_static),axis =2) + np.any(np.isnan(D_func),axis =(2,3))
    is_valid = ~is_bad

    Y = np.ma.masked_array(data = Y, mask = is_bad)

    # We will also zero out the entries in the design arrays corresponding
    # to patient/visit pairs which are not valid.
    for p in range(Y.shape[0]):
        for v in range(Y.shape[1]):
            if is_bad[p,v]:
                D_static[p,v,:] = 0.0
                D_func[p,v,:,:] = 0.0
    assert np.all(np.isfinite(D_static))
    assert np.all(np.isfinite(D_func))


    if standardize_inputs:
        D_func   = (D_func - np.mean(D_func,axis = (0,1,2))) / np.std(D_func,axis = (0,1,2))
        D_static = (D_static - np.mean(D_static,axis = (0,1))) / np.std(D_static,axis = (0,1))

    return D_func,D_static,Y
