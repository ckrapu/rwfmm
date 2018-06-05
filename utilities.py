import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_random_sigmoid(T,squashiness = 0.2,vertical_scale = 0.01):
    """Produces a vector f(x) where f is a sigmoid function."""
    x = np.linspace(-1.0,1.0,T)
    rescaling_factor = np.abs(np.random.randn()*squashiness)
    f = sigmoid(x / rescaling_factor)
    return f * vertical_scale

def generate_ar(T,coef = 0.9,jump_sd = 0.01):
    """Creates a realization of an AR(1) process with normally distributed jumps."""
    jumps = np.random.randn(T) * jump_sd
    signal = np.zeros(T)
    for i in range(1,T):
        signal[i] = signal[i-1] * coef + jumps[i]
    return signal

def generate_rw(T,vertical_scale = 0.03):
    """Returns a scalar random walk in a numpy array of length T."""
    return np.cumsum(np.random.randn(T)) * vertical_scale

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
                     xlabel='Timestep',ylabel='B(t)'):
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
        axes[i].set_title(titles[i])
    plt.tight_layout()

    return figure,axes





class FuncDataset(object):

    def __init__(self,n=300,p_static=8,p_func=4,T = 336,noise_sd = 0.1):
        self.n         = n
        self.p_static  = p_static
        self.p_func    = p_func
        self.T         = T
        self.noise_sd  = noise_sd

    def populate(self,seed=None,coefs_func = None,design_func = None,
                            coef_func_producer = generate_random_sigmoid,
                            design_func_producer= generate_rw):

        if seed is not None:
            np.random.seed(seed)

        self.coef_func_producer   = coef_func_producer
        self.design_func_producer = design_func_producer

        self.coefs_func  = coefs_func
        self.design_func = design_func

        self.populate_static()
        self.populate_func()
        self.populate_responses()

    def populate_static(self):
        self.design_static = np.random.randn(self.n,self.p_static)
        self.coef_static   = np.random.randn(self.p_static)

    def populate_func(self):
        if self.coefs_func is not None:
            assert self.coefs_func.shape == (self.T,self.p_func)
        else:
            self.coefs_func = np.zeros([self.T,self.p_func])

            for i in range(self.p_func):
                self.coefs_func[:,i] = (self.coef_func_producer)(self.T)


        if self.design_func is not None:
            assert self.design_func.shape == (self.T,self.n)
        else:
            self.design_func = np.zeros([self.T,self.n])
            for i in range(self.n):
                self.design_func[:,i] = (self.design_func_producer)(self.T)

    def populate_responses(self):

        self.response_from_static = self.design_static.dot(self.coef_static)
        self.response_from_func   =  np.mean(np.sum(self.coefs_func.T.dot(self.design_func),axis=0))

        self.noise = np.random.normal(scale = self.noise_sd,size = self.n)

        self.response = self.response_from_static + self.response_from_func + self.noise


def get_data(response_col,functional_covariates,static_covariates,log_transform_response = False,T=336,standardize_inputs = False):
    df = pd.read_pickle('/home/ubuntu/Dropbox/wfmm/intermediate/no_wavelet_dataframe_5_6.p')
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
    print '{0} patient/visit pairs have valid observations.'.format(np.sum(is_valid))

    if standardize_inputs:
        D_func   = (D_func - np.mean(D_func,axis = (0,1,2))) / np.std(D_func,axis = (0,1,2))
        D_static = (D_static - np.mean(D_static,axis = (0,1))) / np.std(D_static,axis = (0,1))

    return D_func,D_static,Y
