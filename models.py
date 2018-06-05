import pymc3 as pm

import theano.tensor as tt

def rwfmm(functional_data,static_data,Y,tune=2000,draws = 1000,chains=2,
        func_coef_sd = 'prior',method='nuts',n_iter_approx=30000,
        scalarize=False,robust=False,func_coef_sd_hypersd = 0.05,
        coefficient_prior='flat',include_random_effect = True,
        level_scale = 1.0):
    '''
    Fits a functional mixed model with a random-walk model of
    the functional coefficient.

    Parameters
    ----------
    functional_data : 4D Numpy array
        Data inputs for functional covariates with expected shape (S,V,T,F)
        where S denotes the number of subjects, V denotes the number of
        visits or repeated observations for each subject, T denotes the
        dimension of the functional data (i.e. number of timesteps)
        and F denotes the number of functional coefficients.
    static_data: 3D Numpy array
        Data inputs for static (i.e. non-functional) covariates which are
        constant for each subject/visits combination.
        This array is expected to have the shape (S,V,C) where
        C denotes the number of static covariates.
    Y: 3D Numpy array
        Responses for the functional regression. This array is expected to
        have the same dimensions as static_dataself.
    tune: int
        Number of tuning steps used in MCMC
    draws: int
        Number of post-tuning draws sampled.
    chains: int
        Number of MCMC chains used.
    func_coef_sd: float or string
        The standard deviation of the Gaussian random walk for all
        functional coefficients. If set to "prior", then this quantity
        will also be treated as a parameter that needs to be estimated.
    method: string
        Designates the method to be ued to fit the model.
        This must be one of "nuts", "mh" or one of the approximate inference
        methods at https://docs.pymc.io/api/inference.html#variational.
    n_iter_approx: int
        Number of optimization iterations to be used if the model fitting
        method is an approximate inference method.
    scalarize: bool
        Determines whether or not the functional coefficients should be
        collapsed into a single coefficient. Setting this to True converts
        the model into a linear mixed model with the functional covariates
        reduced to their averages over the dimension with length T.
    robust: bool
        Determines whether a normal error model or a robust Student-T error
        model is assumed for the residuals.
    func_coef_sd_hypersd: float
        If func_coef_sd is set to "prior", then this parameter sets the
        standard deviation of the half-normal distribution over the
        functional coefficient standard deviation (func_coef_sd). Note that
        in this case, each functional coefficient gets its own standard
        deviation drawn from the same prior defined by this parameter.
    coefficient_prior: string
        Determines the prior placed on the static covariate coefficients as
        well as the mean (a.k.a. the level) of the functional coefficient.
        The options are "flat","normal","horseshoe","finnish_horseshoe".
    include_random_effect: bool
        Determines whether or not a per-subject random intercept is included.
    level_scale: float
        The order of magnitude of the mean of the functional coefficient.
        This will usually not need to be adjuste unless there is a major
        mismatch between the sampler's guess at the mean of the functional
        coefficient and the actual contribution from that part of the model
        to the response.

    Returns
    _______
    trace: pymc3 Trace
        Samples produced either via MCMC or approximate inference during
        fitting.
    model: pymc3 Model
        The model object describing the RWFMM.



    Notes
    -----
    '''

    with pm.Model() as model:
        S,V,T,F = functional_data.shape
        _,_,C   = static_data.shape

        n_covariates = F + C

        if include_random_effect:
            random_effect_mean = pm.Flat('random_effect_mean')
            random_effect_sd   = pm.HalfCauchy('random_effect_sd',beta = 1.0)
            random_effect      = pm.Normal('random_effect',mu = random_effect_mean,sd = random_effect_sd,shape = [S,1])
        else:
            random_effect  = 0.0

        if coefficient_prior is 'flat':
            coef = pm.Flat('coef',shape = n_covariates)

        elif coefficient_prior is 'normal':
            coef_level_sd = pm.HalfCauchy('static_coef_sd',beta = 1.0)
            coef          = pm.Normal('static_coef',sd = coef_level_sd,shape = n_covariates )

        elif coefficient_prior is 'cauchy':
            coef_level_sd = pm.HalfCauchy('static_coef_sd',beta = 1.0)
            coef          = pm.Cauchy('static_coef',beta = coef_level_sd,shape = n_covariates )

        elif coefficient_prior is 'horseshoe':
            loc_shrink = pm.HalfCauchy('loc_shrink_level',beta = 1,shape = n_covariates)
            glob_shrink= pm.HalfCauchy('glob_shrink_level',beta = 1)
            coef = pm.Normal('static_coef',sd = (loc_shrink_static * glob_shrink_static))

        elif coefficient_prior is 'finnish_horseshoe':
            '''nu_c = pm.Gamma(alpha = 2.0, beta = 0.1)
            c_squared = pm.InverseGamma(alpha = nu_c/2,)'''
            raise NotImplementedError

        # Setting scalarize to True makes this into a vanilla linear mixed model.
        if scalarize:
            func_contrib = tt.tensordot(tt.mean(functional_data,axis=2),level,axes=[[2],[0]])

        else:

            if func_coef_sd == 'prior':
                func_coef_sd = pm.HalfNormal('func_coef_sd',sd = func_coef_sd_hypersd,shape=F)

            # The 'jumps' are the small deviations about the mean of the functional
            # coefficient, which is defined as 'level'.
            jumps        = pm.Normal('jumps',sd = func_coef_sd,shape=(T,F))
            random_walks = tt.cumsum(jumps,axis=0) + coef[C:]

            func_coef = pm.Deterministic('func_coef',random_walks)
            func_contrib = tt.tensordot(functional_data,func_coef,axes=[[2,3],[0,1]])/T

        # The part of the response that comes from the static covariates
        static_contrib = tt.tensordot(static_data,coef[0:C],axes = [2,0])

        noise_sd = pm.HalfCauchy('noise_sd',beta = 1.0)

        y_hat = pm.Deterministic('y_hat', static_contrib + func_contrib + random_effect)

        # If the robust error option is used, then a gamma-Student-T distribution
        # is placed on the residuals.
        if robust:
            DOF = pm.Gamma('DOF',alpha = 2, beta = 0.1)
            response = pm.StudentT('response',mu = y_hat,sd = noise_sd,nu = DOF,observed = Y)
        else:
            response = pm.Normal('response',mu = y_hat,sd = noise_sd,observed = Y)

        # NUTS is the default PyMC3 sampler and is what we recommend for fitting.
        if method == 'nuts':
            trace = pm.sample(draws,tune = tune,chains = chains)

        # Metropolis-Hastings does poorly with lots of correlated parameters,
        # so this fitting method should only be used if T is small or you are
        # fitting a scalarized model.
        elif method == 'mh':
            trace = pm.sample(draws,tune = tune,chains = chains,step = pm.Metropolis())

        # There are a number of approximate inference methods available, but
        # none of them gave results that were close to what we got with MCMC.
        else:
            approx = pm.fit(n=n_iter_approx,method=method)
            trace = approx.sample(draws)

    return trace,model
