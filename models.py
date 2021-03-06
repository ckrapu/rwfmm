import pymc3 as pm
import numpy as np
import scipy as sp
import theano.tensor as tt
import patsy as p

import utilities


def cross_validate_rwfmm(rwfmm_args,rwfmm_kwargs,param_for_tuning,tuning_set,criterion='LOO'):
    model_dict = {}
    trace_list = []

    for param_val in tuning_set:
        modified_kwargs = rwfmm_kwargs.copy()
        modified_kwargs[param_for_tuning] = param_val
        trace,model = rwfmm(*rwfmm_args,**modified_kwargs)
        model_dict[model] = trace

    rankings = pm.stats.compare(model_dict,ic = criterion)
    return rankings, model_dict



def rwfmm(functional_data,static_data,Y,
        func_coef_sd = 'prior', method='nuts',
        robust=False, func_coef_sd_hypersd = 0.1,
        coefficient_prior='flat', include_random_effect = True,
        variable_func_scale = True, time_rescale_func = False,
        sampler_kwargs = {'init':'adapt_diag','chains':1,'tune':500,'draws':500},
        return_model_only = False, n_spline_knots = 20,
        func_coef_type = 'random_walk', spline_degree = 3,spline_coef_sd = 'prior',
        spline_coef_hyper_sd = 2.,
        spline_coef_prior = 'random_walk',spline_rw_sd = 1.,average_every_n = 1,
        spline_rw_hyper_sd = 1.,poly_order=4):
    '''
    Fits a functional mixed model with a random-walk model of
    the functional coefficient. A range of different priors is available for
    the model coefficients.

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
    variable_func_scale : bool
        Determines whether or not to allow the functional coefficients be
        multiplied by a positive number. This can lead to identifiability issues
        if a weak prior is specified on the functional coefficient evolution
        variance.
    time_rescale_func : bool
        If true, divides the functional coefficient by T. This can help make
        the coefficient more interpretable.
    sampler_kwargs: dict
        Any additional arguments to be passed to pm.sample().
    return_model_only: bool
        If true, returns only the model object without sampling. This can be
        helpful for debugging.
    func_coef_type : string
        One of 'constant','random_walk', 'bspline_recursive', 'natural_spline',
        'linear','bspline_design' or 'polynomial'.
        This determines how the functional coefficient will be parameterized. If it
        is 'random_walk', then the coefficient will be computed as the cumulative
        sum of many small normally-distributed jumps whose standard deviation
        is controlled by 'func_coef_sd'. Alternatively, if one of the bspline
        options is used, then the functional coefficient will be a bspline. The option
        'bspline_recursive' builds the coefficient using the de Boor algorithm
        while the options 'bspline_design' and 'natural_spline' build a design
        matrix using patsy's functionality and then estimates the coefficients
        linking that matrix to the functional coefficients. Using 'polynomial'
        specifies the functional coefficient as a polynomial of order 'poly_order'.
        'linear' makes the functional coefficient a linear function of the function
        domain.
    poly_order : int
        The degree of the polynomial used if the functional coefficient type is
        set to 'polynomial'.
    n_spline_knots : int
        In the event that the functional coefficient is one of the bspline choices,
        then this controls how many knots or breakpoints the spline has. In general,
        higher numbers for this value are required for higher spline orders.
    spline_degree : int
        The order of the spline if the functional coefficient is parameterized as a
        bspline. This is also the order of the polynomial for each spline section
        plus 1. Set this equal to 4 for cubic polynomial approximations in the spline.
    spline_coef_sd : float
        The standard deviation of the normal prior on the spline coefficients.
    spline_coef_prior : string
        One of 'normal', 'flat', or 'random_walk'. This controls how the
        bspline coefficients are smoothed.
    spline_rw_sd : string or float
        Either 'prior' or a float. This controls how much the spline coefficients
        are allowed to jump when using a random walk for the spline coefficient
        prior.
    spline_rw_hyper_sd : float
        If 'spline_rw_sd' is set to 'prior', this is the standard deviation
        of the half-Normal prior on the spline random walk jump standard
        deviation.
    average_every_n : int
        This is used to average every n measurements of the functional data
        together. For example, if the functional data corresponds to 96 hourly
        timesteps' worth of data, setting this to 4 would take the 24 hour average
        and reduce the size of T from 96 to 24. The default setting of 1 leaves
        the data unchanged.

    Returns
    -------
    trace: pymc3 Trace
        Samples produced either via MCMC or approximate inference during
        fitting.
    model: pymc3 Model
        The model object describing the RWFMM.
    '''

    with pm.Model() as model:
        S,V,T,F = functional_data.shape
        _,_,C   = static_data.shape

        #functional_data = np.mean(functional_data.reshape(-1, average_every_n), axis=1)


        # We want to make sure the two data arrays agree in the number of
        # subjects (S) and visits (V).
        assert static_data.shape[0:2] == functional_data.shape[0:2]

        # Total number of functional and static coefficients.
        # This does not include the random-walk jumps.
        n_covariates = F + C

        if include_random_effect:
            random_effect_mean     = pm.Flat('random_effect_mean')
            random_effect_sd       = pm.HalfCauchy('random_effect_sd',beta = 1.)
            random_effect_unscaled = pm.Normal('random_effect_unscaled',shape = [S,1])
            random_effect          = pm.Deterministic('random_effect',random_effect_unscaled * random_effect_sd + random_effect_mean)
        else:
            random_effect  = 0.

        if coefficient_prior == 'flat':
            coef = pm.Flat('coef',shape = n_covariates)

        elif coefficient_prior == 'normal':
            coef_sd = pm.HalfCauchy('coef_sd',beta = 1.)
            coef    = pm.Normal('coef',sd = coef_sd,shape = [n_covariates] )

        elif coefficient_prior == 'cauchy':
            coef_sd = pm.HalfCauchy('coef_sd',beta = 1.0)
            coef    = pm.Cauchy('coef',alpha = 0., beta = coef_sd,shape = [n_covariates] )

        elif coefficient_prior == 'horseshoe':
            loc_shrink = pm.HalfCauchy('loc_shrink',beta = 1,shape = [n_covariates])
            glob_shrink= pm.HalfCauchy('glob_shrink',beta = 1)
            coef = pm.Normal('coef',sd = (loc_shrink * glob_shrink),shape = [n_covariates])

        # Implemented per Piironnen and Vehtari '18
        elif coefficient_prior == 'finnish_horseshoe':

            loc_shrink  = pm.HalfCauchy('loc_shrink',beta = 1,shape = [n_covariates])
            glob_shrink = pm.HalfCauchy('glob_shrink',beta = 1)

            # In order to get some of the values within the prior calculations,
            # we need to know the variance of the predictors.
            static_var = np.var(static_data,axis = (0,1))
            func_var   = np.var(functional_data,axis = (0,1,2))
            variances  = np.concatenate([static_var,func_var])

            nu_c = pm.Gamma('nu_c',alpha = 2.0, beta = 0.1)
            c    = pm.InverseGamma('c',alpha = nu_c/2, beta = nu_c * variances / 2,shape = [n_covariates])

            regularized_loc_shrink = c * loc_shrink**2 / (c + glob_shrink**2 * loc_shrink**2)

            coef = pm.Normal('coef',sd = (regularized_loc_shrink * glob_shrink**2)**0.5,shape = [n_covariates])

        if func_coef_type == 'constant':
            func_coef = pm.Deterministic('func_coef',tt.zeros([T,F]) + coef[C:])

        elif func_coef_type == 'random_walk':

                if func_coef_sd == 'prior':
                    func_coef_sd = pm.HalfNormal('func_coef_sd',sd = func_coef_sd_hypersd,shape=F)

                # The 'jumps' are the small deviations about the mean of the functional
                # coefficient.
                if variable_func_scale:
                    log_scale = pm.Normal('log_scale',shape = F)
                else:
                    log_scale = 0.0

                jumps        = pm.Normal('jumps',sd = func_coef_sd,shape=(T,F))
                random_walks = tt.cumsum(jumps,axis=0) * tt.exp(log_scale) + coef[C:]
                func_coef    = pm.Deterministic('func_coef',random_walks)

        elif (func_coef_type == 'natural_spline' or func_coef_type == 'bspline_design'):

            x = np.arange(T)

            # The -1 in the design matrix creation is to make sure that there
            # is no constant term which would be made superfluous by 'coef'
            # which is added to the functional coefficient later.
            if func_coef_type == 'natural_spline':
                spline_basis = p.dmatrix("cr(x, df = {0}) - 1".format(n_spline_knots),{"x":x},return_type = 'dataframe').values
            elif func_coef_type == 'bspline_design':
                spline_basis  = p.dmatrix("bs(x, df = {0}) - 1".format(n_spline_knots),{"x":x},return_type = 'dataframe').values

            # If this produces a curve which is too spiky or rapidly-varying,
            # then a smoothing prior such as a Gaussian random walk could
            # instead be used here.
            if spline_coef_prior == 'normal':
                spline_coef = pm.Normal('spline_coef',sd = spline_coef_sd,shape = [n_spline_knots,F])
            elif spline_coef_prior == 'flat':
                spline_coef = pm.Flat('spline_coef',shape = [n_spline_knots,F])
            elif spline_coef_prior == 'random_walk':
                if spline_rw_sd == 'prior':
                    spline_rw_sd = pm.HalfNormal('spline_rw_sd',sd = spline_rw_hyper_sd,shape = F)
                spline_jumps = pm.Normal('spline_jumps', shape = [n_spline_knots,F])

                spline_coef = pm.Deterministic('spline_coef',tt.cumsum(spline_jumps * spline_rw_sd ,axis = 0))

            # This inner product sums over the spline coefficients
            func_coef = pm.Deterministic('func_coef', (tt.tensordot(spline_basis,spline_coef,axes=[[1],[0]]) + coef[C:]))

        # This is deprecated - it is missing some priors.
        elif func_coef_type == 'bspline_recursive':
            n_spline_coefficients = spline_degree + n_spline_knots + 1
            spline_coef = pm.Normal('spline_coef',sd = spline_coef_sd,shape = [n_spline_coefficients,F])
            x = np.linspace(-4,4,T)

            func_coefs = []
            for f in range(F):
                func_coefs.append(utilities.bspline(spline_coef[:,f],spline_degree,n_spline_knots,x))
            func_coef = pm.Deterministic('func_coef',(tt.stack(func_coefs,axis=1) ))

        elif func_coef_type == 'polynomial':
            poly_basis = np.zeros([T,poly_order])
            for i in range(1,poly_order+1):
                poly_basis[:,i-1] = np.arange(T)**i

            poly_coef = pm.Flat('poly_coef',shape = [poly_order,F])
            func_coef = pm.Deterministic('func_coef',tt.tensordot(poly_basis,poly_coef,axes=[[1],[0]]) + coef[C:])

        elif func_coef_type == 'linear':
                linear_basis = np.zeros([T,F])
                for i in range(F):
                    linear_basis[:,i] = np.arange(T)

                linear_coef = pm.Flat('linear_coef',[F])
                func_coef = pm.Deterministic('func_coef',linear_basis * linear_coef + coef[C:])

        else:
            raise ValueError('Functional coefficient type not recognized.""')

        # This is the additive term in y_hat that comes from the functional
        # part of the model.
        func_contrib = tt.tensordot(functional_data,func_coef,axes=[[2,3],[0,1]])
        if time_rescale_func:
            func_contrib = func_contrib / T
        # The part of y_hat that comes from the static covariates
        static_contrib = tt.tensordot(static_data,coef[0:C],axes = [2,0])

        noise_sd = pm.HalfCauchy('noise_sd',beta = 1.0)

        # y_hat is the predictive mean.
        y_hat = pm.Deterministic('y_hat', static_contrib + func_contrib + random_effect)
        #y_hat = pm.Deterministic('y_hat', static_contrib +func_contrib )

        # If the robust error option is used, then a gamma-Student-T distribution
        # is placed on the residuals.
        if robust:
            DOF = pm.Gamma('DOF',alpha = 2, beta = 0.1)
            response = pm.StudentT('response',mu = y_hat,sd = noise_sd,nu = DOF,observed = Y)
        else:
            response = pm.Normal('response',mu = y_hat,sd = noise_sd,observed = Y)

        if return_model_only:
            return model

        # NUTS is the default PyMC3 sampler and is what we recommend for fitting.
        if method == 'nuts':
            trace = pm.sample(**sampler_kwargs)

        # Metropolis-Hastings does poorly with lots of correlated parameters,
        # so this fitting method should only be used if T is small or you are
        # fitting a scalarized model.
        elif method == 'mh':
            trace = pm.sample(step = pm.Metropolis(),**sampler_kwargs)

        # There are a number of approximate inference methods available, but
        # none of them gave results that were close to what we got with MCMC.
        else:
            approx = pm.fit(method=method,**sampler_kwargs)
            trace = approx.sample(draws)

    return trace,model
