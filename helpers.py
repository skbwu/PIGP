import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from scipy.special import kv, kvp, gamma
from tqdm.notebook import tqdm


# take in timesteps I + hparams (phi1, phi2, v) and returns (C_d, m_d, K_d) for a given component dim
def build_matrices(I, phi1, phi2, v=2.01):
    '''
    Takes in discretized timesteps I and hparams (phi1, phi2, v). Returns (C_d, m_d, K_d) for component d.
    - I is an np.array of discretized timesteps, phi1 & phi2 are floats.
    '''

    # tile appropriately to facilitate vectorization
    s = np.tile(A=I.reshape(-1, 1), reps=I.shape[0]); t = s.T

    # l = |s-t|, u = sqrt(2*nu) * l / phi2 - let's nan out diagonals to avoid imprecision errors.
    l = np.abs(s - t); u = np.sqrt(2*v) * l / phi2; np.fill_diagonal(a=u, val=np.nan)

    # pre-compute Bessel function + derivatives
    Bv0, Bv1, Bv2 = kvp(v=v, z=u, n=0), kvp(v=v, z=u, n=1), kvp(v=v, z=u, n=2)

    # 1. Kappa itself, but we need to correct everywhere with l=|s-t|=0 to have value exp(0.0) = 1.0
    Kappa = (phi1/gamma(v)) * (2 ** (1 - (v/2))) * ((np.sqrt(v) / phi2) ** v)
    Kappa *= Bv0
    Kappa *= (l ** v)
    
    # https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    np.fill_diagonal(Kappa, val=phi1) # behavior as |s-t| \to 0^+

    # 2. p_Kappa, but need to replace everywhere with l=|s-t|=0 to have value 0.0.
    p_Kappa = (2 ** (1 - (v/2)))
    p_Kappa *= phi1 * ((u / np.sqrt(2)) ** v)
    p_Kappa *= ( (u * phi2 * Bv1) + (v*phi2*Bv0) )
    p_Kappa /= (phi2 * (s-t) * gamma(v))
    np.fill_diagonal(p_Kappa, val=0.0) # behavior as |s-t| \to 0^+

    # 3. Kappa_p (by symmetry)
    Kappa_p = p_Kappa * -1

    # 4. Kappa_pp - let's proceed term-by-term (save multiplier terms at the end)
    Kappa_pp = 2 * np.sqrt(2) * (v ** 1.5) * phi2 * l * Bv1
    Kappa_pp += ( ( (v ** 2) * (phi2 ** 2) ) - ( v * (phi2 ** 2) ) ) * Bv0
    Kappa_pp += ( (2 * v * (s ** 2)) - (4 * v * s * t) + (2 * v * (t ** 2)) ) * Bv2
    Kappa_pp *= ( -1.0 * (2 ** (1 - (v/2))) * phi1 * ((u / np.sqrt(2)) ** v) )
    Kappa_pp /= ( (phi2 ** 2) * (l ** 2) * gamma(v) )
    
    # CHECK WITH PROF. YANG ABOUT THIS ONE! SHOULD THERE BE A NEGATIVE HERE?
    np.fill_diagonal(Kappa_pp, val=v*phi1/( (phi2 ** 2) * (v-1) )) # behavior as |s-t| \to 0^+

    # 5. form our C, m, and K matrices (let's not do any band approximations yet!)
    C_d, Kappa_inv = Kappa.copy(), np.linalg.pinv(Kappa)
    m_d = p_Kappa @ Kappa_inv
    K_d = Kappa_pp - (p_Kappa @ Kappa_inv @ Kappa_p)
    
    # 6. return our three matrices
    return C_d, m_d, K_d


# function for fitting (phi1, phi2, sigma_sq) assuming noise is not specified for ALL OBSERVED COMPONENTS!
def fit_hparams(I, X_obs_discret, verbose=True, observed=True):
    
    # check which components are observed + track how many components we have
    observed_indicators = (~np.isnan(X_obs_discret)).mean(axis=0) > 0
    D = X_obs_discret.shape[1]
    observed_components = np.arange(D)[observed_indicators]
    
    # how many OBSERVED components do we have?
    D_observed = len(observed_components)
    
    # note that I is already defined. Let's interpolate only our OBSERVED COMPONENTS
    X_interp = X_obs_discret.copy() # includes nans for missing components still!
    I = I.reshape(-1, 1); indices = np.arange(I.shape[0])

    # data structures to store our prior-means, mu_phi2s, and sd_phi2s for OBSERVED COMPONENTS
    mu_ds_observed, mu_phi2s_observed, sd_phi2s_observed = [], [], []

    #### LINEAR INTERPOLATION + GETTING PARAMS FOR THE PRIORS!

    # start by doing linear interpolation + getting priors on phi2
    for d in observed_components:

        # linear interpolation if we need to (i.e., there are NaNs in this column)
        if np.any(np.isnan(X_interp[:,d])):
            X_interp[:,d] = np.interp(x=indices, xp=indices[~np.isnan(X_obs_discret[:,d])], 
                                      fp=X_obs_discret[~np.isnan(X_obs_discret[:,d]),d])

        # phi2 priors: mean + SD
        z = np.fft.fft(X_interp[:,d]); zmod = np.abs(z)
        zmod_effective = zmod[1:(len(zmod) - 1) // 2 + 1]; zmod_effective_sq = zmod_effective ** 2
        idxs = np.linspace(1, len(zmod_effective), len(zmod_effective))
        freq = np.sum(idxs * zmod_effective_sq) / np.sum(zmod_effective_sq)
        mu_phi2 = 0.5 / freq; sd_phi2 = (1 - mu_phi2) / 3

        # get prior mean for our data, too
        mu_d = X_interp[:,d].mean() # CHECK WITH PROF. YANG ABOUT THIS ONE!

        # add to our lists
        mu_ds_observed.append(mu_d); mu_phi2s_observed.append(mu_phi2); sd_phi2s_observed.append(sd_phi2)

    # convert to arrays
    mu_ds_observed = np.array(mu_ds_observed)
    mu_phi2s_observed = np.array(mu_phi2s_observed)
    sd_phi2s_observed = np.array(sd_phi2s_observed)

    #### BUILDING GP PROCESSES (NO ODE INFO) FOR ALL OBSERVED COMPONENTS
    def build_gps_observed(phi1s_observed, sigma_sqs_observed, phi2s_observed):

        # broadcast across components!
        if D_observed != 1:
            kernel = tfk.GeneralizedMatern(df=2.01, 
                                           amplitude=tf.sqrt(phi1s_observed)[:,None], 
                                           length_scale=phi2s_observed[:,None])
        else:
            kernel = tfk.GeneralizedMatern(df=2.01, 
                                           amplitude=tf.sqrt(phi1s_observed), 
                                           length_scale=phi2s_observed)

        # custom mean function
        def mean_fn(x):
            mu_reshaped = tf.reshape(mu_ds_observed, (D_observed, 1, 1))
            return tf.broadcast_to(mu_reshaped, (D_observed, 1, x.shape[-1]))

        # no need for a separate mean function -- just return a scalar! Directly build GP.    
        gps = tfd.GaussianProcess(kernel=kernel, 
                                  index_points=I,
                                  mean_fn=mean_fn,
                                  observation_noise_variance=sigma_sqs_observed[:,None])

        # everything combined
        return gps


    #### GP MODEL FOR FITTING OBSERVED COMPONENTS' (phi1, phi2, sigma_sq) VALUES
    '''
    8/20/2024:
    1. By vectorizing the Gaussian Process, we are overly computing our prior contributions x D_observed times.
    2. Resolve by noting that for Normal / TruncatedNormal, scaling LLH by 1/D ~ scaling Normal variance by D.
    3. gpjm.log_prob() is a D_observed x D_observed matrix of partial derivatives. We can take TRACE!
    '''
    gpjm = tfd.JointDistributionNamed({"phi1s_observed" : 
                                       tfd.TruncatedNormal(loc=np.float64([0.] * D_observed), 
                                                           low=np.float64([0.] * D_observed), 
                                                           high=np.float64([np.inf] * D_observed),
                                                           scale=np.float64([1000.0 * np.sqrt(D_observed)]\
                                                                            * D_observed)), # flat prior
                                       "sigma_sqs_observed" : 
                                       tfd.TruncatedNormal(loc=np.float64([0.] * D_observed), 
                                                           low=np.float64([0.] * D_observed), 
                                                           high=np.float64([np.inf] * D_observed),
                                                           scale=np.float64([1000.0 * np.sqrt(D_observed)]\
                                                                            * D_observed)), # flat prior
                                       "phi2s_observed" : 
                                       tfd.Normal(loc=np.float64(mu_phi2s_observed), 
                                                  scale=np.float64(sd_phi2s_observed\
                                                                   * np.sqrt(D_observed))), # Fourier-informed prior
                                       "observations" : build_gps_observed})


    # define our TO-BE-TRAINABLE variables + constrain them to be positive, and then make them positive.
    phi1s_observed_var = tfp.util.\
    TransformedVariable(initial_value=X_interp[:,observed_components].std(axis=0) ** 2, 
                        bijector=tfp.bijectors.Softplus(), 
                        name="phi1s_observed",
                        dtype=np.float64) # overall variance
    phi2s_observed_var = tfp.util.\
    TransformedVariable(initial_value=mu_phi2s_observed, 
                        bijector=tfp.bijectors.Softplus(), 
                        name="phi2s_observed",
                        dtype=np.float64) # bandwidth
    sigma_sqs_observed_var = tfp.util.\
    TransformedVariable(initial_value=X_interp[:,observed_components].std(axis=0) ** 2,
                        bijector=tfp.bijectors.Softplus(), 
                        name="sigma_sqs_observed",
                        dtype=np.float64) # noise

    # which variables are we attempting to fit?
    trainable_variables_observed = [v.trainable_variables[0] for v in [phi1s_observed_var, 
                                                                       phi2s_observed_var, 
                                                                       sigma_sqs_observed_var]]

    # optimization function + Adam routine initialize
    X_interp_observed = X_interp[:,observed_components].T[:,np.newaxis,:]
    def target_log_prob(phi1s_observed, sigma_sqs_observed, phi2s_observed):
      return gpjm.log_prob({"phi1s_observed": phi1s_observed, 
                            "sigma_sqs_observed": sigma_sqs_observed, 
                            "phi2s_observed": phi2s_observed, 
                            "observations": X_interp_observed})
    num_iters = 1000; optimizer = tf_keras.optimizers.Adam(learning_rate=.01)

    # taking one step of Adam + scaling up to train our model
    @tf.function(autograph=True, jit_compile=True)
    def train_model():
      with tf.GradientTape() as tape:
        loss = -target_log_prob(phi1s_observed_var, 
                                sigma_sqs_observed_var, 
                                phi2s_observed_var)
      grads = tape.gradient(loss, trainable_variables_observed)
      optimizer.apply_gradients(zip(grads, trainable_variables_observed))
      return loss

    if verbose:
        
        # get our status message to not confuse the user
        if observed:
            desc=f"Fitting hparams for {D_observed} observed components"
        else:
            desc=f"Fitting hparams for {D_observed} unobserved components"
            
        for i in tqdm(range(num_iters), desc=desc):
          loss = train_model()
    else:
        for i in range(num_iters):
            loss = train_model()

    # return as outputs our (D_observed,) phi1s, phi2s, and sigma_sqs + intermediate outputs
    return {"phi1s_observed" : phi1s_observed_var._value().numpy(),
            "phi2s_observed" : phi2s_observed_var._value().numpy(),
            "sigma_sqs_observed" : sigma_sqs_observed_var._value().numpy(),
            "X_interp" : X_interp}