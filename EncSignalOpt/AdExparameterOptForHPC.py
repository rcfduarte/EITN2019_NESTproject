import nest
import numpy as np
import pylab as pl
import sys
from sklearn.linear_model import LinearRegression
def compute_capacity(state_mat, signal):
    """
    Compute capacity to reconstruct y based on linearly combining a
    :param a: state matrix (NxT)
    :param y: target output (1xT)
    :return y_hat: estimated signal
    :return capacity: 
    :return error:
    """
    state_mat = state_mat.T
    signal_pred = LinearRegression(n_jobs=-1, fit_intercept=False, normalize=True, copy_X=False).fit(state_mat, signal).predict(state_mat)
    MSE_pred = np.mean((signal-signal_pred)**2)
    return signal_pred, 1. - (MSE_pred / np.var(signal)), MSE_pred

# parameters
T = 1000  # total number of time steps
dt = 0.1  # simulation resolution
nEnc = 1000 # neurons encoding layer
J_bias = 200. # [pA]

# Initialize NEST
np.random.seed(42)
nest.ResetKernel()
nest.SetKernelStatus({
    'resolution': dt,
    'print_time': True,
    'local_num_threads': 8})

tuning = 250. * np.random.randn(nEnc) + 1000.

# randomize thresholds and initial states
thresholds = 5 * np.random.randn(nEnc) - 50.
Vm0 = np.array(np.random.uniform(low=-70., high=-50., size=int(nEnc)))

u_vals = float(sys.argv[1])
d_vals = int(sys.argv[2])

print("computing capacity for offset: {0} and duration: {1}".format(str(u_vals), str(d_vals)))

# Input parameters
u_range = [u_vals, 1.]
duration = float(d_vals) # [ms]

u = np.random.uniform(low=u_range[0], high=u_range[1], size=T)
input_times = np.arange(dt, T*duration, duration)

neuron_params = {'C_m': 250.0,
                 'Delta_T': 2.0,
                 'E_L': -70.,
                 'E_ex': 0.0,
                 'E_in': -75.0,
                 'I_e': 0.,
                 'V_m': -70.,
                 'V_th': -50.,
                 'V_reset': -60.0,
                 'V_peak': 0.0,
                 'a': 4.0,
                 'b': 80.5,
                 'g_L': 16.7,
                 'g_ex': 1.0,
                 'g_in': 1.0,
                 't_ref': 2.0,
                 'tau_minus': 20.,
                 'tau_minus_triplet': 200.,
                 'tau_w': 144.0,
                 'tau_syn_ex': 2.,
                 'tau_syn_in': 6.0}

nest.SetDefaults('aeif_cond_exp', neuron_params)
enc_layer = nest.Create('aeif_cond_exp ', nEnc, {'I_e': J_bias})


step_generator = nest.Create('step_current_generator', nEnc)
amplitudes = np.zeros((nEnc, len(u)))

for n in range(nEnc):
    amplitudes[n, :] = u * tuning[n]
    nest.SetStatus([enc_layer[n]], {'V_m': Vm0[n], 'V_th': thresholds[n]})
    nest.SetStatus([step_generator[n]], {'amplitude_times': input_times,
                                         'amplitude_values': amplitudes[n]})
    nest.Connect([step_generator[n]], [enc_layer[n]])

enc_v = nest.Create('multimeter', 1, {'record_from': ['V_m'], 'interval':dt})
nest.Connect(enc_v, enc_layer)

nest.Simulate(T*duration+dt)

time_vector = np.arange(0., T*duration, dt)

signal = np.zeros_like(time_vector)

for tt in range(len(input_times)-1):
    signal[int(input_times[tt]/dt):int(input_times[tt+1]/dt)] = u[tt]

enc_activity = nest.GetStatus(enc_v)[0]['events']

enc_indices = np.sort(np.unique(enc_activity['senders']))
enc_states = np.zeros((nEnc, int(T*duration/dt)))
for idx, i in enumerate(enc_indices):
    enc_states[idx, :] = enc_activity['V_m'][np.where(enc_activity['senders']==i)[0]]

max_lag = 110.   # [ms] in this example
step_lag = 10.   # [ms] - if != dt (index the time axis)
time_lags = np.arange(0., max_lag, step_lag)
indices = [np.where(idx==time_vector)[0][0] for idx in time_lags]

encoder_capacity = []

for idx, lag in zip(indices, time_lags):

    # shift the target signal
    if idx > 0:
        shifted_signal = signal[:-idx]
    else:
        shifted_signal = signal

    # shift the population states
    enc_st = enc_states[:, idx:]

    # compute capacity
    enc_estimate, enc_capacity, enc_error = compute_capacity(enc_st, shifted_signal)

    with open ("resultsParamAndDur.csv","a") as results_file:
        results_file.write("{0},{1},{2},{3},{4},{5}\n".format(
            str(0), str(u_vals), str(d_vals),str(lag), str(enc_capacity), str(enc_error)))

    results_file.close()

    print("Lag = {0} ms".format(str(lag)))
    print("Encoding Layer: \n\t- Capacity={0}, MSE={1}".format(str(enc_capacity), str(enc_error)))

    encoder_capacity.append(enc_capacity)

print("Total capacity (encoder): {0} ms".format(str(np.sum(encoder_capacity)*step_lag)))