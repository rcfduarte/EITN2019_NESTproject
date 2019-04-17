# add nest and results paths
import sys
# sys.path.append("/project/3018037.01/Experiment3.2_ERC/tommys_folder/nest-simulator/install/lib64/python3.4/site-packages")
# results_path='/project/3018037.01/Experiment3.2_ERC/tommys_folder/nest_results/'
import nest
import numpy as np
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
    signal_pred = LinearRegression(n_jobs=-1, fit_intercept=False,
                                   normalize=True, copy_X=False).fit(state_mat,
                                                                     signal).predict(
        state_mat)
    MSE_pred = np.mean((signal - signal_pred) ** 2)
    return signal_pred, 1. - (MSE_pred / np.var(signal)), MSE_pred


# from functions import *

network_scale = int(sys.argv[1])

try: results_path
except NameError: results_path = './'

# Global parameters
T = 1000  # total number of time steps
dt = 0.1  # simulation resolution

subsampling_factor = 100  # select a subset of samples (1 = no timewise subsampling)

for u_low in [0., 0.1, 0.2, 0.3, 0.4, 0.5]:

    for duration in [70., 10., 20., 30., 40., 50., 60., 80., 90., 100.]: # [ms]

        np.random.seed(42)
        nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': dt,
            'print_time': True,
            'local_num_threads': 32})

        ###
        # Input parameters

        u = np.random.uniform(low=u_low, high=1., size=T)
        input_times = np.arange(dt, T * duration, duration)

        ###
        # Parameters
        nEnc = 100*network_scale
        J_bias = 200.  # [pA]
        tuning = 250. * np.random.randn(nEnc) + 1000.

        # randomize thresholds and initial states
        thresholds = 5 * np.random.randn(nEnc) - 50.
        Vm0 = np.array(np.random.uniform(low=-70., high=-50., size=int(nEnc)))

        enc_layer = nest.Create('iaf_psc_delta', nEnc, {'I_e': J_bias})

        step_generator = nest.Create('step_current_generator', nEnc)
        amplitudes = np.zeros((nEnc, len(u)))
        for n in range(nEnc):
            amplitudes[n, :] = u * tuning[n]
            nest.SetStatus([enc_layer[n]], {'V_m': Vm0[n], 'V_th': thresholds[n]})
            nest.SetStatus([step_generator[n]], {'amplitude_times': input_times,
                                                 'amplitude_values': amplitudes[n]})
            nest.Connect([step_generator[n]], [enc_layer[n]])

        enc_v = nest.Create('multimeter', 1, {'record_from': ['V_m'],
                                              'interval': dt * subsampling_factor})
        nest.Connect(enc_v, enc_layer)

        #### PARAMETERS ###
        # network parameters
        gamma = 0.25               # relative number of inhibitory connections
        NE = 500*network_scale                   # number of excitatory neurons (10.000 in [1])
        NI = int(gamma * NE)       # number of inhibitory neurons
        CE = 100*network_scale                   # indegree from excitatory neurons
        CI = int(gamma * CE)       # indegree from inhibitory neurons

        # synapse parameters
        w = 0.1                    # excitatory synaptic weight (mV)
        g = 5.                     # relative inhibitory to excitatory synaptic weight
        d = 1.5                    # synaptic transmission delay (ms)

        # neuron paramters
        V_th = 20.                 # spike threshold (mV)
        tau_m = 20.                # membrane time constant (ms)
        neuron_params = {
            'C_m': 1.0,            # membrane capacity (pF)
            'E_L': 0.,             # resting membrane potential (mV)
            'I_e': 0.,             # external input current (pA)
            'V_m': 0.,             # membrane potential (mV)
            'V_reset': 10.,        # reset membrane potential after a spike (mV)
            'V_th': V_th,          #
            't_ref': 2.0,          # refractory period (ms)
            'tau_m': tau_m,        #
        }

        # set default parameters for neurons and create neurons
        nest.SetDefaults('iaf_psc_delta', neuron_params)
        neurons_e = nest.Create('iaf_psc_delta', NE)
        neurons_i = nest.Create('iaf_psc_delta', NI)

        # E synapses
        syn_exc = {'delay': d, 'weight': w}
        conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
        nest.Connect(neurons_e, neurons_e, conn_exc, syn_exc)
        nest.Connect(neurons_e, neurons_i, conn_exc, syn_exc)

        # I synapses
        syn_inh = {'delay': d, 'weight': - g * w}
        conn_inh = {'rule': 'fixed_indegree', 'indegree': CI}
        nest.Connect(neurons_i, neurons_e, conn_inh, syn_inh)
        nest.Connect(neurons_i, neurons_i, conn_inh, syn_inh)

        # device
        net_v = nest.Create('multimeter', 1, {'record_from': ['V_m'], 'interval': dt*subsampling_factor})
        nest.Connect(net_v, neurons_e)

        nest.Connect(enc_layer, neurons_e, conn_exc, syn_exc)
        nest.Connect(enc_layer, neurons_i, conn_exc, syn_exc)

        nest.Simulate(T*duration+dt)

        time_vector = np.arange(0., T*duration, dt) # take 1 sample per step
        signal = np.zeros_like(time_vector)
        for tt in range(len(input_times)-1):
            signal[int(input_times[tt]/dt):int(input_times[tt+1]/dt)] = u[tt]

        ind_init = subsampling_factor*dt

        if ind_init<1:
            ind_init = 0

        # select sub-sampled signal components
        inds_subsampling = np.arange(ind_init, len(signal)+ind_init,subsampling_factor,dtype=int)
        signal=signal[inds_subsampling]

        # get events and purge
        enc_activity = nest.GetStatus(enc_v)[0]['events']
        enc_v = []
        net_activity = nest.GetStatus(net_v)[0]['events']
        net_v = []

        enc_indices = np.sort(np.unique(enc_activity['senders']))
        enc_states = np.zeros((nEnc, int(T * duration / dt / subsampling_factor)))
        for idx, i in enumerate(enc_indices):
            enc_states[idx, :] = enc_activity['V_m'][
                np.where(enc_activity['senders'] == i)[0]]

        net_indices = np.sort(np.unique(net_activity['senders']))
        e_states = np.zeros((NE, int(T * duration / dt / subsampling_factor)))
        for idx, i in enumerate(net_indices):
            e_states[idx, :] = net_activity['V_m'][
                np.where(net_activity['senders'] == i)[0]]

        max_lag = 210. / (subsampling_factor)  # [ms] in this example
        step_lag = 10. / (subsampling_factor)  # [ms] - if != dt (index the time axis)
        time_lags = np.arange(0., max_lag, step_lag)
        indices = [np.where(idx == time_vector)[0][0] for idx in time_lags]

        for idx, lag in zip(indices, time_lags):

            # shift the target signal
            if idx > 0:
                shifted_signal = signal[:-idx]
            else:
                shifted_signal = signal

            # shift the population states
            enc_st = enc_states[:, idx:]
            circ_st = e_states[:, idx:]

            # compute capacity
            enc_estimate, enc_capacity, enc_error = compute_capacity(enc_st,
                                                                     shifted_signal)
            circ_estimate, circ_capacity, circ_error = compute_capacity(circ_st,
                                                                        shifted_signal)


            with open (results_path+"resultsParamAndDur_networkScale_{0}.csv".format(str(network_scale)),"a") as results_file:
                results_file.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(
                    str(network_scale), str(u_low), str(duration), str(lag*subsampling_factor), str(enc_capacity),
                    str(enc_error),str(circ_capacity), str(circ_error)))

