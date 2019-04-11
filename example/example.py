__author__ = 'duarte'
import nest
import numpy as np
import pylab as pl
from functions import *
##################################################################################################
# Network Simulation
##################################################################################################

# Parameters
nValues = 5    # Number of brightness values
nTrials = 10     # Determine the number of trials
stimulus_duration = 20.
N = 50
dt = 0.1
SimT = nTrials*nValues*stimulus_duration

# Set Input:
brightness = np.linspace(0, 90, nValues)

# Create a random stimulus order from 1 to nTrials*nValues
stimulus_order = np.random.permutation(nTrials*nValues)

# Create a range from 0-9 with 20 instances each (easier using modulus)
stimulus_order = stimulus_order % nValues

x = np.array([(brightness[n]/100.)+.1 for n in stimulus_order])
times = np.arange(dt, (nTrials*nValues*stimulus_duration), stimulus_duration)

# Tuning curves for encoding layer
tuning = 250. * np.random.randn(N) + 1000.

amplitudes = np.zeros((N, len(x)))
J_bias = 200.0  # [pA] - constant bias current

# randomize thresholds and initial states
thresholds = 5 * np.random.randn(N) - 50.
Vm0 = np.array(np.random.uniform(low=-70., high=-50., size=int(N)))

# Prepare simulation
nest.ResetKernel()
nest.SetKernelStatus({'resolution': dt,
                      'print_time': True,
                      'local_num_threads': 8})

step_generator = nest.Create('step_current_generator', N)
pop = nest.Create('iaf_neuron', N, {'I_e': J_bias})
spk_det = nest.Create('spike_detector')

for n in range(N):
	amplitudes[n, :] = x * tuning[n]
	nest.SetStatus([pop[n]], {'V_m': Vm0[n], 'V_th': thresholds[n]})
	nest.SetStatus([step_generator[n]], {'amplitude_times': times,
	                                     'amplitude_values': amplitudes[n]})

	nest.Connect([step_generator[n]], [pop[n]])

nest.Connect(pop, spk_det)

nest.Simulate(SimT)

#############################################################################
# Analyse data
#############################################################################
spike_times = nest.GetStatus(spk_det)[0]['events']['times']
neuron_ids = nest.GetStatus(spk_det)[0]['events']['senders']

time_vector = np.arange(0., SimT, dt)
signal = np.zeros_like(time_vector)
for tt in range(len(times)-1):
	signal[int(times[tt]/dt):int(times[tt+1]/dt)] = x[tt]

state_matrix = filter_spikes(spike_times, neuron_ids, N, 0., SimT, 0.1, 30.)
Gamma = np.zeros((N, N))
Ups = np.zeros(N)

## Plot
fig = pl.figure()
fig.suptitle('Results')
ax11 = fig.add_subplot(211)
ax12 = fig.add_subplot(212)

ax11.plot(spike_times, neuron_ids, '.k', markersize=1)
ax12.plot(signal, 'r', linewidth=2)

for i in range(N):
	for j in range(N):
		Gamma[i,j] = np.dot(state_matrix[i,:], np.transpose(state_matrix[j,:]))
		Ups[j] = np.dot(state_matrix[j,:], signal)

decoder = np.dot(np.linalg.pinv(Gamma), Ups)
decoded_x = np.dot(state_matrix.transpose(), decoder)

ax12.plot(decoded_x)

print('MSE:', np.mean((signal-decoded_x)**2))
pl.show()
