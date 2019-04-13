import sys
import time
import numpy as np
from sklearn.linear_model import LinearRegression


#######################################################################################
def filter_spikes(spike_times, neuron_ids, nNeurons, t_start, t_stop, dt, tau):
	"""
	Returns an NxT matrix where each row represents the filtered spiking activity of
	one neuron and the columns represent time...

	Inputs:
		- spike_times - list of spike times
		- neuron_ids - list of spike ids
		- dt - time step
		- tau - kernel time constant
	"""

	neurons = np.unique(neuron_ids)
	new_ids = neuron_ids - min(neuron_ids)
	N = round((t_stop - t_start) / dt)
	StateMat = np.zeros((int(nNeurons), int(N)))

	# include a simple progress bar
	toolbar_width = len(neurons)
	sys.stdout.write("Filtering SpikeTrains [%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1))

	for i, n in enumerate(neurons):
		idx = np.where(neuron_ids == n)[0]

		spk_times = spike_times[idx]
		StateMat[new_ids[idx][0], :] = spikes_to_states(spk_times, t_start, t_stop, dt, tau)

		sys.stdout.write(">")
		sys.stdout.flush()
	sys.stdout.write("\n")

	return StateMat


#######################################################################################
def spikes_to_states(spike_times, t_start, t_stop, dt, tau):
	"""
	Converts a spike train into an analogue variable (liquid state),
	by convolving it with an exponential function.
	This process is supposed to mimic the integration performed by the
	postsynaptic membrane upon an incoming spike.

	Inputs:
		spike_times - array of spike times for a single neuron
		dt     - time step
		tau    - decay time constant
	Examples:
	>> spikes_to_states(spk_times, 0.1, 20.)
	"""

	nSpk = len(spike_times)
	state = 0.
	#t_start = np.min(spike_times)
	#t_stop = np.max(spike_times)
	N = round((t_stop - t_start) / dt)

	States = np.zeros((1, int(N)))[0]

	TimeVec = np.round(np.arange(t_start, t_stop, dt), 1)
	decay = np.exp(-dt / tau)

	if nSpk:
		idx_Spk = 0
		SpkT = spike_times[idx_Spk]

		for i, t in enumerate(TimeVec):
			if (np.round(SpkT, 1) == np.round(t,1)):  # and (idx_Spk<nSpk-1):
				state += 1.
				if (idx_Spk < nSpk - 1):
					idx_Spk += 1
					SpkT = spike_times[idx_Spk]
			else:
				state = state * decay
			if i < int(N):
				States[i] = state

	return States


def compute_capacity(x, z):
	"""
	Compute capacity to reconstruct z based on linearly combining x
	:param x: state matrix (NxT)
	:param z: target output (1xT)
	:return: capacity
	"""
	# explicit method 1
	# W_out = np.dot(np.linalg.pinv(x.T), z.T)
	# z_hat = np.dot(W_out, x)
	# print time.time()
	t_start = time.time()
	reg = LinearRegression(n_jobs=-1, fit_intercept=False, normalize=True, copy_X=False).fit(x.T, z)
	W_out = reg.coef_
	z_hat = np.dot(W_out, x)
	print "\nElapsed time: ", time.time() - t_start
	# pl.plot(z, 'r')
	# pl.plot(z_hat, 'g')
	# pl.show()

	capacity = 1. - (np.mean((z - z_hat) ** 2) / np.var(z))
	error = np.mean((z - z_hat) ** 2)
	return z_hat, capacity, error, np.linalg.norm(W_out)