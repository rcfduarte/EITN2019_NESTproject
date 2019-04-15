import numpy as np
import matplotlib.pyplot as pl
import sys
import time
from sklearn.linear_model import LinearRegression
from scipy import ndimage

# We may need to use large arrays and long input signals, for which we need to speed up some aspects of the analysis,
# namely the filtering of the spike trains and the capacity computation


########################################################################################################################
# 1) Filter spikes
#######################################################################################
def filter_spikes(spike_times, neuron_ids, nNeurons, t_start, t_stop, dt, tau, method=0):
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
		if method == 0:
			StateMat[new_ids[idx][0], :] = spikes_to_states(spk_times, t_start, t_stop, dt, tau)
		elif method == 1:
			StateMat[new_ids[idx][0], :] = shotnoise_fromspikes(spk_times, 1., tau, dt=dt, t_start=t_start, t_stop=t_stop,
			                                                    eps=1.0e-8)

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


#############################################################
def shotnoise_fromspikes(spike_times, q, tau, dt=0.1, t_start=None, t_stop=None, eps=1.0e-8):
	"""
	Convolves the provided spike train with shot decaying exponentials yielding so called shot noise
	if the spike train is Poisson-like. Returns an AnalogSignal if array=False, otherwise (shotnoise,t)
	as numpy arrays.
	:param spike_train: a SpikeTrain object
	:param q: the shot jump for each spike
	:param tau: the shot decay time constant in milliseconds
	:param dt: the resolution of the resulting shotnoise in milliseconds
	:param t_start: start time of the resulting AnalogSignal. If unspecified, t_start of spike_train is used
	:param t_stop: stop time of the resulting AnalogSignal. If unspecified, t_stop of spike_train is used
	:param array: if True, returns (shotnoise,t) as numpy arrays, otherwise an AnalogSignal.
	:param eps: - a numerical parameter indicating at what value of the shot kernel the tail is cut.  The
	default is usually fine.
	"""
	def spike_index_search(t_steps, spike_times):
		"""
		For each spike, assign an index on the window timeline (t_steps)
		:param t_steps: numpy array with time points representing the binning of the time window by dt
		:param spike_times: numpy array with spike times of a spike train
		:return:
		"""
		result_ = []
		spike_times.sort()
		cnt = 0
		for idx_, val in enumerate(t_steps):
			if cnt >= len(spike_times):
				break
			# check for approximate equality due to floating point fluctuations
			if np.isclose(val, spike_times[cnt], atol=0.099999):
				result_.append(idx_)
				cnt += 1
		return result_

	# time of vanishing significance
	vs_t = -tau * np.log(eps / q)

	t_size = int(np.round((t_stop - t_start) / dt))
	t = np.linspace(t_start, t_stop, num=t_size, endpoint=False)
	kern = q * np.exp(-np.arange(0.0, vs_t, dt) / tau)

	spike_t_idx = spike_index_search(t, spike_times)

	idx = np.clip(spike_t_idx, 0, len(t) - 1)
	a = np.zeros(np.shape(t), float)
	if len(spike_t_idx) > 0:
		a[idx] = 1.0
	y = np.convolve(a, kern)[0:len(t)]

	signal_t_size = int(np.round((t_stop - t_start) / dt))
	signal_t = np.linspace(t_start, t_stop, num=signal_t_size, endpoint=False) #np.arange(window_start, t_stop, dt)
	signal_y = y[-len(signal_t):]
	return signal_y


def filter_spikes2(spk_times, tau, dt=0.1, t_start=None, t_stop=None, eps=1.0e-8):
	def get_histogram(spikes, window=None, dt=0.1, bins=None):
		if window is None:
			window = (spikes[0], spikes[-1])

		if bins is None:
			bins = np.arange(window[0], window[1], dt)
			bins = np.append(bins, window[1])

		return np.histogram(spikes, bins)

	# time of vanishing significance
	vs_t = -tau * np.log(eps / 1.)

	t_size = int(np.round((t_stop - t_start) / dt))
	t = np.linspace(t_start, t_stop, num=t_size, endpoint=False)
	kern = 1. * np.exp(-np.arange(0.0, vs_t, dt) / tau)


	(discrete, bin_edges) = get_histogram(spk_times, window=None, dt=dt)
	ndimage.convolve1d(np.asfarray(discrete), weights=kern, origin='left', mode='constant')



###################################################################
N = 100
T = 1000
spk_times = np.random.uniform(low=0., high=T, size=100000)
neuron_ids = np.random.randint(low=0, high=N, size=100000)

# - current method
start_t = time.time()
state1 = filter_spikes(spk_times, neuron_ids, N, 0., T, dt=0.1, tau=20., method=0)
print("Current method: {0}".format(time.time()-start_t))

start_t = time.time()
state2 = filter_spikes(spk_times, neuron_ids, N, 0., T, dt=0.1, tau=20., method=1)
print("New method: {0}".format(time.time()-start_t))


# pl.plot(spk_times, neuron_ids, '.')
# pl.show()

########################################################################################################################
# Capacity calculation
def compute_capacity(x, z, method=0):
	"""
	Compute capacity to reconstruct z based on linearly combining x
	:param x: state matrix (NxT)
	:param z: target output (1xT)
	:return: capacity
	"""
	# explicit method 1
	if method == 0:
		W_out = np.dot(np.linalg.pinv(x.T), z.T)
		z_hat = np.dot(W_out, x)

	elif method == 1:
		reg = LinearRegression(n_jobs=-1, fit_intercept=False, normalize=True, copy_X=False).fit(x.T, z)
		W_out = reg.coef_
		z_hat = np.dot(W_out, x)

	capacity = 1. - (np.mean((z - z_hat) ** 2) / np.var(z))

	return z_hat, capacity


T = 100000
N = 1000
states = np.random.normal(loc=5., scale=2., size=(N, T))
u = np.random.uniform(low=0., high=1., size=T)

start_t = time.time()
out1, cap1 = compute_capacity(states, u, method=0)
print("Current method: {0}".format(time.time()-start_t))

start_t = time.time()
out2, cap2 = compute_capacity(states, u, method=1)
print("New method: {0}".format(time.time()-start_t))