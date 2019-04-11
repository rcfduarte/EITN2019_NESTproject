import sys
import numpy as np

try:
    import tty, termios
except ImportError:
    try:
        import msvcrt
    except ImportError:
        raise ImportError('getch not available')
    else:
        getch = msvcrt.getch
else:
    def getch():
        """
        getch() -> key character

        Read a single keypress from stdin and return the resulting character.
        Nothing is echoed to the console. This call will block if a keypress
        is not already available, but will not wait for Enter to be pressed.

        If the pressed key was a modifier key, nothing will be detected; if
        it were a special function key, it may return the first character of
        of an escape sequence, leaving additional characters in the buffer.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


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
