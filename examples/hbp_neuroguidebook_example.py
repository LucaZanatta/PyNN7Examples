"""
A population of integrate-and-firing neurons with different input firing rates
(example used in the HBP Neuromorphic Computing Guidebook)
"""

import numpy as np
import matplotlib.pyplot as plt
#try:
#    import pyNN.spiNNaker as sim
#except Exception as e:
#    import spynnaker7.pyNN as sim
import spynnaker7.pyNN as sim

sim.setup(timestep=1.0, min_delay=1.0)
print("pyNN0.7 hbp neuro guidebook example")

# create cells
cell_params = {
    'cm': 0.25, 'tau_m': 10.0, 'tau_refrac': 2.0,
    'tau_syn_E': 2.5, 'tau_syn_I': 2.5,
    'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -55.0}

neurons = sim.Population(100, sim.IF_cond_exp, cell_params)
inputs = sim.Population(100, sim.SpikeSourcePoisson, {"rate": 0.0})

# set input firing rates as a linear function of cell index
input_firing_rates = np.linspace(0.0, 1000.0, num=inputs.size)
inputs.tset("rate", input_firing_rates)

# create one-to-one connections
wiring = sim.OneToOneConnector(weights=0.1, delays=2.0)
connections = sim.Projection(inputs, neurons, wiring)

# configure recording
neurons.record()

# run simulation
sim_duration = 10.0  # seconds
sim.run(sim_duration*1000.0)

# retrieve recorded data
spike_counts = neurons.get_spike_counts()
print(spike_counts)
output_firing_rates = np.array(
    [value for (key, value) in sorted(spike_counts.items())])/sim_duration

sim.end()

# plot graph
plt.plot(input_firing_rates, output_firing_rates)
plt.xlabel("Input firing rate (spikes/second)")
plt.ylabel("Output firing rate (spikes/second)")
plt.savefig("simple_example.png")
plt.show()
