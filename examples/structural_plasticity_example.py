"""
Simple test for STDP :

   Reproduces a classical plasticity experiment of plasticity induction by
pre/post synaptic pairing specifically :

 * At the begining of the simulation, "n_stim_test" external stimulations of
   the "pre_pop" (presynaptic) population do not trigger activity in the
   "post_pop" (postsynaptic) population.

 * Then the presynaptic and postsynaptic populations are stimulated together
   "n_stim_pairing" times by an external source so that the "post_pop"
   population spikes 10ms after the "pre_pop" population.

 * Ater that period, only the "pre_pop" population is externally stimulated
   "n_stim_test" times, but now it should trigger activity in the "post_pop"
   population (due to STDP learning)

Run as :

   $ ./stdp_example

This example requires that the NeuroTools package is installed
(http://neuralensemble.org/trac/NeuroTools)

Authors : Catherine Wacongne < catherine.waco@gmail.com >

          Xavier Lagorce < Xavier.Lagorce@crans.org >

April 2013
"""


import numpy as np
import pylab

from pacman.model.constraints.partitioner_constraints.partitioner_maximum_size_constraint import \
    PartitionerMaximumSizeConstraint
from pacman.model.constraints.placer_constraints.placer_chip_and_core_constraint import PlacerChipAndCoreConstraint

import spynnaker7.pyNN as sim

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
# sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp
sim.set_number_of_neurons_per_core("IF_curr_exp", 100)
cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }

# Other simulation parameters
e_rate = 80
in_rate = 300

n_stim_test = 5
n_stim_pairing = 20
dur_stim = 20

pop_size = 14**2
# pop_size=7**2

ISI = 90.
start_test_pre_pairing = 200.
start_pairing = 1500.
start_test_post_pairing = 700.

simtime = (start_pairing + start_test_post_pairing
           + ISI * (n_stim_pairing + n_stim_test) + 550.)

# Initialisations of the different types of populations
IAddPre = []
IAddPost = []

# +-------------------------------------------------------------------+
# | Creation of neuron populations                                    |
# +-------------------------------------------------------------------+

# Neuron populations
pre_pop = sim.Population(pop_size, model, cell_params, label="PRE_SYN_POP")
post_pop = sim.Population(pop_size, model, cell_params, label="STRUCTURAL_POP")

# Test of the effect of activity of the pre_pop population on the post_pop
# population prior to the "pairing" protocol : only pre_pop is stimulated
for i in range(n_stim_test):
    IAddPre.append(sim.Population(pop_size,
                                  sim.SpikeSourcePoisson,
                                  {'rate': in_rate,
                                   'start': start_test_pre_pairing + ISI * (i),
                                   'duration': dur_stim
                                   }))

# Pairing protocol : pre_pop and post_pop are stimulated with a 10 ms
# difference
for i in range(n_stim_pairing):
    IAddPre.append(sim.Population(pop_size,
                                  sim.SpikeSourcePoisson,
                                  {'rate': in_rate,
                                   'start': start_pairing + ISI * (i),
                                   'duration': dur_stim
                                   }))
    IAddPost.append(sim.Population(pop_size,
                                   sim.SpikeSourcePoisson,
                                   {'rate': in_rate,
                                    'start': start_pairing + ISI * (i) + 10.,
                                    'duration': dur_stim
                                    }))

# Test post pairing : only pre_pop is stimulated (and should trigger activity
# in Post)
for i in range(n_stim_test):
    IAddPre.append(sim.Population(pop_size,
                                  sim.SpikeSourcePoisson,
                                  {'rate': in_rate,
                                   'start': (start_pairing
                                             + ISI * (n_stim_pairing)
                                             + start_test_post_pairing
                                             + ISI * (i)),
                                   'duration': dur_stim
                                   }))

# Noise inputs
INoisePre = sim.Population(pop_size,
                           sim.SpikeSourcePoisson,
                           {'rate': e_rate, 'start': 0, 'duration': simtime},
                           label="expoisson")
INoisePost = sim.Population(pop_size,
                            sim.SpikeSourcePoisson,
                            {'rate': e_rate, 'start': 0, 'duration': simtime},
                            label="expoisson")

# +-------------------------------------------------------------------+
# | Creation of connections                                           |
# +-------------------------------------------------------------------+

# Connection parameters
JEE = 3.

# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector(weights=JEE * 0.05)

# Noise projections
sim.Projection(INoisePre, pre_pop, ee_connector, target='excitatory')
sim.Projection(INoisePost, post_pop, ee_connector, target='excitatory')

# Additional Inputs projections
for i in range(len(IAddPre)):
    sim.Projection(IAddPre[i], pre_pop, ee_connector, target='excitatory')
for i in range(len(IAddPost)):
    sim.Projection(IAddPost[i], post_pop, ee_connector, target='excitatory')

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9,
                                                   A_plus=0.02, A_minus=0.02)
)

structure_model_w_stdp = sim.StructuralMechanism(stdp_model=stdp_model, weight=0.2, s_max=32,grid=[np.sqrt(pop_size), np.sqrt(pop_size)])
# structure_model_w_stdp = sim.StructuralMechanism(weight=.1, s_max=32, grid=[pop_size, 1])

plastic_projection = sim.Projection(
    # pre_pop, post_pop, sim.FixedNumberPreConnector(32),
    pre_pop, post_pop, sim.FixedProbabilityConnector(0.1),  # TODO what about starting from 0?
    synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
    label="plastic_projection"
)


# plastic_projection2 = sim.Projection(
#     # pre_pop, post_pop, sim.FixedNumberPreConnector(32),
#     post_pop, post_pop, sim.FixedNumberPostConnector(1),  # TODO what about starting from 0?
#     synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
#     label="plastic_projection"
# )

# Recurrent connection (checking if this is working)
# sim.Projection(
#     # pre_pop, post_pop, sim.FixedNumberPreConnector(32),
#     post_pop, post_pop, sim.FixedNumberPreConnector(32), # TODO what about starting from 0?
#     synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
#     label="plastic_projection"
# )

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
pre_pop.record_v()
post_pop.record_v()

# Record spikes
pre_pop.record()
post_pop.record()

# Run simulation
sim.run(simtime)

print("Weights:", plastic_projection.getWeights())


def plot_spikes(spikes, title):
    if spikes is not None:
        pylab.figure()
        pylab.xlim((0, simtime))
        pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title(title)

    else:
        print "No spikes received"

weights = plastic_projection._get_synaptic_data(False, 'weight')

pre_spikes = pre_pop.getSpikes(compatible_output=True)
post_spikes = post_pop.getSpikes(compatible_output=True)
# End simulation on SpiNNaker
sim.end()
np.savez("structural_results_stdp", pre_spike=pre_spikes, post_spikes=post_spikes)

plot_spikes(pre_spikes, "pre-synaptic")
plot_spikes(post_spikes, "post-synaptic")
pylab.show()



f, ax1= pylab.subplots(1, 1, figsize=(8, 8))
i = ax1.matshow(np.nan_to_num(weights.reshape(pop_size, pop_size)))
ax1.grid(visible=False)
ax1.set_title("Feedforward connectivity matrix", fontsize=16)
cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
cbar = f.colorbar(i, cax=cbar_ax)
cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
pylab.show()
