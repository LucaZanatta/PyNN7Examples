"""
Simplest model - array of neurons firing into a single memory neuron

File: simplest_model.py

"""
#!/usr/bin/python
import spynnaker7.pyNN as p
#import pyNN.spiNNaker as p
#import spynnaker_extra_pynn_models as q
import numpy, pylab, pickle
import os, sys
from pyNN.random import NumpyRNG, RandomDistribution
#import patternGenerator as pg
#import spikeTrains as st

numpy.random.seed(seed=1)
rng = NumpyRNG(seed=1)

timeStep = 0.2

p.setup(timestep=timeStep, min_delay = timeStep, max_delay = timeStep * 14)
p.set_number_of_neurons_per_core("IF_curr_exp", 150)
#p.set_number_of_neurons_per_core("IF_curr_comb_exp_2E2I", 150)
p.set_number_of_neurons_per_core("SpikeSourceArray", 6000)

nSourceNeurons = 100 # number of input (excitatory) neurons
nExcitNeurons  = 100 # number of excitatory neurons in the recurrent memory
sourcePartitionSz = 50 # Number of spike sources in a single projection
numPartitions = 1.0 * nSourceNeurons / sourcePartitionSz
if numPartitions != int(numPartitions):
   print "Invalid partition size! Exiting!!!"
   quit()
numPartitions = int(numPartitions)
nInhibNeurons  = 10  # number of inhibitory neurons in the recurrent memory
nTeachNeurons  = nExcitNeurons
nNoiseNeurons  = 10
ProbFiring     = 0.5 # was 0.05!
connProb       = 0.11   # was 11
myJitter       = 0.0   # was 0.25 # was 0.119
tolerance      = 2.5 # ms
#total_delay    = 2.0 # ms
#dendriticDelayFraction = 0.5
total_delay    = timeStep # ms
dendriticDelayFraction = 1.0

#!param
ProbFiring     = 0.05
myJitter       = 0.0 # was 0.25 # was 0.119
tolerance      = 2.5 # ms
#-----
delay_e2e      = 1.0 # was 0.8 # ms
pconn_e2e      = 0.1

delay_e2i      = 0.25 # ms
pconn_e2i      = 0.08
weight_e2i     = 0.12  # Working with no i2i when 0.12

delay_i2e      = 0.25 # ms
pconn_i2e      = 0.2  # 0.2 was too little
weight_i2e     = 0.15

pconn_i2i      = 0.24
delay_i2i      = 0.25 # ms
weight_i2i     = 0.15

# -------------------------------------------------------------
# Learning Parameters:
accDecayPerSecond      = 1.0
# Excitatory:
potentiationRateExcit  = 0.0 # 1.0 # SD! was 0.8
accPotThresholdExcit   = 20
depressionRateExcit    = 0.0 # was 0.11 # 0.0  # was 0.4
accDepThresholdExcit   = -18
meanPreWindowExcit     = 15.0 # 8
meanPostWindowExcit    = 1.0 # 8 
maxWeightExcit         = 1.80
minWeightExcit         = 0.00
# Excitatory2:
potentiationRateExcit2 = 0.0 # 1.0 # SD! was 0.8
accPotThresholdExcit2  = 2
depressionRateExcit2   = 0.0 # was 0.11 # 0.0  # was 0.4
accDepThresholdExcit2  = -8
meanPreWindowExcit2    = 15.0 # 8
meanPostWindowExcit2   = 1.0 # 8 
maxWeightExcit2        = 1.80
minWeightExcit2        = 0.00
# Inhibitory:
potentiationRateInhib  = 0.0
accPotThresholdInhib   = 5
depressionRateInhib    = 0.0
accDepThresholdInhib   = -5
meanPreWindowInhib     = 10.0
meanPostWindowInhib    = 10.0
maxWeightInhib         = 1.00  # was 0.1
minWeightInhib         = 0.00
# Inhibitory2:
potentiationRateInhib2 = 0.0
accPotThresholdInhib2  = 5
depressionRateInhib2   = 0.0
accDepThresholdInhib2  = -5
meanPreWindowInhib2    = 10.0
meanPostWindowInhib2   = 10.0
maxWeightInhib2        = 1.00  # was 0.1
minWeightInhib2        = 0.00
# -------------------------------------------------------------

baseline_excit_weight = 1.0

cell_params_lif_2E2I   = {'cm'        : 0.25, # nF was 0.25
                     'i_offset'  : 0.0,
                     'tau_m'     : 10.0,
                     'tau_refrac': 2.0,
                     'exc_a_tau' : 0.2,
                     'exc_b_tau' : 2.0,
                     'inh_a_tau' : 0.2,
                     'inh_b_tau' : 2.0,
                     'v_reset'   : -10.0,
                     'v_rest'    : 0.0,
                     'v_thresh'  : 20.0
                     }

runTime = 500
print "Run time is ", runTime

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Input stimulus

spikeStreams = list()
for i in range(nSourceNeurons):
   spikeStreams.append((40, 80, 120, 160, 200))
  
spikeArray = {'spike_times': spikeStreams}

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Construct Network

populations = list()
projections = list()


populations.append(p.Population(sourcePartitionSz, p.SpikeSourceArray, spikeArray, label="input"))             # 0

populations.append(p.Population(nExcitNeurons, p.extra_models.IF_curr_comb_exp_2E2I, cell_params_lif_2E2I, label='excit_pop'))  # 2

stdp_model = p.STDPMechanism(
     timing_dependence = p.extra_models.RecurrentRule( accum_decay = accDecayPerSecond,
            accum_dep_thresh_excit  = accDepThresholdExcit, accum_pot_thresh_excit  = accPotThresholdExcit,
               pre_window_tc_excit  = meanPreWindowExcit,     post_window_tc_excit  = meanPostWindowExcit,
            accum_dep_thresh_excit2 = accDepThresholdExcit2, accum_pot_thresh_excit2 = accPotThresholdExcit2,
               pre_window_tc_excit2 = meanPreWindowExcit2,     post_window_tc_excit2 = meanPostWindowExcit2,
            accum_dep_thresh_inhib  = accDepThresholdInhib, accum_pot_thresh_inhib  = accPotThresholdInhib,
               pre_window_tc_inhib  = meanPreWindowInhib,     post_window_tc_inhib  = meanPostWindowInhib,
            accum_dep_thresh_inhib2 = accDepThresholdInhib2, accum_pot_thresh_inhib2 = accPotThresholdInhib2,
               pre_window_tc_inhib2 = meanPreWindowInhib2,     post_window_tc_inhib2 = meanPostWindowInhib2),

     #weight_dependence = p.extra_models.WeightDependenceRecurrent(),
     weight_dependence = p.extra_models.WeightDependenceRecurrent(
       w_min_excit = minWeightExcit, w_max_excit = maxWeightExcit, A_plus_excit = potentiationRateExcit, A_minus_excit = depressionRateExcit,
       w_min_excit2 = minWeightExcit2, w_max_excit2 = maxWeightExcit2, A_plus_excit2 = potentiationRateExcit2, A_minus_excit2 = depressionRateExcit2,
       w_min_inhib = minWeightInhib, w_max_inhib = maxWeightInhib, A_plus_inhib = potentiationRateInhib, A_minus_inhib = depressionRateInhib,
       w_min_inhib2 = minWeightInhib2, w_max_inhib2 = maxWeightInhib2, A_plus_inhib2 = potentiationRateInhib2, A_minus_inhib2 = depressionRateInhib2),
     dendritic_delay_fraction = dendriticDelayFraction)


# Partition main projections into a number of sub-projections:
projections.append(p.Projection(populations[0], populations[1], p.AllToAllConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory2'))
#projections.append(p.Projection(populations[0], populations[1], p.AllToAllConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory2', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))

# XXXXXXXXXXXXXXXXXXXXX
# Run network

populations[0].record()
populations[1].record()
populations[1].record_gsyn()
populations[1].record_v()

p.run(runTime)

# XXXXXXXXXXXXXXXXXXXXXX
# Weight Statistics

if True: # (weight stats)
   count_plus = 0
   count_minus = 0
   count_zero = 0
   weightUse = {}
   final_weights = projections[0].getWeights(format="list")
   #for row in final_weights:
   for j in final_weights:
          myString="%f"%j
          #print "%f "%j
          if myString in weightUse:
              weightUse[myString] += 1
          else:
              weightUse[myString] = 1
          if j >= 1.0:
              count_plus += 1
              #print "Weight: ", j
          if j < 0.0:
              count_minus += 1
          if j == 0.0:
              count_zero += 1
   # Clear memory holding unneeded weight data:
   projections[0]._host_based_synapse_list = None

   print "High weights: ", count_plus
   print "Neg weights: ", count_minus
   print "Zero weights: ", count_zero
   print "Weight usage: ", weightUse
# End if False (weight stats)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Process output spikes against expected patterns

v = None
gsyn = None
spikes = None
teachSpikes = None

v = populations[1].get_v(compatible_output=True)
vgsyn = populations[1].get_gsyn(compatible_output=True)
stimSpikes =  populations[0].getSpikes(compatible_output=True)
spikes =      populations[1].getSpikes(compatible_output=True)

# Go through the output spikes and extract sections that should correspond to the
# individual patterns presented:

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Draw Plots

doPlots = True
if doPlots:
    if stimSpikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in stimSpikes], [i[0] for i in stimSpikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Stimulus Neurons')
        #pylab.show()
    else:
        print "No spikes received"

    if teachSpikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in teachSpikes], [i[0] for i in teachSpikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Teacher Neurons')
        #pylab.show()
    else:
        print "No spikes received"

    if spikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Memory Neurons (1st partition)')
        #pylab.show()
    else:
        print "No spikes received"

    ticks = len(v) / nExcitNeurons

    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('mV')
    pylab.title('Potential of neuron 1')
    for pos in range(1, nExcitNeurons, 35000):
        v_for_neuron = v[pos * ticks : (pos + 1) * ticks]
        pylab.plot([i[1] for i in v_for_neuron], 
                [i[2] for i in v_for_neuron])

    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('mV')
    pylab.title('Gsyn of neuron 1')
    for pos in range(1, nExcitNeurons, 35000):
        v_for_neuron = vgsyn[(pos+1) * ticks : (pos + 2) * ticks]
        pylab.plot([i[1] for i in v_for_neuron], 
                [i[2] for i in v_for_neuron])

    pylab.show()

p.end()

