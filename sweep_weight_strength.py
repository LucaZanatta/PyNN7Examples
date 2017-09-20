"""
sweep weight strength

File: sweep_weight_strength.py

"""
#!/usr/bin/python
import spynnaker7.pyNN as p
#import pyNN.spiNNaker as p
#import spynnaker_extra_pynn_models as q
import numpy, pylab, pickle
import math
import os, sys
from pyNN.random import NumpyRNG, RandomDistribution
#import patternGenerator as pg
#import spikeTrains as st

numpy.random.seed(seed=1)
rng = NumpyRNG(seed=1)

timeStep = 0.2

p.setup(timestep=timeStep, min_delay = timeStep, max_delay = timeStep * 14)
p.set_number_of_neurons_per_core("IF_curr_exp", 32)
#p.set_number_of_neurons_per_core("IF_curr_comb_exp_2E2I", 150)
p.set_number_of_neurons_per_core("SpikeSourceArray", 6000)

nSourceNeurons = 100 # number of input (excitatory) neurons
nExcitNeurons  = 100 # number of excitatory neurons in the recurrent memory
sourcePartitionSz = 100 # Number of spike sources in a single projection
numPartitions = 1.0 * nSourceNeurons / sourcePartitionSz
if numPartitions != int(numPartitions):
   print "Invalid partition size! Exiting!!!"
   quit()
numPartitions = int(numPartitions)
#nInhibNeurons  = 10  # number of inhibitory neurons in the recurrent memory
#nTeachNeurons  = nExcitNeurons
#nNoiseNeurons  = 10
#ProbFiring     = 0.5 # was 0.05!
#connProb       = 0.11   # was 11
#myJitter       = 0.0   # was 0.25 # was 0.119
#tolerance      = 2.5 # ms
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
accPotThresholdExcit2  = 20
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

if True:
   tau_refrac =  2.0
   win_length = 10.0
   tau_syn_e  =  0.5
   nSourceFiring = 0.05 * 32000
   nExcitFiring = 100
   patternCycleTime = 35;
   #--- How much charge in one spike with appropriate diff-of-Gaussians shape?
   T1 = 0.5; T2 = 0.224; a1 = 3.5; a2 = 3.5 # Gives Same shape and size as Kunkel, Diesman Alpha synapse

   taum_decay_compensation_factor = 2.00 # Failed at 1 and 1.5, 1.65, too much at 1.75, 2 and 3

   constant_stim_output_freq = 1000.0 /(tau_refrac + win_length)
   constant_stim_required_current_nA = 0.8 # From graph using constant_stim_output_freq)

   charge_req_over_window = constant_stim_required_current_nA * win_length * 1e-3

   #charge_from_unit_spike = 1e-9 * tau_syn_e * 1e-3
   charge_from_unit_spike = (a1*T1 - a2*T2) * 1e-3

   firing_in_window = nSourceFiring * win_length / patternCycleTime
   visible_firing_in_window = firing_in_window * pconn_e2e

   total_charge_from_visible_unit_spikes = charge_from_unit_spike * visible_firing_in_window

   scale_req_per_spike = charge_req_over_window / total_charge_from_visible_unit_spikes

   weight_max_per_conn = scale_req_per_spike
   maxWeightExcit = weight_max_per_conn * taum_decay_compensation_factor

   # Max_weight for 30K neurons: 0.18, for 40K neurons: 0.135
   print "Required constant stim current for firing: ", constant_stim_required_current_nA, " nA"
   print "Charge required over window: ", charge_req_over_window, " nC"
   print ""
   print "Charge from unit spike: ", charge_from_unit_spike, " nC"
   print ""
   print "Source firing: ", nSourceFiring
   print "Firing in window: ", firing_in_window
   print "Visible firing in window: ", visible_firing_in_window
   print "Total charge from visible firing: ", total_charge_from_visible_unit_spikes
   print "Required scaling per spike: ", scale_req_per_spike

   #print "Pattern cycle time: ", patternCycleTime

   #print "Source neurons: ", nSourceNeurons
   #print "Excit neurons: ", nExcitNeurons
   #print "Source firing: ", nSourceFiring
   #print "Excit firing: ", nExcitFiring
   #print "Jitter SD: ", myJitter
   #print "Pattern cycle time: ", patternCycleTime, "ms"
   #print "Num patterns: ", numPatterns
   #print "Num repeats during learning: ", numRepeats
   #print "Num repeats during recall: ", numRecallRepeats
   #print "Num partitions: ", numPartitions
   #print "Partition size: ", sourcePartitionSz
   #print
   #print "Stim current required %.4f nA" % (constant_stim_required_current_nA * 1e9)

#baseline_excit_weight = 0.0
baseline_excit_weight = scale_req_per_spike * visible_firing_in_window

iList = list()
for i in range(nSourceNeurons):
   iList.append(i*0.001+0.2)


cell_params_lif_2E2I   = {'cm'        : 0.25, # nF was 0.25
                     'i_offset'  : 0.0,
                     'tau_m'     : 10.0,
                     'tau_refrac': 0.5,
                     'exc_a_tau' : T2,   # Was 0.2
                     'exc_b_tau' : T1,   # Was 2.0
                     'inh_a_tau' : 0.6,
                     'inh_b_tau' : 3.0,
                     'v_reset'   : 0.0,
                     'v_rest'    : 0.0,
                     'v_thresh'  : 20.0
                     }

runTime = 100
print "Run time is ", runTime

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Input stimulus

populations = list()
projections = list()

rateStep = 0.01

for i in range(numPartitions):
    rates = list()
    for j in range(sourcePartitionSz):
       rates.append((i* numPartitions + j)*rateStep)
    arrayLabel = "ssPoisson%d" % i
    populations.append(p.Population(sourcePartitionSz, p.SpikeSourcePoisson, {'rate': rates}, label=arrayLabel))

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Construct Network

populations.append(p.Population(nExcitNeurons, p.extra_models.IF_curr_comb_exp_2E2I, cell_params_lif_2E2I, label='excit_pop'))  # 2
#populations.append(p.Population(nExcitNeurons, p.IF_curr_exp, {}, label='excit_pop'))  # 2

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
# projections.append(p.Projection(populations[0], populations[1], p.AllToAllConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory'))
# projections.append(p.Projection(populations[0], populations[1], p.OneToOneConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory2'))

# projections.append(p.Projection(populations[0], populations[1], p.AllToAllConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory2', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))
# projections.append(p.Projection(populations[0], populations[1], p.OneToOneConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))

# XXXXXXXXXXXXXXXXXXXXX
# Run network

populations[0].record()
populations[1].record()
#populations[1].record_gsyn()
#populations[1].record_v()

p.run(runTime)

# XXXXXXXXXXXXXXXXXXXXXX
# Weight Statistics

if True: # (weight stats)
   count_plus = 0
   count_minus = 0
   count_zero = 0
   weightUse = {}
   #final_weights = projections[0].getWeights(format="list")
   final_weights = projections[0].getWeights()
   print final_weights
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
vgsyn = None
spikes = None
teachSpikes = None

#v = populations[1].get_v(compatible_output=True)
#vgsyn = populations[1].get_gsyn(compatible_output=True)
stimSpikes =  populations[0].getSpikes(compatible_output=True)
spikes =      populations[1].getSpikes(compatible_output=True)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Extract spikes of excit neurons and calculate their rates

# Process output spikes:
counts = list()
for i in range(nExcitNeurons):
   counts.append([0, 1e6, 0])

if True:
   fsock=open("./sweep_input_rate_spikes_results.txt", 'a')
   myString = "Nidx  Input current (nA) Count   Mean Interval (ms)"
   fsock.write("%s\n" % myString)
   for i in range(numPartitions):
      spikes_in_single_partition = populations[1].getSpikes(compatible_output=True)
      totalSpikesThisBlock = 0
      for spike in spikes_in_single_partition:
          totalSpikesThisBlock += 1
          neuronid, timeIndex = spike
          entry = counts[int(neuronid)]
          entryCount, entryFirstSpikeTime, entryLastSpikeTime = entry
          entryCount += 1
          if timeIndex < entryFirstSpikeTime:
              entryFirstSpikeTime = timeIndex
          if timeIndex > entryLastSpikeTime:
              entryLastSpikeTime = timeIndex
          counts[int(neuronid)] = [entryCount, entryFirstSpikeTime, entryLastSpikeTime]
      # Calculate the mean rate for the entire block:
      meanRate = 1.0*totalSpikesThisBlock/sourcePartitionSz
      # Now calculate mean interval for each neuron and the variance against the
      # mean for the population:
      varianceTotal = 0
      for j in range(sourcePartitionSz):
         entry = counts[int(j)]
         myCount, firstSpikeTime, lastSpikeTime = entry
         if myCount > 1:
            meanInterval = (1.0*lastSpikeTime - 1.0*firstSpikeTime) / (1.0*myCount - 1.0)
            spikeRate = 1000.0/meanInterval
         else:
            meanInterval = 0.0
            spikeRate = 0.0
         varianceTotal += (spikeRate - meanRate)**2
         myString = "%d  %.3f       %d     %.3f" % (j, iList[j], myCount, meanInterval)
         fsock.write("%s\n" % myString)

      variance = varianceTotal/sourcePartitionSz
      stdDev = math.sqrt(variance)
      myString = "Mean rate: %.3f   Std.dev:  %.3f" % (meanRate, stdDev)
      fsock.write("%s\n" % myString)
   fsock.close()

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

    if v is not None:
       ticks = len(v) / nExcitNeurons
    else:
       ticks = 1

    if v is not None:
       pylab.figure()
       pylab.xlabel('Time/ms')
       pylab.ylabel('mV')
       pylab.title('Potential of neuron 1')
       for pos in range(1, nExcitNeurons, 35000):
           v_for_neuron = v[pos * ticks : (pos + 1) * ticks]
           pylab.plot([i[1] for i in v_for_neuron],
                   [i[2] for i in v_for_neuron])

    if vgsyn is not None:
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

