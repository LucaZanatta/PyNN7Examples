"""
Single Neuron Test

File: single_neuron_test.py

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

numpy.random.seed(seed=101)
rng = NumpyRNG(seed=100)

timeStep = 0.2

p.setup(timestep=timeStep, min_delay = timeStep, max_delay = timeStep * 14)
p.set_number_of_neurons_per_core("IF_curr_exp", 20)
#p.set_number_of_neurons_per_core("IF_curr_comb_exp_2E2I", 150)
p.set_number_of_neurons_per_core("SpikeSourceArray", 6000)

nSourceNeurons = 1 # number of input (excitatory) neurons
nExcitNeurons  = 1 # number of excitatory neurons in the recurrent memory
numExcitBlocks = 1
sourcePartitionSz = 1 # Number of spike sources in a single projection
numPartitions = 1
if numPartitions != int(numPartitions):
   print "Invalid partition size! Exiting!!!"
   quit()
numPartitions = int(numPartitions)
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

T1 = 0.5; T2 = 0.224

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

runTime = 50
print "Run time is ", runTime

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Input stimulus

populations = list()
projections = list()

rateStep = 0.01

iList = list()
for i in range(sourcePartitionSz):
   iList.append(7.0)

for i in range(numPartitions):
    rates = list()
    for j in range(sourcePartitionSz):
       rates.append((i* numPartitions + j)*rateStep)
    arrayLabel = "ssPoisson%d" % i
    populations.append(p.Population(sourcePartitionSz, p.SpikeSourcePoisson, {'rate': 7.0}, label=arrayLabel))
    excit = i + 1

label = "excit_pop%d" % i
populations.append(p.Population(nExcitNeurons, p.extra_models.IF_curr_comb_exp_2E2I, cell_params_lif_2E2I, label=label))  # 2

projections.append(p.Projection(populations[0], populations[1], p.FixedProbabilityConnector(p_connect=0.1, weights=0.02, delays=0.1), target='excitatory'))

# XXXXXXXXXXXXXXXXXXXXX
# Run network

populations[0].record()
populations[1].record()

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
inhibSpikes = None
teachSpikes = None

#v = populations[1].get_v(compatible_output=True)
#vgsyn = populations[1].get_gsyn(compatible_output=True)
stimSpikes  = populations[0].getSpikes(compatible_output=True)
#spikes      = populations[excit].getSpikes(compatible_output=True)

print "Mean stimulus spike rate: ", len(stimSpikes)*(1000.0/runTime)/nSourceNeurons

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Extract spikes of excit neurons and calculate their rates

# Process output spikes:

if True:
   meanStats = list()
   fsock=open("./scan_weight_spike_rates.txt", 'a')
   #myString = "         Nidx  Weight Count   Mean Interval (ms)"
   #fsock.write("%s\n" % myString)
   myString = "BlkIdx  Weight"
   fsock.write("%s\n" % myString)
   for i in range(1):
      counts = list()
      for j in range(nExcitNeurons):
         counts.append([0, 1e6, 0])
      spikes_in_single_partition = populations[numPartitions+i].getSpikes(compatible_output=True)
      spikesLen = len(spikes_in_single_partition)
      print "spikes: %d, Mean spike rate: %f" % (spikesLen, spikesLen*(1000.0/runTime))
      if ((spikes_in_single_partition != None)):
         pylab.figure()
         pylab.plot([i1[1] for i1 in spikes_in_single_partition], [i1[0] for i1 in spikes_in_single_partition], ".")
         pylab.xlabel('Time/ms')
         pylab.ylabel('spikes')
         titleString = "Spikes of Excit Neurons, block %s" %(i)
         pylab.title(titleString)
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
      for j in range(nExcitNeurons):
         entry = counts[int(j)]
         myCount, firstSpikeTime, lastSpikeTime = entry
         if myCount > 1:
            meanInterval = runTime / (1.0*myCount)
            spikeRate = 1000.0/meanInterval
         else:
            meanInterval = 0.0
            spikeRate = 0.0
         varianceTotal += (spikeRate - meanRate)**2
         #myString = "              %d     %d     %.3f" % (j, myCount, meanInterval)
         #fsock.write("%s\n" % myString)

      variance = varianceTotal/sourcePartitionSz
      stdDev = math.sqrt(variance)
      meanStats.append([meanRate, stdDev])
   # Print men statistics at the end:
   fsock.close()
   pylab.show()

#p.end()

