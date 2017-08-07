"""
Simple Associative Memory

File: partitioned_single_network.py

"""
#!/usr/bin/python
import spynnaker7.pyNN as p
#import pyNN.spiNNaker as p
#import spynnaker_extra_pynn_models as q
import numpy, pylab, pickle
import os, sys
from pyNN.random import NumpyRNG, RandomDistribution
import patternGenerator as pg
import spikeTrains as st

numpy.random.seed(seed=1)
rng = NumpyRNG(seed=1)

# Reset the board before starting:

timeStep = 0.2

backgroundNoise = False
noiseSpikeFreq = 7    # Hz per neuron
fullyConnected = False

p.setup(timestep=timeStep, min_delay = timeStep, max_delay = timeStep * 14)
p.set_number_of_neurons_per_core("IF_curr_exp", 150)
#p.set_number_of_neurons_per_core("IF_curr_comb_exp_2E2I", 150)
p.set_number_of_neurons_per_core("SpikeSourceArray", 6000)

nSourceNeurons = 500 # number of input (excitatory) neurons
nExcitNeurons  = 500 # number of excitatory neurons in the recurrent memory
sourcePartitionSz = 50 # Number of spike sources in a single projection
numPartitions = 1.0 * nSourceNeurons / sourcePartitionSz
if numPartitions != int(numPartitions):
   print "Invalid partition size! Exiting!!!"
   quit()
numPartitions = int(numPartitions)
nInhibNeurons  = 10  # number of inhibitory neurons in the recurrent memory
nTeachNeurons  = nExcitNeurons
nNoiseNeurons  = 10
ProbFiring     = 0.05
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

#---
nInhibNoiseSrcs  = 40
noiseRate        = 100
delayInhibNoise  = 0.25
weightInhibNoise = 0.09  # was 0.02
#---
nSourceFiring  = int(nSourceNeurons * ProbFiring)
nExcitFiring   = int(nExcitNeurons * ProbFiring)

patternCycleTime = 35
numPatterns = int(sys.argv[1])
numRepeats  = 8 # was 8
numRecallRepeats  = 2
binSize = 4
numBins = patternCycleTime/binSize
interPatternGap = 0    # was 10

# -------------------------------------------------------------
# Learning Parameters:
accDecayPerSecond     = 1.0
# Excitatory
potentiationRateExcit = 1.0 # 1.0 # SD! was 0.8
accPotThresholdExcit  = 4
depressionRateExcit   = 0.0 # was 0.11 # 0.0  # was 0.4
accDepThresholdExcit  = -4
meanPreWindowExcit    = 18.0 # 8
meanPostWindowExcit   = 10.0 # 8 
maxWeightExcit        = 0.05
minWeightExcit        = 0.00
# Inhibitory:
potentiationRateInhib = 0.0
accPotThresholdInhib  = 15
depressionRateInhib   = 0.0
accDepThresholdInhib  = -15
meanPreWindowInhib    = 10.0
meanPostWindowInhib   = 10.0
maxWeightInhib        = 0.05  # was 0.1
minWeightInhib        = 0.00
# -------------------------------------------------------------

nSourceFiring  = int(nSourceNeurons * ProbFiring)
nExcitFiring   = int(nExcitNeurons * ProbFiring)

patternCycleTime = 35
numPatterns = int(sys.argv[1]) 
numRepeats  = 15 # was 8
numRecallRepeats  = 1
binSize = 4
numBins = patternCycleTime/binSize
interPatternGap = 0    # was 10
potentiationRate = 0.80
accPotThreshold = 5 
depressionRate = 0.40  # was 0.66
accDepThreshold = -5 
meanPostWindow = 8.0

windowSz = 10.0 # tolerance for errors during recall

baseline_excit_weight = 0.0
weight_to_force_firing = 18.0
# Max_weight for 30K neurons: 0.18, for 40K neurons: 0.135
max_weight = 0.6 # was 0.25          # 0.8               # Max weight! was 0.66 in best run so far
min_weight = 0.0
print "Pattern cycle time: ", patternCycleTime

print "Source neurons: ", nSourceNeurons
print "Excit neurons: ", nExcitNeurons
print "Source firing: ", nSourceFiring
print "Excit firing: ", nExcitFiring
print "Jitter SD: ", myJitter
print "Pattern cycle time: ", patternCycleTime, "ms"
print "Num patterns: ", numPatterns
print "Num repeats during learning: ", numRepeats
print "Num repeats during recall: ", numRecallRepeats
print "Num partitions: ", numPartitions
print "Partition size: ", sourcePartitionSz

cell_params_lif   = {'cm'        : 0.25, # nF was 0.25
                     'i_offset'  : 0.0,
                     'tau_m'     : 10.0,
                     'tau_refrac': 2.0,
                     'tau_syn_E' : 0.5,
                     'tau_syn_I' : 0.5,
                     'v_reset'   : -10.0,
                     'v_rest'    : 0.0,
                     'v_thresh'  : 20.0
                     }

cell_params_lif_2E2I   = {'cm'        : 0.25, # nF was 0.25
                     'i_offset'  : 0.0,
                     'tau_m'     : 10.0,
                     'tau_refrac': 2.0,
                     'exc_a_tau' : 2.0,
                     'exc_b_tau' : 0.2,
                     'inh_a_tau' : 2.0,
                     'inh_b_tau' : 0.2,
                     'v_reset'   : -10.0,
                     'v_rest'    : 0.0,
                     'v_thresh'  : 20.0
                     }

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Construct pattern set

inPatterns = list()
outPatterns = list()
for i in range(numPatterns):
   patt = pg.pattern(nSourceNeurons, nSourceFiring, patternCycleTime, numBins, rng, spikeTrain=False, jitterSD = myJitter, spuriousSpikeProb = 0.0)
   #patt.events=[]
   inPatterns.append(patt)
   patt = pg.pattern(nExcitNeurons, nExcitFiring, patternCycleTime, numBins, rng, spikeTrain=False, jitterSD = myJitter, spuriousSpikeProb = 0.0)
   #patt.events=[]    # Force output to be just noise....
   outPatterns.append(patt)

timeCount = 0
patternPresentationOrder = list()
patternPresentationOrder.append(-1)
teachingOrder = list()
teachingOrder.append(-1)
# Teaching phase:
for patt in range(numPatterns):
    for rpt in range(numRepeats):
        patternPresentationOrder.append(patt)
        teachingOrder.append(patt)
        timeCount +=1
# Gap:
patternPresentationOrder.append(-1)
patternPresentationOrder.append(-1)
timeCount +=2
# Recall phase:
for patt in range(numPatterns):
    for rpt in range(numRecallRepeats):
        patternPresentationOrder.append(patt)
        patternPresentationOrder.append(patt)
        timeCount +=1

myStimulus=pg.spikeStream()
patternTiming = myStimulus.buildStream(numSources=nSourceNeurons, patterns=inPatterns, interPatternGap=interPatternGap, offset=0.0, order=patternPresentationOrder)

teachingInput=pg.spikeStream()
teachingInput.buildStream(numSources=nExcitNeurons, patterns=outPatterns, interPatternGap=interPatternGap, offset=-0.5, order=teachingOrder)

runTime = myStimulus.endTime + 500
print "Run time is ", runTime

# Add background spurious spikes if requested:
if backgroundNoise:
   myStimulus.addNoiseToStreams(rng=rng, numNeurons = nSourceNeurons, noiseFrequency=noiseSpikeFreq, startTime = 0, endTime = runTime, timeStep = timeStep)

#print myStimulus.streams[1]

# Save network info:
netInfo= dict()
netInfo['sourceNeurons']    = nSourceNeurons
netInfo['excitNeurons']     = nExcitNeurons
netInfo['probFiring']       = ProbFiring
netInfo['backgroundNoise']  = backgroundNoise
netInfo['noiseSpikeFreq']   = noiseSpikeFreq
netInfo['fullyConnected']   = fullyConnected
netInfo['connProb']         = connProb
netInfo['jitter']           = myJitter
netInfo['tolerance']        = tolerance
netInfo['totalDelay']       = total_delay
netInfo['dendriticFrac']    = dendriticDelayFraction
netInfo['cycleTime']        = patternCycleTime
netInfo['numPatterns']      = numPatterns
netInfo['numRepeats']       = numRepeats
netInfo['numRecallRepeats'] = numRecallRepeats
netInfo['runTime']          = runTime
netInfo['potRate']          = potentiationRate
netInfo['depRate']          = depressionRate
netInfo['potThresh']        = accPotThreshold
netInfo['depThresh']        = accDepThreshold
netInfo['meanPreWindowExcit']  = meanPreWindowExcit
netInfo['meanPostWindowExcit'] = meanPostWindowExcit
netInfo['maxWeight']        = max_weight
netInfo['minWeight']        = min_weight
dirName = "./myResults/patts_%d" % numPatterns
os.system("mkdir %s" % dirName)
numpy.save(dirName+"/neuronParams", cell_params_lif)
with open(dirName+"/networkParams", "wb") as outfile:
    pickle.dump(netInfo, outfile, protocol=pickle.HIGHEST_PROTOCOL)
with open(dirName+"/inputPatterns", "wb") as outfile:
    pickle.dump(inPatterns, outfile, protocol=pickle.HIGHEST_PROTOCOL)
with open(dirName+"/outputPatterns", "wb") as outfile:
    pickle.dump(outPatterns, outfile, protocol=pickle.HIGHEST_PROTOCOL)
with open(dirName+"/patternTiming", "wb") as outfile:
    pickle.dump(patternTiming, outfile, protocol=pickle.HIGHEST_PROTOCOL)

#with open("mydict", "rb") as outfile:
#    a=pickle.load(outfile)


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Construct Network

populations = list()
projections = list()

stimulus = 0
inhib = numPartitions
excit  = numPartitions + 1
teacher = numPartitions + 2

teachingSpikeArray = {'spike_times': teachingInput.streams}

for i in range(numPartitions):
    arrayLabel = "ssArray%d" % i
    startIdx = i * sourcePartitionSz
    endIdx = startIdx + sourcePartitionSz
    streamSubset = myStimulus.streams[startIdx:endIdx]
    spikeArray = {'spike_times': streamSubset}
    populations.append(p.Population(sourcePartitionSz, p.SpikeSourceArray, spikeArray, label=arrayLabel))             # 0

populations.append(p.Population(nInhibNeurons, p.IF_curr_exp, cell_params_lif, label='inhib_pop'))                 # 1
populations.append(p.Population(nExcitNeurons, p.extra_models.IF_curr_comb_exp_2E2I, cell_params_lif_2E2I, label='excit_pop'))  # 2
populations.append(p.Population(nTeachNeurons, p.SpikeSourceArray, teachingSpikeArray, label='teaching_ss_array')) # 3

stdp_model = p.STDPMechanism(
     timing_dependence = p.extra_models.RecurrentRule( accum_decay = accDecayPerSecond,
            accum_dep_thresh_excit = accDepThresholdExcit, accum_pot_thresh_excit = accPotThresholdExcit,
               pre_window_tc_excit = meanPreWindowExcit,     post_window_tc_excit = meanPostWindowExcit,
            accum_dep_thresh_inhib = accDepThresholdInhib, accum_pot_thresh_inhib = accPotThresholdInhib,
               pre_window_tc_inhib = meanPreWindowInhib,     post_window_tc_inhib = meanPostWindowInhib),

     weight_dependence = p.extra_models.WeightDependenceRecurrent(
       w_min_excit = minWeightExcit, w_max_excit = maxWeightExcit, A_plus_excit = potentiationRateExcit, A_minus_excit = depressionRateExcit,
       w_min_inhib = minWeightInhib, w_max_inhib = maxWeightInhib, A_plus_inhib = potentiationRateInhib, A_minus_inhib = depressionRateInhib),

        dendritic_delay_fraction = dendriticDelayFraction)

#stdp_model = p.STDPMechanism( timing_dependence = q.RecurrentRule(accumulator_depression = accDepThreshold, accumulator_potentiation = accPotThreshold, mean_pre_window = meanPreWindow, mean_post_window = meanPostWindow, mean_inhib_pre_window = 12.0, dual_fsm=True), weight_dependence = p.MultiplicativeWeightDependence(w_min=min_weight, w_max=max_weight, A_plus=potentiationRate, A_minus=depressionRate, w_min_inhib = 0.1), mad=True, dendritic_delay_fraction = dendriticDelayFraction)

# Partition main projections into a number of sub-projections:
for i in range(numPartitions):
   if fullyConnected:
      projections.append(p.Projection(populations[i], populations[excit], p.AllToAllConnector(weights=baseline_excit_weight, delays=total_delay), target='excitatory', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))
   else:
      projections.append(p.Projection(populations[i], populations[excit], p.FixedProbabilityConnector(p_connect=connProb, weights=baseline_excit_weight, delays=total_delay), target='excitatory', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))

projections.append(p.Projection(populations[teacher], populations[excit], p.OneToOneConnector(weights=weight_to_force_firing, delays=timeStep), target='excitatory'))

# XXXXXXXXXXXXXXXXXXXXX
# Run network

populations[stimulus].record()
populations[excit].record()

p.run(1000)
#p.run(runTime)

# XXXXXXXXXXXXXXXXXXXXXX
# Weight Statistics

if True: # (weight stats)
   count_plus = 0
   count_minus = 0
   weightUse = {}
   for i in range(numPartitions):
       final_weights = projections[i].getWeights(format="array")
       for row in final_weights:
           partCount = 0
           for j in row:
              myString="%f"%j
              #print "%f "%j
              if myString in weightUse:
                  weightUse[myString] += 1
              else:
                  weightUse[myString] = 1
              if j > 0.0:
                  count_plus += 1
                  partCount += 1
              if j <= 0.0:
                  count_minus += 1
           #print "%d " % partCount
           #print "\n"
       # Clear memory holding unneeded weight data:
       projections[i]._host_based_synapse_list = None

   print "High weights: ", count_plus
   print "Low weights: ", count_minus
   print "Weight usage: ", weightUse
   perPatternUsage = (count_plus*1.0)/(nSourceFiring * numPatterns)
# End if False (weight stats)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Process output spikes against expected patterns

v = None
gsyn = None
spikes = None

#v = populations[excit].get_v(compatible_output=True)
stimSpikes =  populations[stimulus].getSpikes(compatible_output=True)
spikes =      populations[excit].getSpikes(compatible_output=True)

# Go through the output spikes and extract sections that should correspond to the
# individual patterns presented:

numpy.save(dirName+"/outputSpikesFile", spikes)

os.system('date')

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Draw Plots

doPlots = True
if doPlots:
    if spikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Memory Neurons (1st partition)')
        #pylab.show()
    else:
        print "No spikes received"

    pylab.show()

p.end()

