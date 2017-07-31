"""
inhib play

File: inhib_play.py

"""
#!/usr/bin/python
import spynnaker7.pyNN as p
#import spynnaker_extra_pynn_models as q
import numpy, pylab, pickle
import os, sys
from pyNN.random import NumpyRNG, RandomDistribution
import patternGenerator as pg
import spikeTrains as st

numpy.random.seed(seed=1)
rng = NumpyRNG(seed=1)

printWeightInfo = True
# Reset the board before starting:

timeStep = 0.250

fullyConnected = False
#fullyConnected = True
#withInhib = False
withInhib = True

p.setup(timestep=timeStep, min_delay = timeStep, max_delay = timeStep * 15)
p.set_number_of_neurons_per_core("IF_curr_exp", 20)
p.set_number_of_neurons_per_core("SpikeSourceArray", 500)
#!net
nSourceNeurons = 320 # number of input (excitatory) neurons
nExcitNeurons  = 1 # number of excitatory neurons in the recurrent memory
sourcePartitionSz = 4 # Was 500 # Number of spike sources in a single projection
numPartitions = 1.0 * nSourceNeurons / sourcePartitionSz
if numPartitions != int(numPartitions):
   print "Invalid partition size! Exiting!!!"
   quit()
numPartitions = int(numPartitions)
#nInhibNeurons  = nExcitNeurons/4  # number of inhibitory neurons in the recurrent memory
nInhibNeurons  = 200  # number of inhibitory neurons in the recurrent memory
nTeachNeurons  = nExcitNeurons
nNoiseNeurons  = 10
dendriticDelayFraction = 0.75
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
potentiationRateExcit = 0.0 # 1.0 # SD! was 0.8
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

windowSz = 10.0 # tolerance for errors during recall

baseline_excit_weight = 0.0 # fixed weight case = 0.4   # WAS 0.0!!!!
weight_to_force_firing = 18.0

tau_refrac =  2.0
win_length = 10.0
tau_syn_e  =  0.5
taum_decay_compensation_factor = 2.00 # Failed at 1 and 1.5, 1.65, too much at 1.75, 2 and 3

constant_stim_output_freq = 1000.0 /(tau_refrac + win_length)
constant_stim_required_current_nA = 0.594 * 1e-9 # From graph using constant_stim_output_freq)

charge_req_over_window = constant_stim_required_current_nA * win_length * 1e-3

charge_from_unit_spike = 1e-9 * tau_syn_e * 1e-3

firing_in_window = nSourceFiring * win_length / patternCycleTime
visible_firing_in_window = firing_in_window * pconn_e2e

total_charge_from_visible_unit_spikes = charge_from_unit_spike * visible_firing_in_window

scale_req_per_spike = charge_req_over_window / total_charge_from_visible_unit_spikes

weight_max_per_conn = scale_req_per_spike
maxWeightExcit = weight_max_per_conn * taum_decay_compensation_factor

# Max_weight for 30K neurons: 0.18, for 40K neurons: 0.135
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
print
print "Stim current required %.4f nA" % (constant_stim_required_current_nA * 1e9)
print "Charge req over entire window: %.4f pC" % (charge_req_over_window * 1e12)
print 
print "Charge from unit spike: %.4f pC" % (charge_from_unit_spike * 1e12)
print 
print "Firing in window: ", round(firing_in_window, 1)
print "Connection prob: ", pconn_e2e
print "Visible firing in window: ", round(visible_firing_in_window, 1)
print "total charge from visible unit spikes: %.4f pC" % (total_charge_from_visible_unit_spikes * 1e12)
print
print "Weight scaling required for firing: %.3f" % (scale_req_per_spike)
print

cell_params_lif   = {'cm'        : 0.25, # nF
                     'i_offset'  : 0.0,
                     'tau_m'     : 10.0,
                     'tau_refrac': tau_refrac,
                     'tau_syn_E' : tau_syn_e,
                     'tau_syn_I' : 0.5,
                     'v_reset'   : 0.0,  # was -10.0
                     'v_rest'    : 0.0,
                     'v_thresh'  : 15.0
                     }

iList = list()
for idx in range(nInhibNeurons):
   iList.append(idx*0.0010)   # was 0.0025

cell_params_inhib_lif   \
                  = {'cm'        : 0.25, # nF
                     'i_offset'  : iList,
                     'tau_m'     : 10.0,
                     'tau_refrac': 2.0,
                     'tau_syn_E' : 0.5,
                     'tau_syn_I' : 0.5,
                     'v_reset'   : -10.0,
                     'v_rest'    : 0.0,
                     'v_thresh'  : 15.0
                     }

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Construct pattern set
#!patts
inPatterns = list()
outPatterns = list()
for i in range(numPatterns):
   patt = pg.pattern(nSourceNeurons, nSourceFiring, patternCycleTime, numBins, rng, spikeTrain=False, jitterSD = myJitter, spuriousSpikeProb = 0.0)
   inPatterns.append(patt)
   patt = pg.pattern(nExcitNeurons, nExcitFiring, patternCycleTime, numBins, rng, spikeTrain=False, jitterSD = myJitter, spuriousSpikeProb = 0.0)
   patt.events = [(0, 20.0)]
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
        timeCount +=1

myStimulus=pg.spikeStream()
recordStartTime, patternTiming = myStimulus.buildStream(numSources=nSourceNeurons, patterns=inPatterns, interPatternGap=interPatternGap, offset=0.0, order=patternPresentationOrder)
#print "Pattern timings:"
#print patternTiming
teachingInput=pg.spikeStream()
teachingInput.buildStream(numSources=nExcitNeurons, patterns=outPatterns, interPatternGap=interPatternGap, offset=0.0, order=teachingOrder)

print "pattern timing:"
print patternTiming

runTime = myStimulus.endTime + 500
print "Run time is ", runTime, " ms"
print "Start recording at ", recordStartTime, " ms"

# Save network info:
netInfo= dict()
netInfo['sourceNeurons']= nSourceNeurons
netInfo['excitNeurons']= nExcitNeurons
netInfo['probFiring']= ProbFiring
netInfo['connProb']= pconn_e2e
netInfo['jitter']= myJitter
netInfo['tolerance']= tolerance
netInfo['totalDelay'] = delay_e2e
netInfo['dendriticFrac']= dendriticDelayFraction
netInfo['cycleTime']= patternCycleTime
netInfo['numPatterns']= numPatterns
netInfo['numRepeats']= numRepeats
netInfo['numRecallRepeats']= numRecallRepeats
netInfo['runTime']= runTime
netInfo['potRate']= potentiationRateExcit
netInfo['depRate']= depressionRateExcit
netInfo['potThresh']= accPotThresholdExcit
netInfo['depThresh']= accDepThresholdExcit
netInfo['meanPreWindow']= meanPreWindowExcit
netInfo['meanPostWindow']= meanPostWindowExcit
netInfo['maxWeight']= maxWeightExcit
netInfo['minWeight']= minWeightExcit
#dirName = "./myResults/patts_only_excit_%d" % numPatterns
dirName = "./myResults/inhib_play_%d" % numPatterns
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

if withInhib:
   stimulus = 0
   excit  = numPartitions
   teacher = numPartitions + 1
   inhib = numPartitions + 2
   inhibNoise  = numPartitions + 3
else:
   stimulus = 0
   excit  = numPartitions
   teacher = numPartitions + 1

teachingSpikeArray = {'spike_times': teachingInput.streams}

#!pop
for i in range(numPartitions):
    arrayLabel = "ssArray%d" % i
    startIdx = i * sourcePartitionSz
    endIdx = startIdx + sourcePartitionSz
    streamSubset = myStimulus.streams[startIdx:endIdx]
    spikeArray = {'spike_times': streamSubset}
    populations.append(p.Population(sourcePartitionSz, p.SpikeSourceArray, spikeArray, label=arrayLabel))              # 0

populations.append(p.Population(nExcitNeurons, p.IF_curr_exp, cell_params_lif, label='excit_pop'))                     # 1
populations.append(p.Population(nTeachNeurons, p.SpikeSourceArray, teachingSpikeArray, label='teaching_ss_array'))     # 2
if withInhib:
   populations.append(p.Population(nInhibNeurons, p.IF_curr_exp, cell_params_inhib_lif, label='inhib_pop'))                  # 3
   for n in range(nInhibNoiseSrcs):
      populations.append(p.Population(nInhibNeurons, p.SpikeSourcePoisson, {'rate': noiseRate }, label="inhib_noise_%d"%n))# 4-4+n

# potentiationRate = 0.8
# accPotThreshold = 8 
# depressionRate = 0.11
# accDepThreshold = -99
# meanPreWindow  = 18.0
# meanPostWindow = 19.0

stdp_model = p.STDPMechanism( 
     timing_dependence = p.extra_models.RecurrentRule( accum_decay = accDecayPerSecond,
            accum_dep_thresh_excit = accDepThresholdExcit, accum_pot_thresh_excit = accPotThresholdExcit, 
               pre_window_tc_excit = meanPreWindowExcit,     post_window_tc_excit = meanPostWindowExcit, 
            accum_dep_thresh_inhib = accDepThresholdInhib, accum_pot_thresh_inhib = accPotThresholdInhib, 
               pre_window_tc_inhib = meanPreWindowInhib,     post_window_tc_inhib = meanPostWindowInhib),

     weight_dependence = p.extra_models.WeightDependenceRecurrent( 
       w_min_excit = minWeightExcit, w_max_excit = maxWeightExcit, A_plus_excit = potentiationRateExcit, A_minus_excit = depressionRateExcit,
       w_min_inhib = minWeightInhib, w_max_inhib = maxWeightInhib, A_plus_inhib = potentiationRateInhib, A_minus_inhib = depressionRateInhib), 

     mad = True, dendritic_delay_fraction = dendriticDelayFraction)

# Partition main projections into a number of sub-projections:
#!proj
for i in range(numPartitions):
   if fullyConnected:
      projections.append(p.Projection(populations[i], populations[excit], p.AllToAllConnector(weights=baseline_excit_weight, delays=delay_e2e), target='excitatory', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))
   else:
      projections.append(p.Projection(populations[i], populations[excit], p.FixedProbabilityConnector(p_connect=pconn_e2e, weights=baseline_excit_weight, delays=delay_e2e), target='excitatory', synapse_dynamics=p.SynapseDynamics(slow=stdp_model)))

#if withInhib:
if False:
   for i in range(numPartitions):
       projections.append(p.Projection(populations[i], populations[inhib], p.FixedProbabilityConnector(p_connect=pconn_e2i, weights=weight_e2i, delays=delay_e2i), target='excitatory'))

if withInhib:
   # Recurrent inhib connections:
##   projections.append(p.Projection(populations[inhib], populations[inhib], p.FixedProbabilityConnector(p_connect=pconn_i2i, weights=weight_i2i, delays=delay_i2i), target='inhibitory'))
   # Inhib forward to excit connections:
   #projections.append(p.Projection(populations[inhib], populations[excit], p.FixedProbabilityConnector(p_connect=pconn_i2e, weights=baseline_excit_weight, delays=delay_i2e), target='inhibitory'))
   for n in range(nInhibNoiseSrcs):
      projections.append(p.Projection(populations[inhibNoise+n],  populations[inhib], p.OneToOneConnector(weights=weightInhibNoise, delays=delayInhibNoise), target='excitatory'))

# Teaching input to excit:
projections.append(p.Projection(populations[teacher], populations[excit], p.OneToOneConnector(weights=weight_to_force_firing, delays=timeStep), target='excitatory'))

# XXXXXXXXXXXXXXXXXXXXX
# Run network

#!rec
#populations[stimulus].record()
#populations[excit].record_v()
#populations[excit].record(schedule=[(recordStartTime, runTime)])
populations[excit].record()
populations[teacher].record()
if withInhib:
   populations[inhib].record()
   populations[inhib].record_v()

offset = 5
p.run(runTime)

# build dictionary of spike times by neuron-ID:
spkDict = dict()
for el in inPatterns[0].events:
  nid, spkTime = el
  spkDict[nid] = spkTime

wList = list()
if True:
   print os.system('date')
   print "Weight processing:"

   # XXXXXXXXXXXXXXXXXXXXXX
   # Weight Statistics

   if True: # (weight stats)
      count_plus = 0
      count_minus = 0
      weightUse = {}
      totalWeights = 0
      synapseCount = 0
      wDict = dict()
      for i in range(numPartitions):
          final_weights = projections[i].getWeights(format="array")
          for row in final_weights:
              rowIndex = 0
              #partCount = 0
              for j in row:
                 wDict[synapseCount] = j
                 myString="%f"%j
                 totalWeights += 1
                 if myString in weightUse:
                     weightUse[myString] += 1
                 else:
                     weightUse[myString] = 1
                 if j > baseline_excit_weight:
                     count_plus += 1
                     #partCount += 1
                     wList.append([synapseCount, j])
                 if j <= baseline_excit_weight:
                     count_minus += 1
                 synapseCount += 1
           # Clear memory holding unneeded weight data:
          projections[i]._host_based_synapse_list = None

      #print "Weight list:\n", wList
      print "Total weights: ", totalWeights, "  High weights: ", count_plus, "   Low weights: ", count_minus
      print "Weight usage: ", weightUse
      # perPatternUsage = (count_plus*1.0)/(nSourceFiring * numPatterns)
   # End if False (weight stats)

#if printWeightInfo:
#    for el in wList:
#       nid, w = el
#       if nid in spkDict.keys():
#          print "NID: %d, T: %f, W: %f" % (nid, spkDict[nid], w)
#       else:
#          print "NID: %d, NO SPIKE, W: %f" % (nid, w)


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Process output spikes against expected patterns

v = None
gsyn = None
stimSpikes = None
spikes = None
inhibSpikes = None
teachSpikes = None

#v = populations[excit].get_v(compatible_output=True)
v = populations[inhib].get_v(compatible_output=True)
#stimSpikes =  populations[stimulus].getSpikes(compatible_output=True)
spikes =      populations[excit].getSpikes(compatible_output=True)
teachSpikes = populations[teacher].getSpikes(compatible_output=True)
if withInhib:
   inhibSpikes = populations[inhib].getSpikes(compatible_output=True)

# Go through the output spikes and extract sections that should correspond to the
# individual patterns presented:

numpy.save(dirName+"/outputSpikesFile", spikes)

print os.system('date')

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Draw Plots
#!plot
doPlots = True
#doPlots = False
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

    if spikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".") 
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Excitatory Neurons')
        #pylab.show()
    else:
        print "No spikes received"

    if inhibSpikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in inhibSpikes], [i[0] for i in inhibSpikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Inhibitory Neurons')
        #pylab.show()
    else:
        print "No spikes received"

    if teachSpikes != None:
        pylab.figure()
        pylab.plot([i[1] for i in teachSpikes], [i[0] for i in teachSpikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title('Spikes of Teaching Neurons')
        # pylab.show()
    else:
        print "No teaching spikes received"

    #ticks = len(v) / nExcitNeurons
    ticks = len(v) / nInhibNeurons
    
    # Excitatory neuron current:
    if v != None:
        pylab.figure()
        pylab.xlabel('Time/ms')
        pylab.ylabel('mV')
        pylab.title('Potential of inhib neuron 0')
        for pos in range(0, nInhibNeurons, 1000):
            v_for_neuron = v[pos * ticks : (pos + 1) * ticks]
            pylab.plot([i[1] for i in v_for_neuron], 
                    [i[2] for i in v_for_neuron])
        pylab.figure()
        pylab.xlabel('Time/ms')
        pylab.ylabel('mV')
        pylab.title('Potential of inhib neuron 100')
        for pos in range(100, nInhibNeurons, 1000):
            v_for_neuron = v[pos * ticks : (pos + 1) * ticks]
            pylab.plot([i[1] for i in v_for_neuron], 
                    [i[2] for i in v_for_neuron])
        pylab.figure()
        pylab.xlabel('Time/ms')
        pylab.ylabel('mV')
        pylab.title('Potential of inhib neuron 20')
        for pos in range(20, nInhibNeurons, 1000):
            v_for_neuron = v[pos * ticks : (pos + 1) * ticks]
            pylab.plot([i[1] for i in v_for_neuron], 
                    [i[2] for i in v_for_neuron])
        pylab.figure()
        pylab.xlabel('Time/ms')
        pylab.ylabel('mV')
        pylab.title('Potential of inhib neuron 80')
        for pos in range(80, nInhibNeurons, 1000):
            v_for_neuron = v[pos * ticks : (pos + 1) * ticks]
            pylab.plot([i[1] for i in v_for_neuron], 
                    [i[2] for i in v_for_neuron])

inhibList = list()
if inhibSpikes != None:
   count = numpy.zeros(nInhibNeurons)
   for i in inhibSpikes:
      nid, spkTime = i
      count[nid] += 1
   open("spikecounts.txt", "w")
   for el in range(ninhibNeurons):
      theString = "%.2f \n" % (count[el]);
      pointsFile.write(theString)
   pointsFile.write("\n")
   pointsFile.close()

if False:
   pointsFile = open("potTraceInhib_0.txt", "w");
   for el in v:
      theString = "%.2f %.2f  %.2f\n" % (el[0], el[1], el[2]);
      pointsFile.write(theString)
      pointsFile.write("\n")
   pointsFile.close()

pylab.show()

p.end()

