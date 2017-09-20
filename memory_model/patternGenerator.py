import numpy, pylab, copy
from pyNN.random import NumpyRNG, RandomDistribution

class pattern(object):
    """
    """

    def __init__(self, totalNeurons, firing, cycleTime, numBins, rng, spikeTrain=False, jitterSD = 0.0, spuriousSpikeProb = 0.0):
        """
        """
        self.totalNeurons = totalNeurons
        self.firing = firing
        self.cycleTime = cycleTime
        self.jitterSD = jitterSD
        self.spuriousSpikeProb = spuriousSpikeProb
        self.numBins = numBins
        self.binSize = (1.0*cycleTime)/(1.0*numBins)
        self.events = list()
        self.binnedPattern = numpy.zeros( (totalNeurons, numBins+1) )
        if (firing > 0):
            self.firingFrac = (totalNeurons * 1.0)/firing
        # If there is no spike train, generate a pattern:
        if not spikeTrain:
           self.generateRandomPattern(rng)
           self.binPattern()
        else:
           # Extract binned values from the spike train:
           self.extractBinnedValues(spikeTrain)
        
    def generateRandomPattern(self, rng):
        """
        """
        unordered_events = list()
        rdNeurons = RandomDistribution('uniform', [0, self.totalNeurons-1], rng)
        rdTimes   = RandomDistribution('uniform', (0, self.cycleTime-1), rng)
        randomNeurons = rdNeurons.next(self.firing)
        randomTimes   = rdTimes.next(self.firing)
        for i in range(self.firing):
            unordered_events.append((int(randomNeurons[i]), int(randomTimes[i]*10)/10.0))
        #self.events = numpy.sort(unordered_events, 0)
        self.events = unordered_events

    def binPattern(self):
        """
        Generate a binned version of this pattern
        """
        for spikeEvent in self.events: 
           (neuron, spikeTime) = spikeEvent
           binNum = int(1.0*spikeTime/self.binSize)
           self.binnedPattern[neuron, binNum] += 1

    def printPattern(self):
        """
        """
        for el in self.events:
            print el

    def aboutPattern(self):
        """
        """
        percentFiring = self.firing * 1.0 / self.totalNeurons
        print "Neurons: %d, firing: %d (%.1f%%),  cycle time: %d ms" % (self.totalNeurons, self.firing, percentFiring, self.cycleTime)

    def displayPattern(self):
        """
        """
        firingFrac = 100.0 * self.firing / self.totalNeurons
        print "*************************"
        print "Total neurons: ", self.totalNeurons
        print "Firing: ", self.firing, "(", firingFrac, "%)"
        print "Cycle time: ", self.cycleTime, "ms"
        print self.events
        print ""

    def patternGraphicalView(self, pattNum):
        if self.events != None:
           pylab.figure()
           pylab.plot([i[1] for i in self.events], [i[0] for i in self.events], ".")
           pylab.xlabel('Time/ms')
           pylab.ylabel('spikes')
           pylab.title('Spikes of pattern %d' % pattNum)


    def compareSpikeTrains(self, observedSpikes, tolerance = 2.5, display=False):
        """
        Count the number of differences between this pattern and another.
        This pattern is considered to be the golden reference.
        This pattern is a pattern-formatted list, the observed spikes is a list of 
        [neuron, time] pairs in neuron-then-time order.
        """
        truePositives  = 0
        falsePositives = 0
        falseNegatives = 0
        myObservedSpikes = copy.deepcopy(observedSpikes)
        for pattSpike in self.events:
            pattNeuronID, pattTimeIndex = pattSpike
            # Is this spike present in the observed spikes:
            matched = False
            for spike in myObservedSpikes:
                neuronID, timeIndex = spike
                if pattNeuronID == neuronID:
                    if display:
                        print "Pattern: ", pattTimeIndex, ",  observed: ", timeIndex
                    ## Spike must be observed before its official time, but no more than windowSz before:
                    #if timeIndex <= pattTimeIndex and (timeIndex + windowSz) >=  pattTimeIndex:
                    # spike must be observed within a tolerance value of its intended time:
                    if pattTimeIndex >= (timeIndex - tolerance) and pattTimeIndex <= (timeIndex + tolerance):
                        #print "True pos @ time ", timeIndex, "teacher @ ", pattTimeIndex
                        # Spike is in correct position, but only count it once:
                        if not matched:
                            truePositives += 1
                            matched = True
                        # Remove this observed spike from the list so we don't use it again:
                        myObservedSpikes.remove(spike)
            # Checked against all the observed spikes. Did we find a match?
            if not matched:
                falseNegatives += 1 # Missing spike for this pattern - that's a false negative
                #print "False neg for teacher @ time ", pattTimeIndex
        # Checked all expected spikes against all observed spikes. Any left are false positives:
        falsePositives = len(myObservedSpikes)
        #print "False pos list: ", myObservedSpikes
        
        return truePositives, falsePositives, falseNegatives

    def extractBinnedValues(self, spikeTrain):
        """
        SpikeTrain is a list of neuron ID/spike time pairs. Extract them and place them in 
        the binning matrix.
        """
        for elem in spikeTrain:
            neuronID, timeStamp = elem
            binNum = int(timeStamp / self.binSize)
            if binNum > self.numBins:
               print "ERROR! in patternGenerator.extractBinnedValues() time stamp ", timeStamp,
               print "mapped to bin # ", binNum, " is too big (max ", self.numBins, ")"
               quit()
               
            if neuronID < 0 or neuronID >= self.totalNeurons:
               print "ERROR! in patternGenerator.extractBinnedValues(), neuronID ", neuronID,
               print "is not in expected range, (0 to ", self.totalNeurons, ")"
               quit()
            self.binnedPattern[neuronID, binNum] += 1

    def writePatternToFile(self, fileName):
        """
        """
        fullName = "./%s" % fileName
        fsock=open(fullName, 'w')
        for event in self.events:
           for elem in event:
              myString = "%f " % elem
              fsock.write("%s" % myString)

           fsock.write("\n")
        fsock.close()

    def writeBinsToFile(self, fileName):
        """
        """
        fullName = "./%s" % fileName
        fsock=open(fullName, 'w')
        for neuron in range(self.totalNeurons):
           header = "\n%d: " % neuron
           fsock.write("%s" % header)
           for binNum in range(self.numBins):
              if self.binnedPattern[neuron, binNum] == 1:
                 myString = "1"
              else:
                 myString = "0"
              fsock.write("%s" % myString)
           footer = "\n"
           fsock.write("%s" % footer)
        fsock.close()


class spikeStream(object):
    """
    """
    def __init__(self):
        """
        """
        self.streams = list()
        self.endTime = 0

    def buildStream(self, numSources=None, patterns=None, interPatternGap=10, rng=None, offset=0.0,order=None, noise=None, printTimes = False):
        """
        """
        # Establish a list of times at which pattern start firing:
        patternTimes = list()
        # Create empty streams, one per source neuron:
        for i in range(numSources):
             self.streams.append(list())

        # Go through order parameter, which is a list of the patterns to be appended.
        # For each one, append it.
        timePtr = 0
        jitterDistribution = RandomDistribution('normal', parameters = [0.0, patterns[0].jitterSD], rng = rng)
        for entry in order:
            if entry == -1:
                # Add blank entry:
                patternEntry = [entry, timePtr]
                patternTimes.append(patternEntry)
                # Create a gap (20ms):
                timePtr += 20
            else:
                if printTimes:
                    print "Pattern ", entry, " starts at time ", timePtr
                if entry >= len(patterns):
                    print "ERROR: Pattern set requested pattern ", entry, \
                          " and pattern set has only ", len(patterns), " patterns"
                    return -1
                elif entry < 0:
                    print "ERROR: Entry less than zero requested!"
                    return -1
                patternEntry = [entry, timePtr]
                patternTimes.append(patternEntry)
                pattern = patterns[entry]
                biggestTimestamp = 0
                for element in pattern.events:
                    index, timestamp = element
                    timestamp += offset
                    if patterns[0].jitterSD > 0:
                        newNum = jitterDistribution.next(1)
                        #print "Base T: ", timestamp, " : ", newNum,
                        timestamp += newNum
                        if timestamp < 0:
                           timestamp = 0.0
                        #print ", new: ", timestamp
                    biggestTimestamp = max(biggestTimestamp, timestamp)
                    self.streams[index].append(timePtr + timestamp)
                timePtr += pattern.cycleTime + interPatternGap
        self.endTime = timePtr
        return patternTimes

