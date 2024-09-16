import numpy as np
#import plotly.express as px
#import matplotlib.pyplot as plt
import soundfile as sf
import time
import math
"""
Here are user settings:
wavfile = "/home/val/PycharmProjects/AEproject/wavFiles/used/OS_9_12_2021_09_07_00_.wav"
click detection parameters:
g_channelchoice = -1      # if stereo, pick channel with higher amplitude
g_threshold = 7000        # max amplitude is 32k  (Integer samples) abs(wavform peak) must be above this threshold
g_zeroCrossMax_micro_s = 200   # micro seconds  -- don't extend click is zero-crossing is longer than this
g_maxClickWidth_micro_s = 1000 # micro seconds  -- don't extend click if total width is greater than this

bout classification parameters:
g_maxBoutClickGap = 2  # if click comes in more that this number of seconds after previous click in bout, start a new bout
g_minClicksInBout = 4
g_maxClickGapSecs = 2   # click lists and bouts end when next click comes in at least this much later than previous click in list

Scan wav file in blocks of wav data (g_block_size):
    Detect a click
    Build a list of clicks
    Group clicks when possible into bouts and classify them
    Eliminate clicks that can't be grouped
    Move completed bouts to final bout list

"""
################################################################  USER PARAMETERS ####################
######################################################################################################

# wavfile = "/home/val/PycharmProjects/AEproject/wavFiles/used/OS_9_12_2021_09_07_00_.wav"
# wavfile="/home/val/PycharmProjects/AEproject/wavFiles/used/OS_9_12_2021_09_37_00_.wav"
# wavfile="/home/val/PycharmProjects/AEproject/wavFiles/OS_9_12_2021_09_37_00_Tst.wav"

wavfile = "../clickFile/OS_9_12_2021_09_37_00__1min.wav"  ###  a very short file for test purposes

g_block_size = 2000
## Peak triggering parameters
g_channelchoice = -1    # if stereo, pick channel with higher amplitude
g_threshold = 6000   # max amplitude is 32k  (Integer samples)
g_zeroCrossMax_micro_s = 200   # micro second - maximum time between click zero crossings
g_maxClickWidth_micro_s = 1000 # micro seconds - maximum time between beginning and end of click

## Bout classification parameters
g_maxBoutClickGap = 2  # if click comes in more that this number of seconds after previous click in bout, start a new bout
g_minClicksInBout = 4
g_maxClickGapSecs = 2   # click lists and bouts end when next click comes in at least this much later than previous click in list

#classification  buzz, slow, fast based on interclick intervals
g_maxBuzzGapSecs = 0.03
g_maxFastGapSecs = 0.35
g_maxSlowGapSecs = 0.75
#########################################################################################################
################################################################
class click():
    def __init__(self, peak, freq, width, wavSec1, wavSec2, idx1, idx2, wavfile):
        self.peak = peak
        self.freq = freq
        self.width = width   # width and other values along time axis are in seconds
        self.wavSec1 = wavSec1
        self.wavSec2 = wavSec2
        self.idx1 = idx1
        self.idx2 = idx2
        self.wavfile = wavfile
        self.gap = 0   # this will be set to the time to previous click as bout develops

class bout():
    def __init__(self, samplerate, clickList):
        self.samplerate = samplerate
        self.clickList = clickList
        self.Nclicks = len(clickList)   # could just use len(self.clickList) here
        self.boutIdx1 = clickList[0].idx1
        self.boutIdx2 = clickList[-1].idx2
        self.gapList = []  # we will initialize this when first gap comes in
        self.gapAve = 0   #gaps are in seconds
        self.gapStd = 10    # these will be updated as clickList develops
        self.peakAve = 0.0
        self.peakStd = 0.0
        self.freqAve = 0.0
        self.freqStd = 0.0
        self.widthAve = 0.0  #widths are in seconds
        self.widthStd = 0.0
        self.boutConstancy = 0
        self.boutClass = 'unclassified'
        self.gapConstancy = 0
        self.peakConstancy = 0
        self.freqConstancy = 0
        self.widthConstancy = 0
        self.updateBoutStats()

    def doStats(self, target):
        values  = []
        for click in self.clickList:
            if target == 'peak': values.append(click.peak)
            if target == 'freq': values.append(click.freq)
            if target == 'width': values.append(click.width)
        ave = np.mean(values)
        std = max(np.std(values), ave/1000)  # this sets floor on std > 0
        return ave, std

    def updateBoutStats(self):

        self.peakAve, self.peakStd = self.doStats('peak')
        self.freqAve, self.freqStd = self.doStats('freq')
        self.widthAve, self.widthStd = self.doStats('width')
        self.Nclicks = len(self.clickList)
        self.boutIdx2 = self.clickList[-1].idx2
        self.gapList = []   #  Note Bene  is there any reason to keep this list in bout object??
        for i in range(1, self.Nclicks):
            self.gapList.append((self.clickList[i].idx1 - self.clickList[i-1].idx2)/self.samplerate)
        self.gapAve = np.mean(self.gapList)
        self.gapStd = max(np.std(self.gapList), self.gapAve/100)
        self.doClassify()

    def doClassify(self):
        # calculate constancies
        self.boutClass = 'unclassified'
        if self.Nclicks >= g_minClicksInBout:
            if self.gapAve < g_maxBuzzGapSecs:
                self.boutClass = "Buzz"
            else:
                if self.gapAve < g_maxFastGapSecs:
                    self.boutClass = "Fast"
                else:
                    if self.gapAve < g_maxSlowGapSecs:
                        self.boutClass = "Slow"
                    else:
                        self.boutClass = "Unclassified"
            self.gapConstancy = self.gapAve / self.gapStd
            self.peakConstancy = self.peakAve / self.peakStd
            self.freqConstancy = self.freqAve / self.freqStd
            self.widthConstancy = self.widthAve / self.widthStd
            self.boutConstancy = (self.gapConstancy + self.peakConstancy + self.freqConstancy + self.widthConstancy) / 4.0



#########################################################################################################
def convertToNumpy(f, typedict, data):  # select channel in wav file and convert data to numpy array
    #  -1 to pick channel with higher amplitude
    global g_channelchoice
    if f.channels == 2:
        if g_channelchoice == -1:
            ch0 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[0::2]))
            ch1 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[1::2]))
            if ch0 > ch1:
                g_channelchoice = 0
            else:
                g_channelchoice = 1
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])[g_channelchoice::2]
    else:
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])
    return npdata

def getNextZero(istart, data, dir):  # dir=1  forward in time  -1 backward
    i = istart
    while (i>0) and (i<len(data)-1) and ((data[i]>0 and data[i+dir]>0) or (data[i]<0 and data[i+dir]<0)) :
###        print(i, data[i], data[i+dir], data[i+2*dir], ((data[i]>0 and data[i+dir]>0) or (data[i]<0 and data[i+dir]<0)))
        i += dir
    i += dir  # just after zero crossing
    if dir>0:
        try:
            maxPk = np.max(np.abs(data[istart:min(i+1, len(data))]))
###            print("in getNextZero", dir, "istart", istart, i, data[i-1:istart], abs(istart - i), "maxPk", maxPk)
        except:
            maxPk = 0
###        print("in getNextZero", dir, "istart", istart, i, data[istart:i + 1], abs(istart - i), "maxPk", maxPk)
    else:
        try:
            maxPk = np.max(np.abs(data[max(0, i-1):istart]))
###            print("in getNextZero", dir, "istart", istart, i, data[i-1:istart], abs(istart - i), "maxPk", maxPk)
        except:
            maxPk = 0
    if (i<=0) or (i>=len(data)):
        return i, 9999, -9999   #signifies that no zero was found in this search
    return i, abs(istart - i), maxPk    # return index of zero  and width to this next zero and max value found in between

def constructClick(npdata, wavIdx, direction, wavFile): # peak is in the center of the npdata array

    idx = len(npdata)//2  # accumulate post and prior zero crossings
    crossNext = 0
    crossCnt = 0
    crossTot = 0
    lowCnt = 0
    while crossNext <= zeroCrossMax_samples and lowCnt < 4:
        idx, crossNext, maxNext = getNextZero(idx, npdata, +1)  # look forward in time
 ###       print("+++",idx, idx + wavIdx, crossNext, maxNext, lowCnt)
        if maxNext < g_threshold:
            lowCnt += 1
        else:
            lowCnt = 0
        if crossNext < zeroCrossMax_samples:
            crossCnt += 1
            crossTot += crossNext
    idx2 = idx        # this is the end of the click
    idx = len(npdata)//2
    crossNext = 0
    lowCnt = 0
    while crossNext < zeroCrossMax_samples and lowCnt < 3:
        priorIdx = idx
        idx, crossNext, maxNext = getNextZero(idx, npdata, -1)  # look backward in time
###        print("---", idx, idx + wavIdx, crossNext, maxNext, lowCnt)
        if maxNext < g_threshold:
            lowCnt += 1
        else:
            lowCnt = 0
        if crossNext < zeroCrossMax_samples:
            crossCnt += 1
            crossTot += crossNext
    if crossTot == 0:
        return None
    idx1 = max(priorIdx, 0) #this is the beginning of the click
    aveZeroCross = crossTot / (crossCnt + 0.01)
    try:
        clickMax = np.max(np.abs(npdata[idx1:idx2]))
    except:
        clickMax = 0
    # click object peak, freq (sec), width (sec), wavSec1 (sec), wavSec2 (sec), idx1, idx2, wavfile)
    idx1 += wavIdx  # point click edges to the original wav file indices
    idx2 += wavIdx
    return click(clickMax, g_samplerate/(2*aveZeroCross), (idx2-idx1)/g_samplerate, idx1/g_samplerate, idx2/g_samplerate, idx1, idx2, wavFile)

def getNextClick(f, wavIdx, threshold, direction):
    if direction == 1:
        f.seek(wavIdx)
        while True:
            if wavIdx + g_block_size > len(f):
                block_size = len(f) - wavIdx  # adjust block_size nearing end of wav file
###                print("blocksize is", block_size, "wavIdx", wavIdx, "len(f)", len(f))
                if block_size == 0:
                    return None
            data = f.buffer_read(g_block_size, dtype=typedict[f.subtype])
            npdata = convertToNumpy(f, typedict, data)
            imax = np.argmax(np.abs(npdata))
            val = abs(npdata[imax])
            if val >= threshold:
###                print("found peak in buffer above threshold, wavIdx", wavIdx, imax)
                #recenter buffer on this peak
                wavIdx = wavIdx + imax - len(npdata)//2
                f.seek(wavIdx)
                data = f.buffer_read(g_block_size, dtype=typedict[f.subtype])
                npdata = convertToNumpy(f, typedict, data)
                break
            wavIdx = wavIdx + len(npdata)
            #print("in getNextCick_ 2 wavIdx is", wavIdx, len(npdata), g_block_size)
    if len(npdata) < 10:
        return None   # no click available
    # we have at least one peak in this npdata buffer
    wavFile = f.name
    nextClick = constructClick(npdata, wavIdx, wavFile, direction)
    return nextClick

def getGapList(cList):
    gapList = []
    for i in range(1, len(cList)):
        gapList.append(cList[i].idx1 - cList[i - 1].idx2)  # g_maxClickGap_samples
    gapMean = np.mean(gapList)
    gapStd = max(np.std(gapList), gapMean/100)
    # if gapList[0]>gapMean + 3*gapStd:
    #     print(gapList)
    #     print(cList[0].idx1, cList[0].idx2)
    #     cList.pop(0)
    #     print(cList[0].idx1, cList[0].idx2)
    #     print("-------------")
    #     getGapList(cList)   ### N.B . should have length test here
    # gapMean = np.mean(gapList)
    # gapStd = max(np.std(gapList), gapMean/100)
    gapRatio = gapMean/gapStd
    return gapMean, gapStd, gapRatio

def buildBouts(thisClick, clickList, bouts_open):
    bestBout = None
    if len(bouts_open) > 0:
        # try to append thisClick to one of the bouts_open
        minNewGap = 9e9
        for about in bouts_open:
            gapMean, gapStd, gapRatio = getGapList(about.clickList)
            newGap = thisClick.idx1 - about.clickList[-1].idx2
###            print("----------",int(gapMean-4*gapStd), newGap, int(gapMean+4*gapStd), thisClick.idx1,about.clickList[-1].idx2)
            if (newGap > gapMean - 4 * gapStd) and (newGap < gapMean + 4 * gapStd) and (newGap < minNewGap):
                bestBout = about
                minNewGap = newGap

###        print(thisClick.idx1, "minNewGap is", minNewGap, bestBout)
        if bestBout != None:
###            print("prior idx", bestBout.clickList[-1].idx2)
            bestBout.clickList.append(thisClick)
            bestBout.updateBoutStats()

###            gapMean, gapStd, gapRatio = getGapList(bestBout.clickList)
###            print("\nrecalculated gap stats are", int(gapMean), int(gapStd), gapRatio, "len bestBout clicklist is", len(bestBout.clickList), "thisClick idx1", thisClick.idx1)
            return [], bouts_open
    if (len(bouts_open) == 0) or (bestBout == None):
        if len(clickList) < 4:   # start building up a new initial click of 4 clicks
            clickList.append(thisClick)
            if len(clickList) == 4:
                #see if first click should be dropped because gap is Too Large
                gapList = []
                for i in range(1, len(clickList)):
                    gapList.append(clickList[i].idx1 - clickList[i - 1].idx2)
                if DEBUG == 1:
                    print("gapList", gapList)
                if gapList[0] > 2.5* (gapList[1]+gapList[2])/2: #  N.B.  why 2.5???
                    clickList.pop(0)
            return clickList, bouts_open
        else:
            clickList.append(thisClick)

            gapMean, gapStd, gapRatio = getGapList(clickList)

###            print('gap ratio', gapRatio, len(clickList))
            # if gapRatio < 2:
            #     clickList.pop(0)   # drop the earliest click in the list
            #     return clickList, bouts_open
            newBout = bout(g_samplerate, clickList)
            bouts_open.append(newBout)
            print("starting a new bout at", newBout.clickList[0].idx1, newBout.clickList[-1].idx2)
            return [], bouts_open

def checkBoutCompletion(idx1, openBoutList, finalBoutList):

    for bout in openBoutList:
        finalGap = idx1 - bout.clickList[-1].idx2
        if finalGap > g_maxClickGap_samples:
            # move bout to finalBoutList
            bout.doClassify()
            finalBoutList.append(bout)
            openBoutList.remove(bout)
            print("............checkBout...move bout {} to finalBoutList {}\n".format(bout, finalBoutList))

########################################################################################################
########################################################################################################
typedict = {}
typedict['FLOAT'] = 'float32'
typedict['PCM_16'] = 'int16'
with sf.SoundFile(wavfile) as f:
    g_samplerate = f.samplerate
zeroCrossMax_samples = int(g_samplerate * g_zeroCrossMax_micro_s / 1.0e6)  #samples
maxClickWidth_samples = int(g_samplerate * g_maxClickWidth_micro_s / 1.0e6) #samples
g_maxClickGap_samples = int(g_samplerate * g_maxClickGapSecs)
##############  BOUT variables
finalBoutList = []       # these are completed bouts where a bout is a group of related clicks
openBoutList = []   # these are bouts that are being formed but are not yet closed
#########################################
DEBUG = 0
clickList = []
wavIdx = 0
Done = False
tstart = time.time()
direction = 1
threshold = g_threshold
gettingBout = False
with sf.SoundFile(wavfile) as f:
    while not Done:
        thisClick = getNextClick(f, wavIdx, threshold, direction)  # find next click above threshold in specified direction
        if thisClick == None:
            wavIdx += g_block_size
            if wavIdx > len(f) - g_block_size:   # Finished with wav file
                Done = True
        else:
            if DEBUG == 1:
                print("idx=", wavIdx, thisClick.idx1, "jump wavIdx=", thisClick.idx1 - wavIdx)
            checkBoutCompletion(thisClick.idx1, openBoutList, finalBoutList)
            clickList, openBoutList = buildBouts(thisClick, clickList, openBoutList)
            if direction == 1:
                wavIdx = thisClick.idx2 + 1  # set wav file pointer to end of thisClick

###            print("Next wavIdx", wavIdx, "thisClick", thisClick)
        # if len(finalBoutList) >= 25:
        #     Done = True


elapsedTime = time.time()-tstart
print("\nCompleted bout scan of wav file: ", wavfile)
print("Total number of bouts is", len(finalBoutList),"and scan of wav took {:0.2f} seconds".format(elapsedTime))
wav = wavfile.split("/")[-1]
boutFile= "../bouts/bouts_{}_S_threshold_{}.csv".format(wav, g_threshold)
csvFile = open(boutFile, 'w')
header = "Cntr,Nclicks,Class,wavIdx1,wavIdx2,boutConstancy,gapConstancy,peakConstancy,freqConstancy,widthConstancy,meanGap,meanPeak,meanFreq,meanWidth\n"
csvFile.write(header)
cntr = 1
for bout in finalBoutList:
    aline = "{},{},{},{},{},{:0.1f},{:0.1f},{:0.1f},{:0.1f},{:0.1f},{:0.6f},{:0.1f},{:0.1f},{:0.6f}\n".\
        format(cntr, bout.Nclicks, bout.boutClass, bout.boutIdx1, bout.boutIdx2, bout.boutConstancy, bout.gapConstancy,bout.peakConstancy, \
               bout.freqConstancy, bout.widthConstancy, bout.gapAve, bout.peakAve, bout.freqAve, bout.widthAve)
    csvFile.write(aline)
    cntr += 1
csvFile.close()
print("Wrote {} bouts into file", len(finalBoutList), boutFile)

