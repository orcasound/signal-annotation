import numpy as np
#import plotly.express as px
#import matplotlib.pyplot as plt
import soundfile as sf
import time
import math
"""
Here are user settings:
wavfile = "/home/val/PycharmProjects/AEproject/wavFiles/used/OS_9_12_2021_09_07_00_.wav"
g_channelchoice = -1      # if stereo, pick channel with higher amplitude
g_threshold = 7000        # max amplitude is 32k  (Integer samples) abs(wavform peak) must be above this threshold
g_zeroCrossMax_micro_s = 200   # micro seconds  -- don't extend click is zero-crossing is longer than this
g_maxClickWidth_micro_s = 1000 # micro seconds  -- don't extend click if total width is greater than this
g_minAmpFrac = 0.25       # smallest peak in a click must be greater than this fraction of maximum in the click

Scan wav file and determine values for:
peakList = []       #highest amplitude within click
freqList = []       #base frequency of click calculated from zero-crossing times
widthList = []      #width of click in seconds
wavIdx1List = []    #pointer to first sample in click
wavIdx2List = []    #pointer to last sample in click

"""
################################################################  USER PARAMETERS ####################
######################################################################################################


wavfile = "/home/val/PycharmProjects/AEproject/wavFiles/used/OS_9_12_2021_09_07_00_.wav"
wavfile="/home/val/PycharmProjects/AEproject/wavFiles/OS_9_12_2021_09_37_00_10min.wav"
g_channelchoice = -1    # if stereo, pick channel with higher amplitude
g_threshold = 7000   # max amplitude is 32k  (Integer samples)
g_zeroCrossMax_micro_s = 200   # micro seconds
g_maxClickWidth_micro_s = 1000 # micro seconds
g_minAmpFrac = 0.25  # smallest peak for click must be greater than this fraction of pkAmp

#########################################################################################################
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

def scanZerocrossing(wavfile):
    typedict = {}
    typedict['FLOAT'] = 'float32'
    typedict['PCM_16'] = 'int16'
    block_size = 4096 * 5   # idea here is to read a reasonable size and check whole block to see if any click might be present
    wav_idx = []            #  this can speed up getting through file
    delta_idx = []
    amp = []
    prior_amp = 0
    prior_idx = 0
    block_cnt = 0
    n_samples = 0
    runave = 0
    with sf.SoundFile(wavfile) as f:
        availableSamples = f.seek(0, sf.SEEK_END)
        f.seek(0)
        data = f.buffer_read(block_size, dtype=typedict[f.subtype])
        npdata = convertToNumpy(f, typedict, data)
        mn = np.mean(np.abs(npdata))
        std = np.std(np.abs(npdata))
        minAmpAboveBack = mn + 10*std   ## Note Bene this 10 could be made a user parameter
                                        # this is used to assist in zero-crossing detection
        print("Mean abs {}, Stdev of abs {}, Threshold for amplitude is {}".format(mn, std, minAmpAboveBack))
    with sf.SoundFile(wavfile) as f:
        availableSamples = f.seek(0, sf.SEEK_END)
        f.seek(0)

        while n_samples < availableSamples - block_size:
            data = f.buffer_read(block_size, dtype=typedict[f.subtype])
            npdata = convertToNumpy(f, typedict, data)
            n_samples += len(npdata)
            prior_zero_idx = 0
            for idx in range(len(npdata)):
                if (abs(npdata[idx] - runave) > minAmpAboveBack) and \
                        ((prior_amp<runave and npdata[idx]>=runave) or (prior_amp>runave and npdata[idx]<=runave)):
                    #got a zero crossing
                    wav_idx.append(idx + block_cnt * block_size)
                    prior_max = np.max(np.abs(npdata[prior_zero_idx:idx]))
                    amp.append(prior_max)
                    prior_zero_idx = idx
                prior_amp = npdata[idx]
            prior_amp = 0  # only lookiing for zero crossing INSIDE the buffer
            runave = np.mean(npdata)  # update the running ave for next buffer
            block_cnt += 1
            if block_cnt == 3:    # DEBUGGING
                break
    return np.array(wav_idx), np.array(delta_idx), np.array(amp)


def getNextZero(istart, data, dir):  # dir=1  forward in time  -1 backward
    i = istart
    while (i>0) and (i<len(data)-1) and ((data[i]>0 and data[i+dir]>0) or (data[i]<0 and data[i+dir]<0)) :
###        print(i, data[i], data[i+dir], data[i+2*dir], ((data[i]>0 and data[i+dir]>0) or (data[i]<0 and data[i+dir]<0)))
        i += dir
    i += dir
    if dir>0:
        try:
            maxPk = np.max(np.abs(data[istart:min(i+1, len(data))]))
        except:
            maxPk = 0
###        print("in getNextZero", dir, "istart", istart, i, data[istart:i + 1], abs(istart - i), "maxPk", maxPk)
    else:
        try:
            maxPk = np.max(np.abs(data[max(0, i-1):istart]))
###        print("in getNextZero", dir, "istart", istart, i, data[i-1:istart], abs(istart - i), "maxPk", maxPk)
        except:
            maxPk = 0

    if (i<=0) or (i>=len(data)) or (maxPk < g_threshold):
        return i, 99999   #signifies that no zero was found in this search
    return i+dir, abs(istart - i)    # return width to this next zero and max value found

def getNextPeak(wavIdx, npdata, idx):
    pk = 0
    pkidx = -1
    crossCnt = 0
    crossTot = 0
    idx1 = 0
    idx2 = 0
    while pk == 0 and idx < len(npdata)-2:
        if (abs(npdata[idx+1]) > g_threshold) and (abs(npdata[idx+1]) > abs(npdata[idx])) and (abs(npdata[idx+1]) > abs(npdata[idx+2])):
            pk = abs(npdata[idx+1])     # this is a local maximum above threshold
            pkidx = idx+1
            pkMinThresh = abs(pk)*g_minAmpFrac
        idx += 1
        if pk > 0:
            idx = pkidx  # accumulate post and prior zero crossings
            crossNext = 0
            crossCnt = 0
            crossTot = 0
            while crossNext < zeroCrossThreshold:
                idx, crossNext = getNextZero(idx, npdata, +1)
                if crossNext < zeroCrossThreshold:
                    crossCnt += 1
                    crossTot += crossNext
            idx2 = min(idx, len(npdata))
            idx = pkidx

            crossNext = 0
            while crossNext < zeroCrossThreshold:
                idx, crossNext = getNextZero(idx, npdata, -1)
                if crossNext < zeroCrossThreshold:
                    crossCnt += 1
                    crossTot += crossNext
            idx1 = max(idx,0)
    return pk, crossTot/(crossCnt+0.01),crossTot,idx1, idx2

def getNextClick(f, wavIdx):
    f.seek(wavIdx)
    block_size = samplerate//10
    block_size = 2000
    while True:
        if wavIdx + block_size > len(f):
            block_size = len(f) - wavIdx  # adjust block_size nearing end of wav file
            print("blocksize is", block_size, "wavIdx", wavIdx, "len(f)", len(f))

            if block_size == 0:
                return 0,0,0,0,0
        data = f.buffer_read(block_size, dtype=typedict[f.subtype])
        npdata = convertToNumpy(f, typedict, data)
        if np.max(np.abs(npdata)) >= g_threshold:
###            print("found peak above threshold, wavIdx", wavIdx)
            break
        wavIdx += len(npdata)
#        print("wavIdx is", wavIdx, len(npdata), block_size)
    if len(npdata) < 10:
        return -999     # no click available
    # we have at least one peak in this npdata buffer
    npdataIdx = 0
    peak, freq, width, npdataIdx1, npdataIdx2 = getNextPeak(wavIdx,npdata, npdataIdx)
###    print("peak", peak, freq, npdataIdx1, npdataIdx2, len(npdata))
    try:
        clickMax = np.max(np.abs(npdata[npdataIdx1:npdataIdx2]))
    except:
        clickMax = 0
###    print("getNextClick at",wavIdx, npdataIdx1, npdataIdx2, npdataIdx1+wavIdx, npdataIdx2+wavIdx)
###    print("-------peak ", clickMax, "width ", width)
    return clickMax, freq, width, npdataIdx1+wavIdx, npdataIdx2+wavIdx

########################################################################################################
########################################################################################################
typedict = {}
typedict['FLOAT'] = 'float32'
typedict['PCM_16'] = 'int16'
with sf.SoundFile(wavfile) as f:
    samplerate = f.samplerate
zeroCrossThreshold = int(samplerate * g_zeroCrossMax_micro_s / 1.0e6)  #samples
maxClickWidth = int(samplerate * g_maxClickWidth_micro_s / 1.0e6) #samples

peakList = []
freqList = []
widthList = []
wavIdx1List = []
wavIdx2List = []
wavIdx = 0
Done = False
tstart = time.time()
with sf.SoundFile(wavfile) as f:
    while not Done:
        peak, freq, width, idx1, idx2 = getNextClick(f, wavIdx)
###        print(">>>>>>>>>\npeak {}, freq {:0.2f}, width {}, idx1 {}, idx2 {} ".format(peak, freq, width, idx1, idx2 ))
        wavIdx = idx2 + 1
        if idx2 == 0:
            Done = True
###        print("update wavIdx to", wavIdx)
        if freq>0:
            peakList.append(peak)
            freqList.append(samplerate/(2*freq))
            widthList.append(width/samplerate)
            wavIdx1List.append(idx1)
            wavIdx2List.append(idx2)
            if (len(peakList) % 100) == 0:
                print(len(peakList), wavIdx)

elapsedTime = time.time()-tstart
print("Completed peak scan of wav file ", wavfile)
print("Total number of peaks is", len(peakList)," and scan of wav took {:0.2f} seconds".format(elapsedTime))
# for i in range(10):
#     print(i, peakList[i], freqList[i], widthList[i], wavIdx1List[i], wavIdx2List[i])
wav = wavfile.split("/")[-1]
peakFile= "peaks_{}_S_7000.csv".format(wav)
csvFile = open(peakFile, 'w')
header = "peak,freq,width,wavsec1,wavsec2,wavidx1,wavidx2\n"
csvFile.write(header)
for i in range(len(peakList)):
    aline = "{},{:0.0f},{:0.6f},{:0.6f},{:0.6f},{},{}\n".format(peakList[i], freqList[i], widthList[i], wavIdx1List[i]/samplerate, wavIdx2List[i]/samplerate, wavIdx1List[i], wavIdx2List[i])
    csvFile.write(aline)
csvFile.close()
print("Wrote peaks into file", peakFile)

