import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import soundfile as sf
import math

def convertToNumpy(f, typedict, data, channelchoice):
    #  -1 to pick channel with higher amplitude
    if f.channels == 2:
        if channelchoice == -1:
            ch0 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[0::2]))
            ch1 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[1::2]))
            if ch0 > ch1:
                channelchoice = 0
            else:
                channelchoice = 1
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])[channelchoice::2]
    else:
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])
    return npdata


def scanZerocrossing(wavfile):
    channelchoice = -1    # pick channel with higher amplitude
    typedict = {}
    typedict['FLOAT'] = 'float32'
    typedict['PCM_16'] = 'int16'
    block_size = 4096 * 5
    minAmpAboveBack = 10000
    wav_idx = []
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
        npdata = convertToNumpy(f, typedict, data, channelchoice)
        mn = np.mean(np.abs(npdata))
        std = np.std(np.abs(npdata))
        minAmpAboveBack = mn + 10*std
        print("Mean abs {}, Stdev of abs {}, Threshold for amplitude is {}".format(mn, std, minAmpAboveBack))
    with sf.SoundFile(wavfile) as f:
        availableSamples = f.seek(0, sf.SEEK_END)
        f.seek(0)

        while n_samples < availableSamples - block_size:
            data = f.buffer_read(block_size, dtype=typedict[f.subtype])
            npdata = convertToNumpy(f, typedict, data, channelchoice)
            n_samples += len(npdata)
            prior_zero_idx = 0
            for idx in range(len(npdata)):
                if (abs(npdata[idx] - runave) > minAmpAboveBack) and ((prior_amp<runave and npdata[idx]>=runave) or (prior_amp>runave and npdata[idx]<=runave)):
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

def getNextPeak(npdata, idx, threshold):
    pk = 0
    pkidx = -1
    while pk == 0 and idx < len(npdata)-2:
        # if idx > 6640:
        #     print(idx, npdata[idx])
        if (npdata[idx+1] > threshold) and (npdata[idx+1] > npdata[idx]) and (npdata[idx+1] > npdata[idx+2]):
            pk = npdata[idx+1]
            pkidx = idx+1
        idx += 1
        if pk > 0:
            idx = pkidx
            # print("pk=",pk, "pkidx", pkidx)
            # for i in range(10):
            #     print(pkidx + i - 5, npdata[pkidx + i - 5])
            break
    if idx == len(npdata)-2:
        idx = len(npdata)
    return pk, idx

def findPeaks(wavfile, threshold):
    pkAmp = []
    pkWavIdx = []
    block_size = 4096 * 5
    with sf.SoundFile(wavfile) as f:
        totalSamples = len(f)
        samplerate = f.samplerate
        idxWav = 0   # index into the wav file
        blockCnt = 0
        while idxWav < totalSamples - block_size:
            data = f.buffer_read(block_size, dtype=typedict[f.subtype])
            npdata = convertToNumpy(f, typedict, data, channelchoice)
            blockMax = np.max(np.abs(npdata))
 #           print("------blockCnt",blockCnt, blockMax, threshold, block_size, "idxWav", idxWav)
            if blockMax > threshold:
                idx = 0
                while idx < block_size:
                    peak, idx = getNextPeak(np.abs(npdata), idx, threshold)
                    #print(peak, idx, idx + blockCnt * block_size, blockCnt)
                    if peak>0:
                        idxWavPeak = idx + blockCnt * block_size
#                        print("peak", peak, "idxWavPtr", idxWavPeak, idx, blockCnt, block_size)
                        pkAmp.append(peak)
                        pkWavIdx.append(idxWavPeak)
            idxWav += block_size
            blockCnt += 1
    return pkAmp, pkWavIdx, samplerate

def getNextZero(istart, data, dir):  # dir=1  forward in time  -1 backward
    i = istart
    while (i>0) and (i<len(data)-1 and ((data[i]>0 and data[i+dir]>0) or (data[i]<0 and data[i+dir]<0))):
        i += dir
    i += dir
    print(dir, istart, i, abs(istart - i))
    if (i<=0) or (i>=len(data)):
        return i, 9999   #signifies that no zero was found in this search
    return i, abs(istart - i)    # return width to this next zero

def findClicks(wavfile, pkAmp, pkWavIdx, zeroCrossThreshold, maxClickWidth, minAmpFrac):
    clickAmp=[]
    clickWidth=[]
    clickFreq=[]
    clickWavIdx1=[]
    clickWavIdx2=[]
    clickZeroCnt = []
    #  for each peak at pkWavIdx in wav file, define clicks outward at sucessive zero crossings < zeroCrossThreshold
    #      and limiting width of click to less than the maxClickWidth (in samples)
    with sf.SoundFile(wavfile) as f:
        for i in range(len(pkWavIdx)):
            if i == 0 or pkWavIdx[i] > clickWavIdx2[-1]:
                f.seek(pkWavIdx[i] - maxClickWidth)
                data = f.buffer_read(2*maxClickWidth, dtype=typedict[f.subtype])
                npdata = convertToNumpy(f, typedict, data, channelchoice)  # this window encompases wav data possibly containing a click
                print(i, pkWavIdx[i], pkAmp[i], maxClickWidth, npdata)
                iprior = maxClickWidth  # start in the middle of the data window
                ipost = maxClickWidth
                width = 0
                done = False
                freqAve = 0
                freqAve2 = 0
                nZeroCross = 0
                minAmp = minAmpFrac*np.max(np.abs(npdata))
                while not done:   # look forward in time series
                    ipostNext, delta = getNextZero(ipost, npdata, +1)
                    freq = samplerate/(2.0*delta)
                    freqAve += freq
                    freqAve2 += freq**2
                    nZeroCross += 1
                    width += delta
                    if width >= maxClickWidth:
                        done = True
                    if delta > zeroCrossThreshold or np.max(np.abs(npdata[ipost: ipostNext])) < minAmp:
                        done = True
                    ipost = ipostNext

                done = False
                if width < maxClickWidth:
                    while not done:  # now look backward
                        ipriorNext, delta = getNextZero(iprior, npdata, -1)
                        freq = samplerate / (2.0 * delta)
                        freqAve += freq
                        freqAve2 += freq ** 2
                        nZeroCross += 1
                        width += delta
                        if width > maxClickWidth:
                            done = True
                        if (width<maxClickWidth) and (delta > zeroCrossThreshold or np.max(np.abs(npdata[ipriorNext:iprior])) < minAmp):
                            clickAmp.append(np.max(np.abs(npdata)))
                            clickWidth.append(width)
                            freqAve = int(freqAve/nZeroCross)
                            freqSig = int(math.sqrt((freqAve2 - nZeroCross*freqAve**2)/(nZeroCross - 1)))
                            clickFreq.append([freqAve, freqSig])
                            clickWavIdx1.append(pkWavIdx[i] - maxClickWidth + ipriorNext)
                            clickWavIdx2.append(pkWavIdx[i] - maxClickWidth + ipostNext)
                            clickZeroCnt.append(nZeroCross)
                            done = True
                        iprior = ipriorNext
    return clickAmp, clickWidth, clickFreq, clickWavIdx1, clickWavIdx2, clickZeroCnt

def getClass(deltaIdx,minSlowClickSamples, minFastClickSamples, minBuzzSamples):
    if deltaIdx > minSlowClickSamples:
        return 0 #"single"
    if deltaIdx > minFastClickSamples:
        return 1 #"slow"
    if deltaIdx > minBuzzSamples:
        return 2 #"fast"
    else:
        return 3 #"buzz"

def classifyClicks(minClickCnt, clickFreq, clickWavIdx1, clickWavIdx2, maxClickIntervalStdFrac, minSlowClickSamples, minFastClickSamples, minBuzzSamples):
    className = []
    classNumClicks = []
    classWavIdx1 = []
    classWavIdx2 = []

    i1 = 0
    i2 = 0
    # try to group and then classify clicks based on their intervals
    groupClickCnt = 0
    ptr = clickWavIdx2[0]  # start with first click's 2 index
    typeClick = 0    # typeClick is used to try to aggregate sucessive clicks of a single type 0=single 1=slow 2=fast 3=buzz
    typeClickPrior = -1
    typeCnt   = 0
    for i in range(1, len(clickWavIdx1)):
        deltaIdx = clickWavIdx1[i] - ptr
        thisClass = getClass(deltaIdx,minSlowClickSamples, minFastClickSamples, minBuzzSamples)
        if thisClass == typeClick:
            typeCnt += 1
            i2 = clickWavIdx2[i]
        else:
#            if typeCnt >= minClickCnt:  # we are on to a new class so save this one (if long enough for slow/fast/buzz)
            if i1>0 and i2>0:
                className.append(thisClass)
                classNumClicks.append(typeCnt)
                classWavIdx1.append(i1)
                classWavIdx2.append(i2)

            typeClick = thisClass
            typeCnt = 1
            i1 = clickWavIdx1[i]
        ptr = clickWavIdx1[i]


    return className, classNumClicks, classWavIdx1, classWavIdx2

# import pandas as pd
#
# iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
# import plotly.express as px
# fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
#               color='species')
# fig.show()
################################################################
channelchoice = -1    # pick channel with higher amplitude
typedict = {}
typedict['FLOAT'] = 'float32'
typedict['PCM_16'] = 'int16'

wavfile="/home/val/PycharmProjects/AEproject/wavFiles/OS_9_12_2021_09_37_00_10min.wav"

#wav_idx, delta_idx, amp = scanZerocrossing(wavfile)

### get list of all peaks > threshold and their indices into wav file
threshold = 8000   # max amplitude is 32k  (Integer samples)
pkAmp, pkWavIdx, samplerate = findPeaks(wavfile, threshold)

# scan peaks and group into clicks that meet stated criteria:
zeroCrossMax_micro_s = 200  # micro seconds
maxClickWidth_micro_s = 1000 # micro seconds
minAmpFrac = 0.25  # smallest peak for click must be greater than this fraction of pkAmp
zeroCrossThreshold = int(samplerate * zeroCrossMax_micro_s / 1.0e6)  #samples
maxClickWidth = int(samplerate * maxClickWidth_micro_s / 1.0e6) #samples
clickAmp, clickWidth, clickFreq, clickWavIdx1, clickWavIdx2, clickZeroCnt = findClicks(wavfile, pkAmp, pkWavIdx, zeroCrossThreshold, maxClickWidth, minAmpFrac)

# group clicks into classes  ['single', 'repeating' with repeating separated into 'slow', 'fast'  'buzz' }
classWavIdx1 = []
classWavIdx2 = []
className = []

minClickCnt = 5  # at least this number of evenly spaced clicks needed for a repeating class
maxClickIntervalStdFrac = 0.1 # standard deviation of click intervals must be less than this fraction of the average interval
minSlowClickInterval = 500        #  slow clicks have intervals longer than this number of milliseconds
minFastClickInterval = 50         #  fast clicks have intervals between this (ms) and slowClicks
minBuzzInterval = 0.5             #  buzz clicks have intervals less than this (ms)
                           #  faster clicks are termed buzzes


className, classNumClicks, classWavIdx1, classWavIdx2 = classifyClicks(minClickCnt, clickFreq, clickWavIdx1, clickWavIdx2, maxClickIntervalStdFrac, \
          int(minSlowClickInterval*samplerate/(1000)),int(minFastClickInterval*samplerate/(1000)),int(minBuzzInterval*samplerate/(1000)))
#
# fig = px.histogram(delta_idx)
# #fig.show()
# N = len(wav_idx)
# d_delta_idx = np.abs(delta_idx[1:] - delta_idx[0:N-1])
#
# bigamp = []
# bigampdelta = []
# bigampidx = []
# for i in range(len(delta_idx)):
#     if amp[i] > 5000:
#         bigampdelta.append(delta_idx[i])
#         bigampidx.append(wav_idx[i])
#         bigamp.append(amp[i])
#
# plt.hist(bigampdelta, bins=80)
# plt.show()
# plt.plot(amp)
# plt.show()
# plt.hist2d(amp, delta_idx, bins=80)
# plt.show()
