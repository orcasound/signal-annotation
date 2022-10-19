# signal-annotation
Datastructures and Python code to annotate timeseries and spectrograms

## program detectAndClassify.py
<pre>
Read a wav file and scan for clicks that meet selected criteria.
Group these clicks into bouts when possible.
Write bouts (groups of related clicks) to csv file.

Logic flow:
Select initial parameters:
wav file
click detection parameters:
g_channelchoice = -1      # if stereo, pick channel with higher amplitude
g_threshold = 7000        # max amplitude is 32k  (Integer samples) abs(wavform peak) must be above this threshold
g_zeroCrossMax_micro_s = 200   # micro seconds  -- don't extend click is zero-crossing is longer than this
g_maxClickWidth_micro_s = 1000 # micro seconds  -- don't extend click if total width is greater than this

bout classification parameters:
g_maxBoutClickGap = 2  # if click comes in more that this number of seconds after previous click in bout, start a new bout
g_minClicksInBout = 4
g_maxClickGapSecs = 2   # click lists and bouts end when next click comes in at least this much later than previous click in list
</pre>

Bout classification is to "Buzz", "Fast", "Slow"

<pre>
g_maxBuzzGapSecs = 0.03  # Buzz if average gap is less than this
g_maxFastGapSecs = 0.35  # Fast if average gap is less than this
g_maxSlowGapSecs = 0.75  # Slow if average gap is less than this
</pre>

Bouts are written out as csv file with these columns:
(The Constancies are the ratio of mean to standard deviation.)
<pre>
Cntr	Nclicks	Class	wavIdx1	wavIdx2	boutConstancy	gapConstancy	peakConstancy	freqConstancy	widthConstancy	meanGap	meanPeak	meanFreq	meanWidth
1	7	Slow	68019	180012	1.7	1.9	1.8	2.3	0.8	0.42254	11839.6	12001.9	0.000612
2	10	Slow	1190719	1353868	8.8	7.2	3.2	8.5	16.3	0.410688	18363.7	15365.3	0.000333
3	5	Buzz	1610211	1614275	19.1	31.7	11.6	9.8	23.3	0.022715	7837.8	13120.4	0.000259

</pre>

