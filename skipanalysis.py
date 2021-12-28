import pickle

from matplotlib import collections
# from monkey_functions import datawash, monkey_data_downsampled
import matplotlib.pyplot as plt
print('loading data')
note='testdcont'
with open("C:/Users/24455/Desktop/bruno_normal_downsample",'rb') as f:
        df = pickle.load(f)

trialtype=df.category.tolist()
uniques=set()
for a in trialtype:
    if a not in uniques:
        uniques.add(a)

trialtypeencoding={}
defaultencoding=0
for a in uniques:
    trialtypeencoding[a]=defaultencoding
    defaultencoding+=1

encodedtrialtype=[]
for a in trialtype:
    encodedtrialtype.append(trialtypeencoding[a])

#plt.plot(encodedtrialtype)

# consective skipps
from collections import deque
nskips=3
window=deque(maxlen=nskips)
for a in encodedtrialtype:
    window.append(a)
    