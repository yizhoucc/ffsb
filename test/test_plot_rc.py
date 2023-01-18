from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif'})
rc('text', usetex=False)
plt.rcParams['font.serif']=[font]

from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['serif']))
font='C:/Users/24455/iCloudDrive/misc/computer-modern/cmunsi.ttf'

import matplotlib.font_manager as font_manager

font_dirs = ['C:/Users/24455/iCloudDrive/misc/computer-modern', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

font_manager.fontManager.addfont(font)
prop = font_manager.FontProperties(fname=font)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


# font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_files)
matplotlib.rcParams['font.family'] = ['cmunsi']


import matplotlib as mpl
import matplotlib.font_manager as fm
font='C:/Users/24455/iCloudDrive/misc/computer-modern/cmunsi.ttf'

fe = fm.FontEntry(
    fname=font,
    name='test')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'

from matplotlib import pyplot as plt
plt.rc('font',family='computer-modern')
plt.figure()
plt.title('asdf')




from matplotlib import font_manager
from pathlib import Path

font_dirs = [Path('C:/Users/24455/iCloudDrive/misc/computer-modern'), ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    
result = font_manager.findfont('cmunsi')

print(result)



import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

print(mpl.rcParams['font.sans-serif'])

# Just write the name of the font
mpl.rcParams['font.sans-serif'] = 'computer-modern'
print(mpl.rcParams['font.sans-serif'])

plt.figure()
plt.plot(range(0,50,10))
plt.title('Font test', size=32)


plt.figure()
x = np.linspace(0, 6.5)
y = np.sin(x)
plt.plot(x, y)
plt.title(r'Just a plot of the function $f(x) = \sin(x)$', size=18)

plt.show()




from matplotlib import font_manager
from pathlib import Path
import matplotlib.pyplot as plt

font_dirs = [Path('C:/Users/24455/iCloudDrive/misc/computer-modern'), ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = '14'
plt.rcParams['axes.unicode_minus'] = False
 
plt.title('title')
plt.xlabel('xlabel')
plt.ylabel('ylabel')



# # check all fonts 
# import matplotlib.font_manager as fm
# for f in fm.fontManager.ttflist:
#    print(f.name)
# for f in fm.fontManager.afmlist:
#    print(f.name)