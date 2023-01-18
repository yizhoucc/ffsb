

from plot_ult import *

# ----------------------------


# A. The firefly task in overhead view. ----------------------------
# schema


# B. An example control trajectory. ----------------------------
# schema


# C. An example path. ----------------------------
# schema


# D. Target locations are randomly drawn within a certain range. Red means monkey skipped this targets. ----------------------------
# load all df
print('loading data')
datapath=Path("Z:\\bruno_normal\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
# plot
overheaddf_tar(df[:1000])
overheaddf_path(df,list(range(1000)))


# E. Monkey paths. ----------------------------
# load all df
# plot


