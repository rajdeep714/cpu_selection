import pandas as pd

archnum = {'sandy bridge': 0, 'ivy bridge': 1, 'haswell': 2, 'broadwell': 3, 'skylake': 4}
df = pd.read_json('intel_chips_all.json')

df = df[[0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 22, 23, 24, 25, 30, 31, 33]]
df = df[df[0].isin(archnum.keys())]
df[0] = df[0].astype('category')                    # codes are assigned in above order
df[0] = df[0].map(archnum)
df = df[df[1] != '']
df[1] = df[1].astype('category')
df = pd.get_dummies(df, columns=[1])
df.loc[df[2] == "Q1 '15", 2] = "Q1'15"
df[2] = df[2].str.replace("Q.'", '', regex=True)
df[2] = df[2].astype('int')
df[4] = df[4].str.replace(' SmartCache', '')
df[4] = df[4].str.replace(' L3', '')
df[4] = df[4].str.replace(' L2', '')
df[4] = df[4].str.replace(' Last Level Cache', '')
df[4] = df[4].str.replace(' MB', '')
df[4] = df[4].astype(float)
df[6] = df[6].str.replace("SSE.*AVX.*", '3', regex=True)
df[6] = df[6].str.replace("SSE.*", '1', regex=True)
df[6] = df[6].str.replace("AVX.*", '2', regex=True)
df[6] = df[6].astype(int)
df[9] = df[9].str.replace("DDR4.*", '1', regex=True)
df[9] = df[9].str.replace(".*DDR3.*", '0', regex=True)
df[9] = df[9].astype(int)
df[11] = df[11].str.replace(" GB/s", '')
df[11] = df[11].astype(float)
df[12] = df[12].astype(float)
df[24] = df[24].str.replace(" GHz", '')
df[24] = df[24].str.replace("800 MHz", '0.8')
df[24] = df[24].str.replace("900 MHz", '0.9')
df[24] = df[24].astype(float)
df[25] = df[25].str.replace(" GHz", '')
df[25] = df[25].astype(float)
df[30] = df[30].str.replace(" W", '')
df[30] = df[30].astype(float)
df.loc[df[31] == 1, 31] = 0
df[31] = df[31].astype(float)
df[33] = df[33].astype(float)
df.drop_duplicates(inplace=True, ignore_index=True)

sdf = pd.read_json('spec_speed.json')

sdf = sdf[[0, 1, 3, 4, 5, 6]]
sdf[3] = sdf[3].str.replace('Intel ', '')
sdf[3] = sdf[3].str.replace('Core ', '')
sdf[3] = sdf[3].str.replace('Xeon ', '')
sdf[3] = sdf[3].str.replace('Pentium ', '')
sdf[3] = sdf[3].str.replace('2 Duo ', '')
sdf[3] = sdf[3].str.replace(' v', 'V')
sdf[3] = sdf[3].str.replace('2 Extreme ', '')
sdf[3] = sdf[3].str.replace('Duo ', '')
sdf[3] = sdf[3].str.replace('Atom ', '')
sdf[3] = sdf[3].str.replace('Extreme Edition ', '')
sdf[3] = sdf[3].str.replace(' Extreme Edition', '')
sdf[3] = sdf[3].str.replace('Dual-Itanium ', '')
sdf[3] = sdf[3].str.replace('Dual-', '')
sdf[3] = sdf[3].str.replace('Celeron ', '')
sdf[3] = sdf[3].str.replace('2 Quad ', '')
sdf.drop_duplicates([0, 3, 4, 5, 6], inplace=True, ignore_index=True)
refmod = 'E3-1230V2'
refrtime = sdf.loc[sdf[3] == refmod, 1].mean()
sdf[1] = (sdf[1] - refrtime) / refrtime  # calculate ref. runtime
sdf = sdf[sdf[5].isin(['DDR3', 'DDR4'])]
sdf[5] = sdf[5].astype('category')
sdf[5] = sdf[5].cat.codes
df1 = df.drop(columns=[9])  # memory type used from spec data instead of intel
sdf = pd.get_dummies(sdf, columns=[0])
sim = sdf.merge(df1, on=3)
# sim.drop(columns=[3], inplace=True)
norm_cols = [1, 2, '4_x', '4_y', '6_x', 11, 22, 23, 30]
for col in norm_cols:
    sim[col] = (sim[col]-sim[col].mean())/sim[col].std()

sim.columns = ['run time', 'model number', 'freq', 'memory type', 'memory size', '400.perlbench', '401.bzip2', '403.gcc', '410.bwaves', '416.gamess', '429.mcf', '433.milc', '434.zeusmp', '435.gromacs', '436.cactusADM', '437.leslie3d', '444.namd', '445.gobmk', '450.soplex', '453.povray', '454.calculix', '456.hmmer', '458.sjeng', '459.GemsFDTD', '462.libquantum', '464.h264ref', '465.tonto', '470.lbm', '471.omnetpp', '473.astar', '481.wrf', '482.sphinx3', '483.xalancbmk', 'uarch', 'year', 'cache', 'instruction set extensions', '# of memory channels', 'max memory bandwidth', 'ecc memory supported', '# of cores', '#of threads', 'base frequency', 'turbo frequency', 'tdp' ,'turbo boost technology', 'hyper-threading', 'desktop', 'embedded', 'mobile', 'server']
sim.to_csv('sim1.csv', index=False)
# sim=>[21280 rows x 50 columns]

gdf = pd.read_json('geek_data.json')

gdf = gdf[gdf[6] == 1]
gdf = gdf[[0, 1, 3, 4, 5, 8]]
gdf = gdf[gdf[0].str.contains('Intel ')]
gdf[0] = gdf[0].str.replace('Intel ', '')
gdf[0] = gdf[0].str.replace('Core ', '')
gdf[0] = gdf[0].str.replace('Xeon ', '')
gdf[0] = gdf[0].str.replace('Pentium ', '')
gdf[0] = gdf[0].str.replace('2 Duo ', '')
gdf[0] = gdf[0].str.replace(' v', 'V')
gdf[0] = gdf[0].str.replace(' 0', '')
gdf[0] = gdf[0].str.replace('- ', '-')
gdf[0] = gdf[0].str.replace('i7-37700K', 'i7-3770K')
gdf.drop_duplicates([0, 1, 3, 4, 5], inplace=True, ignore_index=True)
gdf[8] = 1 / gdf[8]
refrtime1 = gdf.loc[gdf[0] == refmod, 8].mean()
gdf[8] = (gdf[8] - refrtime1) / refrtime1
df2 = df.drop(columns=[4])
gdf = pd.get_dummies(gdf, columns=[5])
gim = gdf.merge(df2, left_on=0, right_on=3)
gim.drop(columns=['0_x', '3_y'], inplace=True)
norm_cols = [1, 2, '3_x', 4, 8, 11, 22, 23, 30]
for col in norm_cols:
    gim[col] = (gim[col]-gim[col].mean())/gim[col].std()

gim.columns = ['model number', 'freq', 'cache', 'memory size', 'run time', 'AES', 'BZip2 Compress', 'BZip2 Decompress', 'BlackScholes', 'Blur Filter', 'DFFT', 'DGEMM', 'Dijkstra', 'JPEG Compress', 'JPEG Decompress', 'Lua', 'Mandelbrot', 'N-Body', 'PNG Compress', 'PNG Decompress', 'Ray Trace', 'SFFT', 'SGEMM', 'SHA1', 'SHA2', 'Sharpen Filter', 'Sobel', 'Stream Add', 'Stream Copy', 'Stream Scale', 'Stream Triad', 'Twofish', 'uarch', 'year', 'instruction set extensions', 'memory type', '# of memory channels', 'max memory bandwidth', 'ecc memory supported', '# of cores', '#of threads', 'base frequency', 'turbo frequency', 'tdp' ,'turbo boost technology', 'hyper-threading', 'desktop', 'embedded', 'mobile', 'server']
gim.to_csv('gim1.csv', index=False)
# gim=>[65664 rows x 49 columns]

# sim['sg'] = 0
# gim['sg'] = 1
# sgim = pd.concat([sim, gim], ignore_index=True)
# sgim = sgim.fillna(0)
# sgim.to_csv('sgim.csv', index=False)
# sgim=>[86944 rows x 78 columns]
