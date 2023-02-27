import pandas as pd

archnum = {'sandy bridge': 0, 'ivy bridge': 1, 'haswell': 2, 'broadwell': 3, 'skylake': 4}
refmod = 'E3-1230V2'                                                   # reference model
df = pd.read_json('intel_chips_all.json')

df = df[[0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 22, 23, 24, 25, 30, 31, 33]] # keep only required columns
df = df[df[0].isin(archnum.keys())]                                    # keep rows with architecture among listed ones
df[0] = df[0].astype('category').map(archnum)                          # codes are assigned to microarchitectures by above mapping
df[1] = df[1].astype('category')
df.loc[df[2] == "Q1 '15", 2] = "Q1'15"                                 # fix abnormal entries
df[2] = df[2].str.replace("Q.'", '', regex=True)                       # remove quarter of year information from launch year
df[2] = df[2].astype(int)
df[4] = df[4].str.replace(' L3', '')                                   # remove irrelevant information
df[4] = df[4].str.replace(' SmartCache', '')
df[4] = df[4].str.replace(' Last Level Cache', '')
df[4] = df[4].str.replace(' MB', '')                                   # remove unit from cache size
df[4] = df[4].astype(float)
df[6] = df[6].str.replace("SSE ?4.1/4.2 AVX 2.0", "SSE4.1/4.2, AVX 2.0", regex=True)
df[6] = df[6].str.replace(', AES', '')
df[6] = df[6].str.replace('SSE4.(2|x)', 'SSE4.1/4.2', regex=True)
df.loc[df[6].str.match('0|1'), 6] = 'AVX'                              # fix abnormal entries
df[6] = df[6].astype('category').cat.codes                             # integer encode instruction set extension 
df[9] = df[9].str.replace("DDR4.*", '4', regex=True)
df[9] = df[9].str.replace(".*DDR3.*", '3', regex=True)                 # integer encode the memory type (only DDR3 and DDR4 values are present in this column)
df[11] = df[11].str.replace(" GB/s", '')                               # remove unit from max memory bandwidth
df[11] = df[11].astype(float)
df[24] = df[24].str.replace(" GHz", '')                                # remove unit from base frequency
df[24] = df[24].str.replace("800 MHz", '0.8')
df[24] = df[24].str.replace("900 MHz", '0.9')
df[24] = df[24].astype(float)
df[25] = df[25].str.replace(" GHz", '')                                # remove unit from turbo frequency
df[25] = df[25].astype(float)
df[30] = df[30].str.replace(" W", '')                                  # remove unit from tdp
df[30] = df[30].astype(float)
df.loc[df[31] == 2, 31] = 1                                            # turbo boost technology (Not supported -> 0, Supported -> 1)
df.drop_duplicates(inplace=True, ignore_index=True)

sdf = pd.read_json('spec_speed.json')

sdf = sdf[[0, 1, 3, 4, 5, 6]]                                          # keep only required columns
sdf[3] = sdf[3].str.replace('Intel ', '')                              # remove irrelevant information from model number
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
sdf[5] = sdf[5].astype('category').map({'DDR' : 1, 'DDR2' : 2, 'DDR3' : 3, 'DDR4' : 4})  # integer encode the memory type
df1 = df.drop(columns=[9])                                             # we use memory type column from spec dataframe instead of Intel
sim = sdf.merge(df1, on=3)                                             # merge the spec and intel dataframes by model number column
sim['1_x'] = sim.groupby(['0_x', '1_y', 3, '4_x', '6_x'])['1_x'].transform('mean')   # average the performance of the workloads with the same (SKU, type, frequency, memory size) configuration
sim.drop_duplicates(ignore_index=True, inplace=True)
refrtime = sim.loc[sim[3] == refmod, '1_x'].mean()                     # average runtime of reference model
sim['1_x'] = (sim['1_x'] - refrtime) / refrtime                        # calculate reference runtimes
sim.drop(columns=[3], inplace=True)                                    # drop model number column
sim=sim[['1_x', '0_y', '1_y', 2, '4_y', '6_y', 5, 10, 11, 12, 22, 23, 24, 25, 30, 31, 33, '0_x', '4_x', '6_x']]     # rearrange columns
sim = pd.get_dummies(sim, columns=['1_y'])                             # one hot encode CPU type
sim = pd.get_dummies(sim, columns=['0_x'])                             # one hot encode benchmark
sim.columns = ['run time', 'uarch', 'year', 'cache', 'instruction set extensions', 'memory type', '# of memory channels', 'max memory bandwidth', 'ecc memory supported', '# of cores', '#of threads', 'base frequency', 'turbo frequency', 'tdp' ,'turbo boost technology', 'hyper-threading', 'freq', 'memory size', 'desktop', 'embedded', 'mobile', 'server',  '400.perlbench', '401.bzip2', '403.gcc', '410.bwaves', '416.gamess', '429.mcf', '433.milc', '434.zeusmp', '435.gromacs', '436.cactusADM', '437.leslie3d', '444.namd', '445.gobmk', '450.soplex', '453.povray', '454.calculix', '456.hmmer', '458.sjeng', '459.GemsFDTD', '462.libquantum', '464.h264ref', '465.tonto', '470.lbm', '471.omnetpp', '473.astar', '481.wrf', '482.sphinx3', '483.xalancbmk']        # rename columns

gdf = pd.read_json('geek_data.json')

gdf = gdf[gdf[6] == 1]                                                 # delete all single-core rows (value 0) as intel dataset has only multicore model data, this halves the number of rows
gdf = gdf[[0, 1, 4, 5, 8]]                                             # keep only required columns
gdf = gdf[gdf[0].str.contains('Intel ')]                               # remove rows of model other than Intel
gdf[0] = gdf[0].str.replace('Intel ', '')                              # remove irrelevant information from model number
gdf[0] = gdf[0].str.replace('Core ', '')
gdf[0] = gdf[0].str.replace('Xeon ', '')
gdf[0] = gdf[0].str.replace('Pentium ', '')
gdf[0] = gdf[0].str.replace('2 Duo ', '')
gdf[0] = gdf[0].str.replace(' v', 'V')
gdf[0] = gdf[0].str.replace(' 0', '')
gdf[0] = gdf[0].str.replace('- ', '-')
gdf[0] = gdf[0].str.replace('i7-37700K', 'i7-3770K')                   # fix the lone abnormal model number
gdf[1] = (gdf[1]*10).astype(int)*100                                   # round up freq to 100s place for less no. of unique values
gdf[4] = gdf[4] / 1024                                                 # convert to MB
gdf[4] = gdf[4].astype(int)                                            # round up float values for less no. of unique values
gim = gdf.merge(df, left_on=0, right_on=3)                             # merge the spec and intel dataframes by model number column
gim[8] = gim.groupby([5, '1_y', 3, '4_x', '1_x'])[8].transform('mean') # average the performance of the workloads with the same (CPU SKU, Frequency, Memory Size) configuration
gim.drop_duplicates(inplace=True, ignore_index=True)                   # remove duplicate rows
gim[8] = 1 / gim[8]                                                    # inverse performance to estimate runtime
refrtime1 = gim.loc[gim[3] == refmod, 8].mean()                        # average runtime of the reference model
gim[8] = (gim[8] - refrtime1) / refrtime1                              # calculate reference runtimes
gim.drop(columns=['key_0', '0_x', 3], inplace=True)                    # drop model number columns
gim=gim[[8, '0_y', '1_y', 2, '4_y', 6, 9, 10, 11, 12, 22, 23, 24, 25, 30, 31, 33, 5, '1_x', '4_x']]     # rearrange columns
gim = pd.get_dummies(gim, columns=['1_y'])                             # one hot encode CPU type
gim = pd.get_dummies(gim, columns=[5])                                 # one hot encode benchmark
gim.columns = ['run time', 'uarch', 'year', 'cache', 'instruction set extensions', 'memory type', '# of memory channels', 'max memory bandwidth', 'ecc memory supported', '# of cores', '#of threads', 'base frequency', 'turbo frequency', 'tdp' ,'turbo boost technology', 'hyper-threading', 'freq', 'memory size', 'desktop', 'embedded', 'mobile', 'server', 'AES', 'BZip2 Compress', 'BZip2 Decompress', 'BlackScholes', 'Blur Filter', 'DFFT', 'DGEMM', 'Dijkstra', 'JPEG Compress', 'JPEG Decompress', 'Lua', 'Mandelbrot', 'N-Body', 'PNG Compress', 'PNG Decompress', 'Ray Trace', 'SFFT', 'SGEMM', 'SHA1', 'SHA2', 'Sharpen Filter', 'Sobel', 'Stream Add', 'Stream Copy', 'Stream Scale', 'Stream Triad', 'Twofish']        # rename columns

# For case study 3, where both SPEC and Geekbench dataset will be input to the model
sim['sg'] = 0                                                          # new column denoting whether spec or geekbench row
gim['sg'] = 1
sgim = pd.concat([sim, gim], ignore_index=True)                        # concatenate the two datframes
sgim = sgim.fillna(0)                                                  # fill empty cells with 0

norm_cols = ['freq', 'year', 'memory size', 'cache', 'run time', 'max memory bandwidth', '# of cores', '#of threads', 'tdp']
for col in norm_cols:                                                  # z-score normalize required columns
    sim[col] = (sim[col]-sim[col].mean())/sim[col].std() 
    gim[col] = (gim[col]-gim[col].mean())/gim[col].std() 
    sgim[col] = (sgim[col]-sgim[col].mean())/sgim[col].std()

sim.drop(columns=['sg'], inplace=True)
gim.drop(columns=['sg'], inplace=True)
sim.to_csv('spec_perf.csv', index=False)                               # sim=>[21280 rows x 50 columns]
gim.to_csv('geek_perf.csv', index=False)                               # gim=>[36936 rows × 49 columns]
sgim.to_csv('sg_perf.csv', index=False)                                # sgim=>[58216 rows × 78 columns]
