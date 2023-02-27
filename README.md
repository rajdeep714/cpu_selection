# cpu_selection
Implementation of the journal "Predicting New Workload or CPU Performance by Analyzing Public Datasets"  
[Journal link](https://dl.acm.org/doi/10.1145/3284127)  
[Data repository link](https://github.com/Emma926/cpu_selection)  

Files geek_data.tar.gz, intel_chips_all.json and spec_speed.json are obtained from the data repository. Please unzip geek_data.tar.gz before use.

parse_dataset.py makes three files after execution:  
+ spec_perf.csv : dataset of Intel SKUs' specifications and their performance on SPEC benchmarks. Contains the following columns:
  1. run time : The runtime of the workloads z-normalized after converting to relative runtime (reference model E3-1230V2 having average runtime 208.5388)
  2. uarch : Intel microarchitecture code names integer encoded using the mapping  
  `{'sandy bridge': 0, 'ivy bridge': 1, 'haswell': 2, 'broadwell': 3, 'skylake': 4}`
  3. year : Last 2 digits of year of release z-normalized
  4. cache : Last level cache size in MB z-normalized
  5. instruction set extensions : Integer encoded using the mapping  
  `{'AVX' : 0,  'AVX 2.0' : 1,  'SSE4.1/4.2' : 2,   'SSE4.1/4.2, AVX' : 3,  'SSE4.1/4.2, AVX 2.0' : 4}`
  6. memory type : Integer encoded using the mapping (taken from the SPEC data repository instead of Intel's)  
  `{'DDR' : 1, 'DDR2' : 2, 'DDR3' : 3, 'DDR4' : 4}`
  7. \# of memory channels : Number of memory channels
  8. max memory bandwidth : Maximum memory bandwidth in GB/s z-normalized
  9. ecc memory supported : Whether ECC memory is supported (binary)
  10. \# of cores : Number of cores z-normalized
  11. \# of threads : Number of threads z-normalized
  12. base frequency : Nominal frequency in GHz 
  13. turbo frequency : Turbo frequency in GHz
  14. tdp : Thermal design power in W z-normalized
  15. turbo boost technology : Whether Turbo Boost is supported (binary)
  16. hyper-threading : Whether hyper-threading is supported (binary)
  17. freq : The dynamic frequency in MHz z-normalized
  18. memory size : The size of off-chip memory in MB z-normalized
  19. type : server, desktop, mobile, or embedded one hot encoded
  23. workload : The name of the 28 SPEC workloads one hot encoded.
  
 + geek_perf.csv : dataset of Intel SKUs' specifications and their performance on Geekbench benchmarks. Contains columns similar to spec_perf.csv with the difference:
   1. run time : inverse of performance values given in the Geekbench data repository z-normalized after converting to relative runtime (reference model E3-1230V2 having average runtime 0.077037)
   2. memory type : Integer encoded using the mapping (taken from the Intel data repository)
   3. workload : The name of the 27 Geekbench workloads one hot encoded.
   
 + sg_perf.csv : Concatenation of spec_perf and geek_perf with additional column 'sg' having value 0 for SPEC and 1 for Geekbench data (used for case study of cross prediction).
 
