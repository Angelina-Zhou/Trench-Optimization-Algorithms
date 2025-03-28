Contains various python scripts to run and evaluate PSO algorithms for optimization of groundwater infiltration. eval_all_serial and eval_all_parallel run 100 runs of each variation of PSO and graphs them on a histogram.

To run all PSO algorithms, do the command:

`python eval_all_serial.py`

which will automatically pull all particle swarm optimization functions from individual_psos. These individual PSO scripts can also be run independently. To do so, navigate to the individual_psos directory and execute the scripts using python. For example:
`python pso_basic.py`

New PSO algorithms can be added to individual_psos. If you want it to be evaluated against other PSO algorithms in the eval_all scripts, be sure to name its PSO function particle_swarm_optimization().

Read more about this project (background, motivation, methodology, equations) here: https://docs.google.com/document/d/1qwrpfpv80Bh2p6FQmkC0Nqp8NmP9r-t8EW70DlCg_r4/edit?usp=sharing
