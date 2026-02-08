This repository contains the key scripts to run the various versions of the coarsening model as described in the publication, where the detailed model description.

Coarsening model of chromosomal crossover placement
Marcel Ernst, Riccardo Rossetto, and David Zwicker
PRX Life - Accepted 30 January, 2026
DOI: https://doi.org/10.1103/8jrt-rb28

We have three different models implemented
(i) The full HEI10 coarsening model that includes the nucleoplasm, the Synaptonemal Complex (SC) and the droplets. Code in Coarsening_model.py
(ii) The simple coarsening model that only considers the Synaptonemal Complex and the droplets. Coarsening_without_nucleoplasm.py
(iii) The coarsening model with abolished Synaptonemal complex where droplets are only exchanging HEI10 via the surrounding nucleoplasm. Coarsening_without_SC.py

The Jupyter Notebook Coarsening_Model.ipynb explains how to run the various models and exemplarily plots the time-development of the average state variables.