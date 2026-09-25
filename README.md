# What does graph construction preserve? A Discrete Ricci curature Diagnostics for point cloud graphs

## Setup
pip install -r requirements.txt

Edit `paths.py`: the JetSet shard root, the QM9 shard glob, where the JetSet cache goes, and where results go. `paths.py` 

## DataSet used are:
1. JetSet
   We used `mc-flavtag-ttbar-small.h5` and subsampled with seed=0 to 300K for training and 50K for testing  
   ATLAS collaboration (2025). ATLAS tt¯ simulation for ML-based jet flavour tagging (JetSet). CERN Open Data Portal.                           DOI:10.7483/OPENDATA.ATLAS.QG8W.TO8P : <https://opendata.cern.ch/record/93940>
2. QM9  MoleculeNet: A Benchmark for Molecular Machine Learning" <https://arxiv.org/abs/1703.00564>
   <https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip> from                                                       <https://pytorchgeometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/datasets/qm9.html#QM9>

## Save graphs first
python -m experiments.jetset.save_jetset_graphs.py
python -m experiments.qm9.save_qm9_graphs.py

## Running qm8 edege recovery
````
python -m experiments.qm9.true_bonds
python -m experiments.qm9.edge_recovery

```




## Running the diagnostic
````
python -m experiments.qm9.run_scores
python -m experiments.qm9.run_bootstrap

python -m experiments.jetset.run_scores
python -m experiments.jetset.run_bootstrap
```


## What the scores are

For a construction g and a comparison (two populations of nodes or edges), 
`S_sep` is the W1 distance between the curvature distributions of the two populations.

For the jets it is averaged over the flavor pairs.
Conditioning on a variable splits the point clouds into bins and gives

```
S_sep_cond    = \sum_b w_b W1(A_b, B_b) / sum_b w_b

S_sep_kept    = W1(concat_b A_b, concat_b B_b)

preserved_sep = S_sep_cond / S_sep_kept

share         = 1 - preserved_sep
```

with w_b the number of point clouds in bin b. Both terms use the same point clouds, so the share is the part of
 the separation the variable accounts for.

The bootstrap resamples point clouds (jets within flavor, molecules as level), never individual curvature values, 
and reuses one draw of counts for every quantity so that constructions stay paired.


## Running the predictive 

```
python -m experiments.qm9.run_qm9_predictive

python -m experiments.jetset.run_jetset_predictive
python -m experiments.jetset.run_jetset_track_control
python -m experiments.jetset.run_jetset_degree_control
```


## JetSet cache

The first JetSet run intersects the shard families, freezes the subsample of jets to `<JETSET_CACHE_DIR>/kept_train_jet_ids_<size>_seed<seed>.npy`
So need to delete the cache directory to rerun larger data




