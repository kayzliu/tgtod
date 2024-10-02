TGTOD: A Global Temporal Graph Transformer for Outlier Detection at Scale
-------------------------------------------------------------------------

The official implementation for paper "TGTOD: A Global Temporal Graph Transformer for Outlier Detection at Scale"

TGTOD is an end-to-end temporal graph Transformer for outlier detection, conducting global spatiotemporal attention at scale.

Our experiments are conducted on [DGraph](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DGraphFin.html#torch_geometric.datasets.DGraphFin) , [Elliptic](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html#torch_geometric.datasets.EllipticBitcoinDataset), and [FiGraph](https://github.com/XiaoguangWang23/FiGraph) datasets.

## Requirements

To run TGTOD, we require the following dependencies:
```
python==3.10
numpy==1.26.4
pandas==2.2.2
torch==2.2.0
torch_geometric==2.5.3
pygod==1.1.0
```

## Usage

Run TGTOD on Elliptic dataset under stationary setting:
```sh
python main.py --dataset elliptic --timeslot 1 --hid_dim 32 --num_parts 64
```
Run TGTOD on DGraph dataset under stationary setting:
```sh
python main.py --dataset dgraph --timeslot 10 --hid_dim 16 --num_parts 64
```
Run TGTOD on FiGraph dataset under stationary setting:
```sh
python main.py --dataset figraph --timeslot 1 --hid_dim 16 --num_parts 1 --graph_weight 0.9
```
Run TGTOD on FiGraph dataset under non-stationary setting:
```sh
python main.py --dataset figraph --timeslot 1 --hid_dim 16 --num_parts 1 --graph_weight 0.9 --station False
```
