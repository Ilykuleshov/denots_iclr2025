## Reproducibility

### Docker (recommended)
1. Clone the repository.
2. Make sure you have a compatible CUDA version. If your version does not match the one from the Docker file, you may encounter errors. To fix them, specify the corresponding base image in the head of the Docker file (e.g. `FROM nvidia/cuda:12.2.2-devel-ubuntu22.04` for CUDA 12.2).
3. Build the docker image:
```
$ ./build_image
```
4. Launch the docker container:
```
$ ./launch_container
```
5. Attach to the launched container, it will be named `<your username>-<name of the repo's directory>`, e.g. `j.doe-denots`:
```
$ docker attach j.doe-denots
```
6. Install the python environment using uv:
```
$ uv venv --relocatable && uv sync
```

7. Download & preprocess the datasets, [following the guide in the `data_import` folder](data_import/README.md).

### No Docker
We use uv, with all main packages specified in `pyproject.toml`, and the whole configuration of the python environment saved in `uv.lock`.
Simply running `uv sync` should be enough to reproduce our environment.
However, you will need to ensure that the external environment is compatible, e.g. `nvcc` is required by the state space models package.

### Usage
We use [Hydra](https://hydra.cc), so different experiments are best selected using overrides with multirun, e.g.:
```
$ python3 experiments/supervised.py -m backbone=denots backbone.depth=1,2,5,10,20,30 benchmark=sepsis seed=0,1,2,3,4
```

## Technical details

### Project structure

Our project is structured as follows:
 - [**assets**](assets): contains the images used in the README.
 - [**config**](config): contains the Hydra configuration files, which specify the exact hyperparameters, used for the experiments.
 - [**data_import**](data_import): contains the automated scripts and notebooks used to downlad and preprocess the data.
 - [**src**](src): contains the source code of the project. The code is locally documented.
 - [**experiments**](experiments): contains the scripts used to run the experiments. Specifically, we launch 3 types of experiments: [supervised](experiments/supervised.py), [attack](experiments/attack.py) and [forecasting](experiments/forecasting.py). We also include a script to benchmark model sizes for convenience.

### Experiment tracking
Since this project involves a lot of long-running experiments, we save all the artifacts to the chosen MLOps tracking platform.
We used a locally-hosted MLFlow instance (see docker-compose), although other trackers may be used in our repository.