<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# TopoSLAM: topographical SLAM using deep visual odometry & visual place recognition

[![Best of ACRV Repository](https://img.shields.io/badge/collection-best--of--acrv-%23a31b2a)](https://roboticvision.org/best-of-acrv)
![Primary language](https://img.shields.io/github/languages/top/best-of-acrv/toposlam)
[![PyPI package](https://img.shields.io/pypi/pyversions/toposlam)](https://pypi.org/project/toposlam/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/toposlam.svg)](https://anaconda.org/conda-forge/toposlam)
[![Conda Recipe](https://img.shields.io/badge/recipe-toposlam-green.svg)](https://anaconda.org/conda-forge/toposlam)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/toposlam.svg)](https://anaconda.org/conda-forge/toposlam)
[![License](https://img.shields.io/github/license/best-of-acrv/toposlam)](./LICENSE.txt)

TopoSLAM is an implementation of topological SLAM, which combines [DF-VO] visual odometry and work in visual place recognition. TODO more context???

TODO: image of the system's output

The repository contains TODO what does it contain?
The package is easily installable with `conda`, and can also be installed via `pip` if you'd prefer to manually manage dependencies.

Our code is free to use, and licensed under BSD-3. We simply ask that you [cite our work](#citing-our-work) if you use TopoSLAM in your own research.

## Related resources

This repository brings the work from a number of sources together. Please see the links below for further details:

- our review paper on visual odometry: ["Visual Odometry Revisited: What Should Be Learnt?"](#citing-our-work)
- our paper on DF-VO: ["DF-VO: What Should Be Learnt for Visual Odometry?"](#citing-our-work)
- the original DF-VO implementation: [https://github.com/Huangying-Zhan/DF-VO](https://github.com/Huangying-Zhan/DF-VO)
- our paper on scalable visual place recognition: ["Scalable place recognition under appearance change for autonomous driving"](#citing-our-work)
- our original visual place recognition repository: [https://github.com/dadung/Visual-Localization-Filtering](https://github.com/dadung/Visual-Localization-Filtering)

## Installing TopoSLAM

We offer three methods for installing TopoSLAM:

1. [Through our Conda package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs TopoSLAM and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies

### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. We provide Conda packages through [Conda Forge](https://conda-forge.org/), which recommends adding their channel globally with strict priority:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once you have access to the `conda-forge` channel, TopoSLAM is installed by running the following from inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
u@pc:~$ conda install toposlam
```

TODO is this even relevant for TopoSLAM?

We don't explicitly lock the PyTorch installation to a CUDA-enabled version to maximise compatibility with our users' possible setups. If you wish to ensure a CUDA-enabled PyTorch is installed, please use the following installation line instead:

```
u@pc:~$ conda install pytorch=*=*cuda* toposlam
```

You can see a list of our Conda dependencies in the [TopoSLAM feedstock's recipe](https://github.com/conda-forge/toposlam-feedstock/blob/master/recipe/meta.yaml).

### Pip

TODO does this have system dependencies??

Before installing via `pip`, you must have the following system dependencies installed if you want CUDA acceleration:

- NVIDIA drivers
- CUDA

Then TopoSLAM, and all its Python dependencies can be installed via:

```
u@pc:~$ pip install toposlam
```

### From source

Installing from source is very similar to the `pip` method above

TODO validate this statement is actually true "due to TopoSLAM only containing Python code".

Simply clone the repository, enter the directory, and install via `pip`:

```
u@pc:~$ pip install -e .
```

_Note: the editable mode flag (`-e`) is optional, but allows you to immediately use any changes you make to the code in your local Python ecosystem._

We also include scripts in the `./scripts` directory to support running FCOS without any `pip` installation, but this workflow means you need to handle all system and Python dependencies manually.

## Using TopoSLAM

### TopoSLAM from the command line

### TopoSLAM Python API

## Citing our work

If using TopoSLAM in your work, please cite our papers below as appropriate. TopoSLAM consists of:

- our [ICRA paper on visual odometry](https://arxiv.org/pdf/1909.09803.pdf):

  ```bibtex
  @inproceedings{zhan2019dfvo,
    author={H. {Zhan} and C. S. {Weerasekera} and J. -W. {Bian} and I. {Reid}},
    booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
    title={Visual Odometry Revisited: What Should Be Learnt?},
    year={2020},
    volume={},
    number={},
    pages={4203-4210},
    doi={10.1109/ICRA40945.2020.9197374}}
  ```

- our recent [arXiv paper on DF-VO](https://arxiv.org/pdf/2103.00933.pdf):

  ```bibtex
  @article{zhan2021df,
    title={DF-VO: What Should Be Learnt for Visual Odometry?},
    author={Zhan, Huangying and Weerasekera, Chamara Saroj and Bian, Jia-Wang and Garg, Ravi and Reid, Ian},
    journal={arXiv preprint arXiv:2103.00933},
    year={2021}
  }
  ```

- our [ICCV paper on scalable place recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Doan_Scalable_Place_Recognition_Under_Appearance_Change_for_Autonomous_Driving_ICCV_2019_paper.pdf)

  ```bibtex
  @inproceedings{doan2019scalable,
    title={Scalable place recognition under appearance change for autonomous driving},
    author={Doan, Anh-Dzung and Latif, Yasir and Chin, Tat-Jun and Liu, Yu and Do, Thanh-Toan and Reid, Ian},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={9319--9328},
    year={2019}
  }
  ```

TODO: delete everything form here down once successfully migrated

# Introduction

This repo implements a topological SLAM system.
Deep Visual Odometry ([DF-VO](https://github.com/Huangying-Zhan/DF-VO)) and [Visual Place Recognition](https://github.com/dadung/Visual-Localization-Filtering) are
combined to form the topological SLAM system.

## Publications

1. [Visual Odometry Revisited: What Should Be Learnt?
   ](https://arxiv.org/abs/1909.09803)

2. [DF-VO: What Should Be Learnt for Visual Odometry?
   ](https://arxiv.org/abs/2103.00933)

3. [Scalable Place Recognition Under Appearance Change for Autonomous Driving](https://openaccess.thecvf.com/content_ICCV_2019/html/Doan_Scalable_Place_Recognition_Under_Appearance_Change_for_Autonomous_Driving_ICCV_2019_paper.html)

```
@INPROCEEDINGS{zhan2019dfvo,
  author={H. {Zhan} and C. S. {Weerasekera} and J. -W. {Bian} and I. {Reid}},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Visual Odometry Revisited: What Should Be Learnt?},
  year={2020},
  volume={},
  number={},
  pages={4203-4210},
  doi={10.1109/ICRA40945.2020.9197374}}

@misc{zhan2021dfvo,
      title={DF-VO: What Should Be Learnt for Visual Odometry?},
      author={Huangying Zhan and Chamara Saroj Weerasekera and Jia-Wang Bian and Ravi Garg and Ian Reid},
      year={2021},
      eprint={2103.00933},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{doan2019scalable,
  title={Scalable place recognition under appearance change for autonomous driving},
  author={Doan, Anh-Dzung and Latif, Yasir and Chin, Tat-Jun and Liu, Yu and Do, Thanh-Toan and Reid, Ian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9319--9328},
  year={2019}
}

```

## Demo:

<a href="https://youtu.be/RhywSFHe5GM"><img src='misc/topo_slam.png' width=640 height=320>

### Contents

1. [Requirements](#part-1-requirements)
2. [Prepare dataset](#part-2-download-dataset-and-models)
3. [Run example](#part-3-run-example)
4. [Result evaluation](#part-4-result-evaluation)

### Part 1. Requirements

This code was tested with Python 3.6, CUDA 10.0, Ubuntu 16.04, and [PyTorch-1.0](https://pytorch.org/).

We suggest use [Anaconda](https://www.anaconda.com/distribution/) for installing the prerequisites.

```
cd envs
conda env create -f min_requirements.yml -p {ANACONDA_DIR/envs/topo_slam} # install prerequisites
conda activate topo_slam  # activate the environment [topo_slam]
```

### Part 2. Download dataset and models

The main dataset used in this project is [KITTI Driving Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). After downloaing the dataset, create a softlink in the current repo.

```
ln -s KITTI_ODOMETRY/sequences dataset/kitti_odom/odom_data
```

For our trained models, please visit [here](https://www.dropbox.com/sh/9by21564eb0xloh/AABHFMlWd_ja14c5wU4R1KUua?dl=0) to download the models and save the models into the directory `model_zoo/`.

### Part 3. Run example

```
# run default kitti setup
python main.py -d options/examples/default.yml  -r data/kitti_odom
```

More configuration examples can be found in [configuration examples](https://github.com/Huangying-Zhan/DF-VO/tree/master/options/examples).

The result (trajectory pose file) is saved in `result_dir` defined in the configuration file.
Please check [Configuration Documentation](https://df-vo.readthedocs.io/en/latest/rsts/configuration.html) for reference.

### Part 4. Result evaluation

Please check [here](https://github.com/Huangying-Zhan/DF-VO#part-4-result-evaluation) for evaluating the result.

### License

Please check License file.

### Acknowledgement

Some of the codes were borrowed from the excellent works of [monodepth2](https://github.com/nianticlabs/monodepth2), [LiteFlowNet](https://github.com/twhui/LiteFlowNet) and [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet). The borrowed files are licensed under their original license respectively.
