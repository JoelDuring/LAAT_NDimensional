# N-Dimensional LAAT
This project is a fork of the original LAAT repository. It adds two additional features:
1. Adds support for the use of LAAT in any dimension using C++ template functions. When using the python bindings this is limited to 3-15 dimensions, but support for more can be added.
2. The (quadratic) memory usage of the LAAT algorithm can be restricted or eliminated by setting a memory limit. This will result in longer running times.

# Installation
To install this module from source, you will need to install a suitable C++ compiler and CMake. Any other external dependencies of this project are included as git submodules and they must be initialized when this repository is cloned:

```sh
git clone https://github.com/abst0603/LAAT.git
cd LAAT
git submodule update --init
```

The project can then be installed as a Python module from the main directory of the project using pip:

```sh
pip install .
```

The module can now be imported and used in python:

```python
import laat
pheromone = laat.LAAT(...)
```

# Usage
An example of the usage of the LAAT module can be found in the `examples` folder. After the LAAT module for Python has been installed, the example can be run with:

```sh
cd examples
python example.py
```

To adapt this example to your data, fit the relevant functions into your data processing pipeline, or store your data in a csv file and load it into python as shown in the example. Tweak the hyper-parameters for both LAAT and MBMS to fit your dataset. Use the result of MBMS for further data processing, or write it straight to a csv file as shown in the example.

## N-Dimensional data
When using the Python module, higher-dimensional data can be run by using the corresponding function, for example: `laat.LAAT4(...)` for data in four dimensions.

## Limiting memory usage
The standard LAAT algorithm has a quadratic space complexity. For large datasets, the memory usage is often too high. Most of this memory is used to store local neighbourhoods to speed up computation.

The memory usage of the algorithm can be limited by setting the `memoryLimit` hyperparameter. This is used to control the amount of memory used for the storing of local neighbourhoods in Gigabytes. This will result in a longer running time of the algorithm. For most applications, limiting the memory to zero Gigabytes will double the computation time. When no memory limit is specified, the algorithm will store all local neighbourhoods to optimize running time.

# Compiling from source
You can choose to build this module from source as a static C++ library for use in C++ projects. To do so, create a build directory and run CMake:

```sh
mkdir build
cd build
cmake ..
make
```

Now the build directory will create both the static C++ library file `liblaatlib.a` and the Python module binary file `laat.cpython`.

# Parameter configuration
To understand the role of each parameter, we highly suggest reading our paper as cited in the following. In summary, there are some rules of thumb and points one should consider when setting the parameters. We discuss each of them individually as follows.

`Pheromone` ($\zeta$): This is the constant amount of pheromone each ant deposits on a point when visiting the point. It can be any number, since only the ratio of pheromone on the point to other points is essential, not the exact value. We usually set it as 0.05.

`Evaporation rate` ($\varphi$): This parameter determines the pace of pheromone evaporation. It should be a value between 0 to 1. Closer to 0, ants pay attention to a high concentration of pheromone greedily, and closer to 1, high evaporation causes noisy behavior of ants. (default = 0.1)

`Kappa` ($\kappa$): $\kappa$ value can vary between 0 to 1, where 0 means only pheromone is influential in determining an ant's next step, and a value of 1 forces the ants to choose only based on the alignment of the neighboring data points.

`Beta` ($\beta$): For information about its interpretation please visit our paper. $\beta=10$ works perfectly for all datasets we have tried.

The other three parameters, namely `number of ants`, `number of steps`, and `number of iterations` are not independent, and we also discuss them combined. The higher the value for these parameters, the algorithm converges better on the structures. However, if the multiplication of these three numbers is higher than 10 to 100 times the total number of dataset points, the algorithm will converge. We suggest bounding the `number of iterations` to a smaller number (default = 100) and increasing the `number of steps` as much as possible. Increasing the `number of ants` gives more exploration power to the algorithm.

# Citing
If you use LAAT in your research, please cite [LAAT: Locally Aligned Ant Technique for discovering multiple faint low dimensional structures of varying density](https://ieeexplore.ieee.org/abstract/document/9780217)
```
@ARTICLE{9780217,
author={Taghribi, Abolfazl and Bunte, Kerstin and Smith, Rory and Shin, Jihye and Mastropietro, Michele and Peletier, Reynier F. and Tiňo, Peter},
journal={IEEE Transactions on Knowledge and Data Engineering},
title={LAAT: Locally Aligned Ant Technique for discovering multiple faint low dimensional structures of varying density},
year={2022},
volume={},
number={},
pages={1-1},
doi={10.1109/TKDE.2022.3177368}
}
```
