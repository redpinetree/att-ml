# Description

This repository contains code for the optimization of adaptive tensor trees using either a Born machine-based approach or a nonnegative approach (or both, in a hybrid scheme). An explanation of the methods implemented in this repository can be found in [our preprint](https://arxiv.org/abs/2504.06722).

## Building

- Building the executable
	```
	make
	```
- Cleanup and removal of temporary files
	```
	make clean
	```

## Usage

After building, the code can be executed by invoking:

```
OMP_NUM_THREADS={number of threads} ./bin/att_ml -i {training data} -L {training labels} -V {test data}  -B {test labels} -T {initial tree type} -r ${maximum rank} -N {number of iterations} -l {learning rate} -b {batch size} -o {output path prefix} -S {RNG seed} [--born|--hybrid] [--struct-opt]
```

The provided scripts contain example commands for the provided datasets.

The following are some details on the options. First, I/O related options:
- `-i` - path to the file containing the training dataset. (required)
- `-L` - path to the file containing the training dataset labels.
- `-V` - path to the file containing the test dataset.
- `-B` - path to the file containing the training dataset labels.
- `-o` - output path prefix for the output of the method.
If `-L` is not set, only `-V` should be supplied if there is a test dataset present. If `-L` is set, then to use a test dataset, `-V` and `-B` must be supplied.

Next, simulation hyperparameters:
- `-T` - initial tree type. Choices include "mps" (matrix product state -- a linear ansatz), "rand" (a randomly-drawn tree structure, and "pbttn" (a perfectly balanced binary tensor tree -- requires that the number of input features is a power of 2).
- `-r` - maximum rank or bond dimension that intermediate bonds are allowed to have in the tensor tree
- `-N` - maximum number of iterations to perform. Here, an iteration refers to the whole optimization step along one bond in the network, and iteration through all the bonds in the network constitutes a sweep.
- `-l` - base learning rate used in the AdamW optimizer.
- `-b` - batch size for the training dataset.
- `-S` - seed for the RNG

As for the flags that control the optimization approach:
- `--born` - use a Born machine defined on an adaptive tensor tree as the optimization scheme. By default, if neither `--born` nor `--hybrid` are supplied, the code uses a nonnegative adaptive tensor tree scheme.
- `--hybrid` - use a hybrid training scheme where a Born machine approach is first used, followed by a second training stage using a nonnegative approach. By default, if neither `--born` nor `--hybrid` are supplied, the code uses a nonnegative adaptive tensor tree scheme.
- `--struct-opt` - enable structural optimization. By default, the algorithm used does not take into account optimization of the structure, so without this option, the structure is fixed.

### Running the examples

The file `data.zip` contains three subfolders:
- `data_gens` contains Python scripts for generating sample data
- `datasets` contains sample datasets prepared using the aforementioned scripts
- `scripts` contains Bash shell scripts that run the code on the sample datasets

To run the examples, extract the contents of `data.zip` into the same directory as the `bin` and `cpp` folders and run a script from the `scripts` folder.