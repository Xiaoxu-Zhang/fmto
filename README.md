# Federated Many-task Optimization Algorithms (FMTOAs)

This repository holds the code for the papers on FMTO algorithms.

<p align="center">
    <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-demo.gif"/>
<p>

## Have a try!

Clone this repository to your local machine:

```bash
git clone https://github.com/Xiaoxu-Zhang/fmto.git
cd fmto
```

Create an environment (`conda` is recommended) and install PyFMTO:

```bash
conda create -n fmto python=3.10
conda activate fmto
pip install pyfmto
# Install requirements for algorithms
pip install -r algorithms/ADDFBO/requirements.txt
pip install -r algorithms/BO/requirements.txt
```

> To run other algorithms, install their requirements first.

Start the experiments:

```bash
pyfmto run
```

Generate reports:

```bash
pyfmto report
```

The reports will be saved in the folder `out/results/<today>`

> **Note**: 
> 1. PyFMTO saves the results of each repeat experiment in a separate file to the folder
> `out/results/<algorithm>/<problem>/<NPD>/Run <run_id>.msgpack`, where `<NPD>` is IID (if 
> np_per_dim=1) or NIIDk (if np_per_dim=k > 1).
> 2. PyFMTO support continuous experiments, you can stop the experiment at any time and resume it 
> later by running `pyfmto run` again. 
> 3. PyFMTO will not save the results of the stopped experiments, in other words, only the 
> results of
> the normally finished experiments will be saved. 
> 4. To terminate the experiments, press `Ctrl+C` (Windows) or `Command+C` (macOS) in the terminal.

## Structure

<div align="center">
  <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/fmto-relation.svg" width="80%">
</div>

This figure is to elaborate the relationship between `fmto` and `PyFMTO`. The Algorithm API and 
Problem API shows in the upper part of this figure are the APIs provided by `PyFMTO`.

It is designed to provide a platform for researchers to compare and evaluate the performance of 
different FMTO algorithms. The repository is built on top of the PyFMTO library, which provides 
a flexible and extensible framework for implementing FMTO algorithms.

## Templates

### Algorithm

The `ALG` can be used as a template for implementing a new algorithm, you can make a copy of 
`ALG` and follow the instructions to implement a new one.

> Note: The package name is the algorithm name in the experiment.

### Problem

The `PROB` can be used as a template for implementing a new problem, you can make a copy of 
`PROB` and follow the instructions to implement a new one.

### Config

The `config.yaml` can be used as a template for configuring the experiments, you can follow the 
instructions to configure the experiments.

## About Config

### Minimal Config

The `minimal.yaml` file contains only the required parameters and can be used to run the 
experiments.

```bash
pyfmto run -c minimal.yaml
```

And generate reports:

```bash
pyfmto report -c minimal.yaml
```

### Problem Config

1. There might be some problem not support other different dim, e.g. SvmLandmine, a realworld 
   problem.
2. You can get the default value of all parameters of problems, algorithms and report formats by 
   pyfmto CLI.

## PyFMTO CLI

PyFMTO provides a command-line interface (CLI) for running experiments, analyzing results and 
get helps. The CLI layers are as follows:

```txt
pyfmto
   ├── -h/--help
   ├── run [-c/--config <config_file>]
   ├── report [-c/--config <config_file>]
   ├── list algorithms/problems/reports
   └── show <result of list>
```

Examples:
- Get help:
    ```bash
    pyfmto -h # or ↓
    # pyfmto --help
    # pyfmto list -h
    ```
- Run experiments:
    ```bash
    pyfmto run # or ↓
    # pyfmto run -c config.yaml
    ```
- Generate reports:
    ```bash
    pyfmto report # or ↓
    # pyfmto report -c config.yaml
    ```
- List something:
    ```bash
    pyfmto list algorithms
    ```
    output:
    ```txt
    Found 6 Algorithms:
    FDEMD
    ADDFBO
    BO
    FMTBO
    IAFFBO
    ALG
    ```
- Show supported configurations:
    
    ```bash
    pyfmto show ALG
    # pyfmto will automatically find the name in 'algorithms', 'problems' and 'reports'
    ```
    
    output:
    ```txt
    client:                                                                                                                                                                                                         ─╯
      alpha: 0.2
    
    server:
      beta: 0.5
    ```