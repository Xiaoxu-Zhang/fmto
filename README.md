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

Create an environment and install dependencies:

```bash
conda create -n fmto python=3.10
conda activate fmto
# Install basic requirements
pip install -r requirements.txt
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

The reports will be saved in the folder `path/to/results/<today>`

> **Notes**: 
> 1. PyFMTO saves the results of each repeat experiment in a separate file to the folder
> `path/to/results/<algorithm>/<problem>/NPD<npd>/Run <run_id>.msgpack`, where `npd` is the number 
> of partitions per dimension.
> 2. PyFMTO support continuous experiments, you can stop the experiment at any time and resume it 
> later by running `pyfmto run` again. 
> 3. PyFMTO will not save the results of the stopped experiments, in other words, only the 
> results of
> the normally finished experiments will be saved. 
> 4. To terminate the experiments, press `Ctrl+C` (Windows/Linux) or `Command+C` (macOS) in the terminal.

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

Before implementing a new algorithm or problem, it is recommended to read the
[CONVENTIONS](CONVENTIONS.md)

- **Algorithm**: The `algorithms/DEMO` can be used as a template for implementing a new algorithm, 
 you can make a copy of `DEMO` and follow the instructions to implement a new one.
- **Problem**: The `problems/demo` can be used as a template for implementing a new problem, you 
 can make a copy of `demo` and follow the instructions to implement a new one.
- **Config**: The `config.yaml` provides instructions for configuring your own experiments.

> Note: The package name is the algorithm/problem name used in the configuration.

## Configuration

PyFMTO allows you to configure the experiments through a configuration file. It will try to use 
`config.yaml` if no configuration file is specified through the `-c/--config <filename>` option in
CLI. 

If `config.yaml` does not exist and no another configuration file is specified, the `run` and 
`report` commands will fail.

### Minimal

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
   `pyfmto show <xxx>`.

### Multiple Sources

If there are multiple sources of algorithms/problems implemented, you can specify the additional
sources by adding the `<absolute path to source>` to the `sources` section in `config.yaml`. For
example, if there are two sources:

```text
/absolute/path/to/source1
    ├──algorithms
    └──problems

/absolute/path/to/source2
    ├──algorithms
    └──problems
```

Specify the sources in config file:

```yaml
launcher:
  sources: [/absolute/path/to/source1, /absolute/path/to/source2]
  # other configuration
```

> The current working directory, where the launcher is executed, will be the only source if no 
> additional sources are provided in config file.

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