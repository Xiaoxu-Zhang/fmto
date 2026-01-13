# Code Conventions

This document outlines the coding rules for all
[PyFMTO](https://github.com/Xiaoxu-Zhang/pyfmto)-based projects.

---

## ❌ Avoid Absolute Imports

When you implement a new `algorithm`/`problem`, you **should** use relative imports to ensure
availability. 

Absolute imports that hardcode the top-level package name (e.g., `algorithms.xxx`, `problems.xxx`) 
**are not allowed**, because they will cause import errors when the package/module is loaded by 
PyFMTO.

**Bad Examples:**

- ❌ `from algorithms.DEMO.demo_utils import Actions` 
- ❌ `from algorithms.BO.bo_utils import ThompsonSampling`
- ❌ `from problems.benchmarks import Ackley`

## ✅ Use Relative Imports

**Good Examples:**

- ✅ `from .demo_utils import Actions`
- ✅ `from ..BO.bo_utils import ThompsonSampling`
- ✅ `from ..benchmarks import Ackley`

## 📦 Export Public Classes

To enable seamless discovery and usage by PyFMTO, **every algorithm or problem module must expose 
its main classes in its `__init__.py` file** using explicit imports.

✅ Required Practice

Add import statements for all public-facing classes in the corresponding `__init__.py`, for example:

- Algorithm Module `algorithms/DEMO/__init__.py`
    ```python
    from .demo_client import DemoClient
    from .demo_server import DemoServer
    ```
- Problem Module `problems/demo/__init__.py` 
    ```python
    from .demo import Demo
    ```

## 🔬 Check Availability

The implemented algorithms/problems are labeled by True/False in the 'pass' column. The msg column 
will tell you why the algorithm/problem is not available if 'pass' is False. for example:

```bash
# Check the availability of algorithms
pyfmto list alg
```

| name | pass  | path                 | msg                                  |
|------|-------|----------------------|--------------------------------------|
| ALG1 | True  | fmto.algorithms.ALG1 |                                      |
| ALG2 | False | fmto.algorithms.ALG2 | Exception: No module named 'xxx'     |
| ALG3 | True  | fmto.algorithms.ALG3 | The subclass of 'Client' not found.  |

> `ALG1` is an example that PyFMTO can load correctly.</br>
> `ALG2` is an example that requires the `xxx` package to be installed.</br>
> `ALG3` is an example that PyFMTO not found the subclass of `Client` in `ADDFBO`.

```bash
# Check the availability of problems
pyfmto list prob
```

| name  | pass  | path                | msg                                           |
|-------|-------|---------------------|-----------------------------------------------|
| prob1 | True  | fmto.problems.prob1 |                                               |
| prob2 | False | fmto.problems.prob2 | The subclass of 'MultiTaskProblem' not found. |

> `prob1` is an example that PyFMTO can load correctly.</br>
> `prob2` is an example that PyFMTO not found the subclass of `MultiTaskProblem` in `prob2`.

`problems.benchmarks`, for example, is a set of SingleTaskProblem for other MultiTaskProblem 
implementations. so its `__init__.py` doesn't export any subclass of `MultiTaskProblem` classes.