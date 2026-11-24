# Contributing to ELB-KAN-PINNs

For a bug report, include the command you ran, the Python and PyTorch versions, whether you used CPU or CUDA, and the shortest traceback or example that reproduces the problem. For a numerical discrepancy, identify the PDE, forward or inverse setting, seed, basis pair, and whether the reported value comes from validation or test coordinates.

For a code change, explain which part of the model or evaluation it affects. From the repository root, run:

```sh
python examples/check_heat.py
```

This smoke check exercises the model on CPU; it does not validate changes to other PDEs, frozen-model selection, or archived metrics. For those changes, report the additional checks you ran and which results you could not verify. Document new experimental settings separately from the configurations used for the manuscript.

Do not attach private experiment archives, checkpoints, credentials, or third-party data to issues or pull requests. For access to research data, use the contact listed in the [README](README.md#reproducing-archived-results).
