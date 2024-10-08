# FIBAD
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/fibad/smoke-test.yml)](https://github.com/lincc-frameworks/fibad/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/lincc-frameworks/fibad/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/fibad)

<hr>

## Introduction
The **Framework for Image-Based Anomaly Detection** (FIBAD) is an efficient tool
to hunt for rare and anomalous sources in large astronomical imaging surveys
(e.g., Rubin-LSST, HSC, Euclid, NGRST, etc.). 
FIBAD is designed to support four primary steps in the anomaly detection workflow:

* Downloading large numbers of cutouts from public data repositories
* Building lower dimensional representations of downloaded images -- the latent space
* Interactive visualization and algorithmic exploration (e.g., clustering, similarity-search, etc.) of the latent space
* Identification & rank-ordering of potential anomalous objects

FIBAD is not tied to a specific anomaly detection algorithm/model or a specific
class of rare/anomalous objects; but rather intended to support any algorithm
that the user may want to apply on imaging data.
If the algorithm you want to use takes in tensors, outputs tensors, and can be
implemented in PyTorch; then chances are FIBAD is the right tool for you!

## Getting Started 
To get started with FIBAD, clone the repository and create a new virtual environment.
If you plan to develop code, run the ``.setup_dev.sh`` script.

```
>> git clone https://github.com/lincc-frameworks/fibad.git
>> conda create -n fibad python=3.10
>> bash .setup_dev.sh (Optional, for developers)
```

## Additional Information
FIBAD is under active development and has limited documentation at the moment.
We aim to have v1 stability and more documentation in the first half of 2025.
If you are an astronomer trying to use FIBAD before then, please get in touch with us!

This project started as a collaboration between different units within the
[LSST Discovery Alliance](https://lsstdiscoveryalliance.org/) --
the [LINCC Frameworks Team](https://lsstdiscoveryalliance.org/programs/lincc-frameworks/)
and LSST-DA Catalyst Fellow, [Aritra Ghosh](https://ghosharitra.com/).

## Acknowledgements

This project is supported by Schmidt Sciences.