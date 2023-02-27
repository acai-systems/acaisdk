# ACAI SDK

This repository is the ACAI SDK to interact with the ACAI system. You can find examples in the `example` folder.

## Setup

Notice that **Python3** is required.

```bash
# Clone the repo
git clone https://github.com/acai-systems/acaisdk.git
# cd to the repo root
cd acaisdk/
```

The ACAI SDK interacts with the ACAI crendential service to authenticate user and forward requests. Once you have the crendential server setup, update `acaisdk/configs.ini` **before acaisdk installation ‼️**  with the endpoint IP address and port. For Phoebe cluster, see instructions below.

```
# Install ACAI SDK (Optional)
pip3 install .

# Alternatively, you can do:
# python3 -m pip install .
```

### Configuration for Phoebe Cluster

ACAI SDK is now permanently available on the Phoebe Cluster. Dashboard is located at `phoebe-mgmt.pdl.local.cmu.edu:31500/`. To integrate with the service on Phoebe within the Jupyter Notebook, set the following:

- In `acaisdk/configs.ini`
```
[phoebe]
credential_endpoint=phoebe-mgmt.pdl.local.cmu.edu
credential_endpoint_port = 30373
```

- In jupyter notebook (before any acaisdk code is executed)
```python
os.environ["CLUSTER"] = 'PHOEBE'
```

## Run an Example

Workflow examples are located at the `example` folder. 

Folders that don't create new projects and user by default:
- CICIDS2018 (not finished yet)
- NLP-GloVe
- Sentiment-IMDB
- cifar10-ResNet (not finished yet)

Folders that create new projects and user by default:
- automl-example
- example1
- example2
- example3
- ray-example
- ray-profiler-example
- spark-tuner-example
- storage-example

The workflow should include uploading the input files, deploying a job on ACAI and downloading the output and its further examination.

ACAI system requires code to be zipped (literally zipped, no `tar` ‼️) in order to be pulled to the container for execution. Zipped codes together with input data are provided in the directories.


To interact with examples, open Jupyter Notebook in `examples` folder:

```bash
jupyter notebook
```

## Documentation

Detailed documentation on the SDK can be found at:
[ACAI SDK Docs](https://acai-systems.github.io/acaisdk/)
