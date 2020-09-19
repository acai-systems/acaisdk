# ACAI SDK

This repository is the ACAI SDK to interact with the ACAI system. You can find examples in the `example` folder.

## Setup

### Update Crendential Server Endpoint

The ACAI SDK interacts with the ACAI crendential service to authenticate user and forward requests. Once you have the crendential server setup, update `acaisdk/configs.ini` with the endpoint IP address and port.

### Installation

Notice that **Python3** is required.

```bash
# Clone the repo
git clone https://github.com/acai-systems/acaisdk.git
# cd to the repo root
cd acaisdk/
# Install ACAI SDK (Optional)
pip3 install .

# Alternatively, you can do:
# python3 -m pip install .
```

## Run an Example

A workflow examples is located at the `example` folder. It is a wordcount task for text files.

The workflow is to create a project and a user under the project, upload the input files, run `wordcount.py` on ACAI, download the output and examine it.

Now ACAI system requires code to be zipped in order to be pulled to the container for execution. A zipped code `wordcount.zip` is already provided for you.

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then use `example_workflow.ipynb` notebook to run the workflow.

## Documentation

Detailed documentation on the SDK can be found at:
[ACAI SDK Docs](https://acai-systems.github.io/acaisdk/)
