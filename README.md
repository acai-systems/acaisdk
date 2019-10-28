## Installation

Notice that **Python3** is required.
```bash
git clone https://github.com/acai-systems/acaisdk.git
cd acaisdk/acaisdk/
pip3 install -r requirements.txt
```

## Run examples
A workflow examples is located at the `example` folder.
It is a wordcount task for text files. 
The workflow is to create a project and a user under
the project, upload the input files, run `wordcount.py`
on the cloud, download the output and examine it.
  
Now ACAI system requires code to be zipped in order
to be pulled to the container for execution. A zipped
code `wordcount.zip` is already provided for you.

Open Jupyter Notebook

```bash
jupyter notebook
``` 

and use `example_workflow.ipynb` notebook to run the 
workflow.


## Documentation

Detailed documentation on the SDK can be found at:
[ACAI SDK Docs](https://acai-systems.github.io/acaisdk/)
