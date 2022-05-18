# path-integration-memory
Memory modelling for insect path integration.


## Getting started
It is recommended to create a virtual Python environment:

    python3 -m venv venv

Activate it by sourcing its `activate` script:

    source venv/bin/activate

Install the required packages using Pip:

    pip3 install -r requirements.txt

The repository's `lib` module can be added to the virtual environment
like so:

    pwd > venv/lib/python*/site-packages/pim.pth

For subsequent sessions only the sourcing step needs to be repeated:

    source venv/bin/activate


### Using with Jupyter

To make the above virtual environment available as a Jupyter kernel,
make sure it is activated and run:

    python3 -m ipykernel install --user --name pim

The `pim` kernel can then be selected from Jupyter.

