#!/bin/bash

if [ -d venv ]; then
    source venv/bin/activate
else
    echo "performing initial setup"

    # create and activate virtual environment
    python3 -m venv venv || exit 1
    source venv/bin/activate

    # install dependencies
    pip install -r requirements.txt

    # add pim module path to site packages
    site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
    realpath lib > $site_packages/pim.pth

    # create an ipykernel for jupyter
    python -m ipykernel install --user --name pim
fi
