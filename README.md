# path-integration-memory
Memory modelling for insect path integration.

## Getting started
It is recommended to use a virtual Python environment,
which can be done by sourcing the `env.sh` script:

    source env.sh

### Using with Jupyter

The `env.sh` script also creates an IPython kernel for use with Jupyter;
choose the `pim` kernel under Kernel -> Change Kernel...

Jupyter's saving of cell outputs can be a nice way to share results,
but it easily wreaks havoc on repositories if one does not pay attention.
It is a good idea to Edit -> Clear All Outputs before committing a notebook,
and a Git hook can be used to help with remembering to do so. Install it like so:

    cp git-hooks/pre-commit .git/hooks/pre-commit

If the hook stops the commit, but you explicitly want to commit cell output,
ignore the hook using `--no-verify` when committing.
