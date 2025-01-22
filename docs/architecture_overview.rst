Architecture overview
=====================

Fibad uses verbs
----------------
Fibad defines a set of commands, called verbs, that are the primary mode of interaction.
Verbs are meant to be intuitive and easy to remember. For instance, to train a model,
you would use the ``train`` verb.
To use a trained model for inference, you would use the ``infer`` verb.

Notebook, CLI, or Both
--------------------------------
Fibad is designed to be used in a Jupyter notebook or from the command line without
modification. This supports exploration and development in a familiar notebook environment
and deployment to an HPC or Slurm system for large scale training.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from fibad import Fibad

           fibad_instance = Fibad(config_file = 'my_config.toml')
           fibad_instance.train()

    .. group-tab:: CLI

        .. code-block:: bash

           >> fibad train -c my_config.toml
