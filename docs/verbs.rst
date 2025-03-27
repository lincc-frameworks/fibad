Hyrax Verbs
===========
The term "verb" is used to describe the functions that Hyrax supports.
For instance, the ``train`` verb is used to train a model.
Each of the builtin verbs are detailed here.


``train``
---------
Train a model

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           f = Hyrax()

           # Train the model specified in the configuration file
           f.train()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax train


``infer``
---------
Run inference using a model

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           f = Hyrax()

           # Train the model specified in the configuration file
           f.infer()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax infer


``umap``
--------
Run UMAP on the output of inference or a dataset

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           f = Hyrax()

           # Train the model specified in the configuration file
           f.umap()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax umap

``visualize``
-------------
Interactively visualize embedded space produced by UMAP.
Due to the fact that the visualization is interactive, it is not available in the CLI.

.. code-block:: python

    from hyrax import Hyrax

    # Create an instance of the Hyrax object
    f = Hyrax()

    # Train the model specified in the configuration file
    f.visualize()

