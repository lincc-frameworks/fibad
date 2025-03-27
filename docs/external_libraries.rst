External libraries
==================

Hyrax supports external libraries for models and data sets.

Two basic conditions must be met to use an external model or dataset library:
#. The relevant class must be defined under the appropriate decorator (``@hyrax_model`` or ``@hyrax_dataset``)
#. The name of the class must be noted in the hyrax config. ``[model]`` ``name`` for models, or ``[data_set]`` ``name`` for data sets

Configuring an external class
-----------------------------

The ``name`` configuration under either the ``[model]`` or ``[data_set]`` config sections is the dotte python 
name used to locate the class starting at the top package level. e.g. if your dataset class is called ``MyDataSet`` and 
is in a package called ``mydataset``, then you would configure as follows:

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

            import hyrax
            f = hyrax.Hyrax()
            f.config["data_set"]["name"] = "mydataset.MyDataSet"

    .. group-tab:: CLI

        .. code-block:: bash

            $ cat hyrax_config.toml
            [data_set]
            name = "mydataset.MyDataSet"

Datasets in the current notebook, or within your own package can simply be referred to by their class names without any dots.

It is a valid usage of this extensibility to write your own dataset or model inline in the notebook where you 
are using Hyrax. Just be sure to re-run the cell where your model class is defined when you change it!

Defining a model
----------------

Models must (for now) be written as a subclasses of ``torch.nn.Module`` and use pytorch for computation, and 
be decorated with ``@hyrax_model``. Models must minimally define ``__init__``, ``forward``, and ``train_step`` 
methods.

``__init__(self, config, shape)``
.................................
On creation of your model Hyrax passes the entire Hyrax config as a nested dictionry in the ``config`` argument
as well as the shape of each item in the dataset we intend to run on in the ``shape`` argument. This data is provided 
to allow your model class to adjust architecture or check that the provided dataset will work appropriately.

``shape`` is a tuple having the length of each individually iterable axis. An image dataset consisting of 
250x250 px images with 3 color channels each might have a shape of (3, 250, 250) indicating that the color channels are 
the first iterable axis of the tensor.


``forward(self, x)``
....................
Hyrax calls this function evaluates your model on a single input ``x``. ``x`` is gauranteed to be a tensor with 
the shape passed to ``__init__``. 

``forward()`` ought return a tensor with the output of your model.


``train_step(self, batch)``
...........................
This is called several times every training epoch with a batch of input tensors for your model, and is the 
inner training loop for your model. This is where you compute loss, perform back propagation, etc depending on 
how your model is trained.

``train_step`` returns a dictionary with a "loss" key who's value is a list of loss values for the individual 
items in the batch. This loss is logged to MLflow and tensorboard.

Defining a dataset class
------------------------

Dataset classes are written (for now) as subclasses of ``torch.utils.data.Dataset``, and decorated with the 
``@hyrax_model`` decorator. Datasets must minimally define the methods below. These are similar in form to 
Torch's `Map-style datasets <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_

``__init__(self, config, split)``
.................................
On creation of your dataset Hyrax passes the entire Hyrax config as a nested dictionry in the ``config`` 
argument. Hyrax also passes you the name of the dataset split this object will be iterating over with 
``__getitem__`` and ``__len__`` below. The split will be either ``"train"``, ``"validate"``, ``"test"``, 
or ``None`` meaning the entire dataset.

Your dataset class is (for now) in charge of implementing the split and randomness defined in the 
``[data_set]`` configs, see :ref:`data_set_splits`

``shape(self)``
...............
Return a shape tuple corresponding to the standard size of tensors in your data set. An image dataset 
consisting of 250x250 px images with 3 color channels each might have a shape of (3, 250, 250) indicating that
the color channels are the first iterable axis of each image tensor.

``ids(self)``
.............
Return a list of IDs for the objects in your dataset. These IDs ought be returned as a string generator

``__getitem(self, idx:int)``
............................
Return a single item in your dataset from a zero-based index. If your dataset was passed a split, 

``__len__(self)``
.................
Return the length of your dataset,