Model evaluation
================

One goal of fibad is to make model evaluation easier. Many tools exist for visualization
and evaluation of models. Fibad integrates with tensorboard and MLFlow to provide
easy access to these tools.

Tensorboard
-----------

Fibad automatically logs training, validation and gpu metrics to tensorboard while
training a model. This allows for easy visualization of the training process.

For more info about tensorboard see <tensorboard link>


MLFlow
------

Fibad supports MLFlow for model tracking and experiment management.
Talk about the defaults that were selected (experiment_name = notebook) and how
to get MLFlow started at the command line. 
``mlflow ui --port 5000 --backend-store-uri <path>``

For more info about MLFlow see <mlflow link>



