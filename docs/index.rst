Welcome to fibad's documentation!
=================================

What is fibad?
--------------
Fibad is the Framework for Image-Base Anomaly Detection.

Why did we build fibad?
-----------------------
Image-based ML in astronomy is challenging work.
We've found many bottlenecks in the process that require signifincant effort to overcome.
Most of the time that effort doesn't accrue to science, it's just a means to an end.
But it's repeated over and over again by many different people.
Fibad is our effort to make the process easier and more efficient taking care of
the common tasks so that scientists can focus on the science.

Fibad's guiding principles
--------------------------
* **Principle 1** Empower the scientists to do science - not software engineering.
Fibad automatically discovers and uses the most performant hardware available
for training without any changes to the users code.
* **Principle 2** Make the software easy to use.
Fibad is designed to be used in a Jupyter notebook for exploration or from the
command line within HPC or Slurm environments without modification.
* **Principle 3** Make the software extensible to support many different use cases.
Fibad is designed to be easily extended to support new models and data sources.

.. toctree::
   :hidden:

   Home page <self>
   Architecture overview <architecture_overview>
   Configuration <configuration>
   External libraries <external_libraries>
   Data set splits <data_set_splits>
   Model evaluation <model_evaluation>
   Developer guide <dev_guide>
   API Reference <autoapi/index>
   Notebooks <notebooks>
