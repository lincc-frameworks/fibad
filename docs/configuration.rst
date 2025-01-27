Configuration
=============

The configuration system
------------------------
Fibad uses configuration files in the TOML syntax to manage runtime settings.
A choice was made early on to limit the number of command line arguments to support
reproducibility and clarity in the code.

<Describe the tiered system of configuration files>

The default configuration file
------------------------------

The following is the default FIBAD runtime configuration file.

.. literalinclude:: ../src/fibad/fibad_default_config.toml
   :language: text
   :linenos: