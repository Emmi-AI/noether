# How to build

To build the documentation locally, simply run:

```bash
make -C docs clean html
```

The documentation will be available under ``/noether/docs/_build/html/index.html`` - you can open it in your browser (
we use Chrome for easier compatibility).

---

# How to maintain

In order to provide users with up-to-date documentation and meaningful examples, we aim to have it close to our 
codebase. With that in mind, we have the following documentation code structure:

```
/docs
    /source
        /_static               # << static files, like custom CSS or similar
        /_templates            # << specific templates for auto generated docs
        /noether               # << Noether module documentation
        /guides                 
        /reference
        /tutorials             
        /explanation 
        *.rst                  # << individual documentation files unrelated to packages
        conf.py                # << Sphinx-related configuration, custom theming, etc.
        index.rst              # << the main entry point of the documentation
    Makefile                   # << provides the basic Sphinx setup
    README.md                  # << you're here
```

## In-code documentation

We follow Google-style docstrings, a typical example:

```python
"""This is an intro line.
This is some more descriptive paragraph.

Args:
    foo: Input object that holds metadata.
    bar: A multiplier in range between 0 and 1.
    
Returns:
    - tuple: A dictionary with values and an object with metrics
    
Raises:
    - RuntimeError: In case of failed compute
    - ValueError: In case of input validation
    
Examples:
    .. code-block:: python

        x = torch.randn(16)
        pipeline = MasterFactory.get("pipeline").create(pipeline_config)
"""
```
for more information check out the [Sphinx docs](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

> Make sure to use syntax highlight based on your example.
