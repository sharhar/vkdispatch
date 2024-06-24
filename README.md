# vkdispatch
A Python module for orchestrating and dispatching large computations across multi-GPU systems using Vulkan.


## Instillation

The vkdispatch package can be installed via Pypi using

```
pip install vkdispatch
```

### Local instillation

If you want a local install of vkdispatch (e.g. for development purposes), then use the following steps to build from source.
Note that its recommended to use a Python environment manager for development

```
git clone https://github.com/sharhar/vkdispatch.git
cd vkdispatch
python fetch_dependencies.py
pip install -r requirements.txt
pip install -e .
```
