# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
SOURCEDIR     = .
BUILDDIR      = _build

# Define destination and filename for the Brython package bundle
LIB_DEST      = special_pages/libs
LIB_BUNDLE    = vkdispatch.brython.js
LIB_STAGE     = $(LIB_DEST)/.vkdispatch_stage

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile bundle_lib

# Target to bundle the library into a single Brython package file
bundle_lib:
	@echo "Bundling vkdispatch for Brython..."
	@$(PYTHON) -c "import brython" > /dev/null
	@rm -rf "$(LIB_DEST)/vkdispatch"
	@mkdir -p "$(LIB_DEST)"
	@rm -f "$(LIB_DEST)/$(LIB_BUNDLE)"
	@rm -f "$(LIB_DEST)/vkdispatch_native.brython.js"
	@rm -rf "$(LIB_STAGE)"
	@mkdir -p "$(LIB_STAGE)"
	@cp -r ../vkdispatch "$(LIB_STAGE)/vkdispatch"
	@cp -r special_pages/libs/vkdispatch_native "$(LIB_STAGE)/vkdispatch_native"
	@cd "$(LIB_STAGE)" && $(PYTHON) -m brython make_package vkdispatch \
		--src-dir . \
		--output-path "$(CURDIR)/$(LIB_DEST)/$(LIB_BUNDLE)"
	@rm -rf "$(LIB_STAGE)"

# Intercept the "html" target to run bundle_lib first
html: bundle_lib
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
