# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Define source and destination for the library copy
LIB_SOURCE    = ../vkdispatch
LIB_DEST      = special_pages/libs/vkdispatch

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile copy_lib

# Target to copy the library files
copy_lib:
	@echo "Copying library files from $(LIB_SOURCE) to $(LIB_DEST)..."
	@rm -rf "$(LIB_DEST)"
	@mkdir -p "$(LIB_DEST)"
	@cp -r "$(LIB_SOURCE)/." "$(LIB_DEST)/"

# Intercept the "html" target to run copy_lib first
html: copy_lib
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)