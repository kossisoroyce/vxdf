# Makefile for building the VXDF documentation with Sphinx.
# Usage examples:
#     make html        # build HTML docs into docs/_build/html
#     make clean       # remove build artifacts

SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@echo "Available targets:"
	@echo "  html   - build HTML documentation"
	@echo "  clean  - remove build artifacts"

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)
