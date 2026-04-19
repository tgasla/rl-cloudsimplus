# Root Makefile — delegates to common/Makefile with paper selection
# Usage: make run paper=main    (default)
#        make run paper=euromlsys
#        make build paper=main

PAPER := $(or $(paper),main)

$(info Using paper=$(PAPER))

# Delegate everything to common/Makefile
include common/Makefile
