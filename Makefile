# Root Makefile — delegates to common/Makefile with domain selection
# Usage: make run domain=vm-management  (default)
#        make run domain=job-placement
#        make build domain=vm-management

DOMAIN := $(or $(domain),vm-management)

$(info Using domain=$(DOMAIN))

# Delegate everything to common/Makefile
include common/Makefile