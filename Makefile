# Root Makefile — delegates to common/Makefile with domain selection
# Usage: make run domain=vm-management
#        make run domain=job-placement

DOMAIN := $(domain)
ifeq ($(DOMAIN),)
$(error DOMAIN is required. Usage: make run domain=vm-management or make run domain=job-placement)
endif

$(info Using domain=$(DOMAIN))

# Delegate everything to common/Makefile
include common/Makefile