# Root Makefile — delegates to common/Makefile with domain selection
# Usage:
#   make run domain=job-placement
#   make build-tensorboard          # no domain needed
#   make run-tensorboard            # no domain needed

DOMAIN := $(domain)

# Targets that work without a domain
DOMAIN_FREE_TARGETS := build-tensorboard run-tensorboard generate-proto stop get-gradle clear-gradle build-manager

# If MAKECMDGOALS is empty the default target (build) runs, which needs domain.
# Otherwise require domain only when at least one goal is not in the free list.
_NEEDS_DOMAIN := $(if $(MAKECMDGOALS),$(filter-out $(DOMAIN_FREE_TARGETS),$(MAKECMDGOALS)),yes)
ifneq ($(_NEEDS_DOMAIN),)
  ifeq ($(DOMAIN),)
    $(error DOMAIN is required for this target. Usage: make <target> domain=vm-management)
  endif
  $(info Using domain=$(DOMAIN))
endif

include common/Makefile
