# paper=main or paper=euromlsys — defaults to main
PAPER ?= main
PAPER_DIR := papers/$(PAPER)

CONFIG_FILE := $(PAPER_DIR)/config.yml
get_yaml_value = $(shell grep -A 20 '^globals:' $(CONFIG_FILE) | grep -m 1 '^ *$(1):' | sed 's/^ *$(1): //')

# Read versions from versions.gradle — single source of truth
GRADLE_VERSION := $(shell grep "gradleVersion" common/versions.gradle | sed "s/.*gradleVersion = '\([^']*\)'.*/\1/")
MANAGER_VERSION := $(shell grep "managerVersion" common/versions.gradle | sed "s/.*managerVersion = '\([^']*\)'.*/\1/")
GATEWAY_VERSION := $(shell grep "gatewayVersion" common/versions.gradle | sed "s/.*gatewayVersion = '\([^']*\)'.*/\1/")

build: build-gateway build-manager

build-gateway:
	cd $(PAPER_DIR)/cloudsimplus-gateway && ./gradlew build --warning-mode all

build-manager:
	docker build -t manager:$(MANAGER_VERSION) -t gateway:$(GATEWAY_VERSION) -f common/rl-manager/Dockerfile .

build-tensorboard:
	docker build -t tensorboard $(PAPER_DIR)/tensorboard

run-tensorboard:
	docker run --rm --name tensorboard -d -v $(PAPER_DIR)/logs/:/logs/ -p 6006:6006 tensorboard

run:
	PAPER_DIR=$(PAPER_DIR) COMPOSE_BAKE=true ATTACHED=$(call get_yaml_value,attached) GPU=$(call get_yaml_value,gpu) PAPER=$(PAPER) common/scripts/run_docker.sh

clean-gateway:
	cd $(PAPER_DIR)/cloudsimplus-gateway && ./gradlew clean

wipe-logs:
	rm -rf $(PAPER_DIR)/logs/*

stop:
	docker compose -f common/docker-compose.yml down --remove-orphans
	docker system prune -f
	docker system prune --volumes -f

get-gradle:
	cd $(PAPER_DIR)/cloudsimplus-gateway && ./gradlew wrapper --gradle-version=$(GRADLE_VERSION) --distribution-type=bin

clear-gradle:
	cd ~/.gradle && rm -rf *

.PHONY: build build-gateway build-tensorboard build-manager run-tensorboard run clean-gateway wipe-logs stop get-gradle clear-gradle
