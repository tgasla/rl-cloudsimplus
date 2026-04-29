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

# Canonical proto source is common/proto/unified/cloudsimplus.proto
# Both papers need a local copy in src/main/proto/ for gradle protobuf plugin
CANONICAL_PROTO = common/proto/unified/cloudsimplus.proto

build-gateway:
	@mkdir -p $(PAPER_DIR)/cloudsimplus-gateway/src/main/proto
	@cp -p $(CANONICAL_PROTO) $(PAPER_DIR)/cloudsimplus-gateway/src/main/proto/
	cd common/cloudsimplus-gateway-shared && ./gradlew :$(PAPER):build $(ARGS)

build-manager:
	docker build -t manager:$(MANAGER_VERSION) -t gateway:$(GATEWAY_VERSION) \
		--build-arg PAPER_DIR=$(PAPER) \
		-f common/rl-manager/Dockerfile common/rl-manager/

build-tensorboard:
	docker build -t tensorboard $(PAPER_DIR)/tensorboard

run-tensorboard:
	docker run --rm --name tensorboard -d -v $(PAPER_DIR)/logs/:/logs/ -p 6006:6006 tensorboard

run:
	PAPER_DIR="$(PAPER)" ATTACHED=$(call get_yaml_value,attached) GPU=$(call get_yaml_value,gpu) PAPER=$(PAPER) bash common/scripts/run_docker.sh

clean-gateway:
	cd common/cloudsimplus-gateway-shared && ./gradlew :$(PAPER):clean 2>/dev/null || true

wipe-logs:
	rm -rf $(PAPER_DIR)/logs/*

stop:
	docker compose -f common/docker-compose.yml down --remove-orphans
# 	docker system prune -f
# 	docker system prune --volumes -f

# Clean all state: containers, volumes, images, gradle builds
clean-all: stop
	docker rmi -f $$(docker images -q 'manager*' 2>/dev/null) 2>/dev/null || true
	docker rmi -f $$(docker images -q 'gateway*' 2>/dev/null) 2>/dev/null || true
	docker rmi -f $$(docker images -q 'tensorboard*' 2>/dev/null) 2>/dev/null || true
	cd common/cloudsimplus-gateway-shared && ./gradlew :$(PAPER):clean 2>/dev/null || true
	rm -rf $(PAPER_DIR)/rl-manager/logs/*

get-gradle:
	cd common/cloudsimplus-gateway-shared && ./gradlew wrapper --gradle-version=$(GRADLE_VERSION) --distribution-type=bin

# Pull latest CloudSim Plus from Git master, build and publish to local maven repo
# After this, use: make build-gateway ARGS="-PuseSnapshot=true"
update-cloudsimplus:
	@if [ ! -d "/tmp/cloudsimplus" ]; then \
		git clone --depth 1 https://github.com/cloudsimplus/cloudsimplus.git /tmp/cloudsimplus; \
	fi
	cd /tmp/cloudsimplus && git pull && ./mvnw clean install -DskipTests -q
	@echo "CloudSim Plus master published to mavenLocal(). Run './gradlew build -PuseSnapshot=true' to use it."

clear-gradle:
	cd ~/.gradle && rm -rf *

check-gateway-deps:
	@echo "=== CloudSim Plus dependency for paper: $(PAPER) ==="
	@cd common/cloudsimplus-gateway-shared && ./gradlew :$(PAPER):dependencies $(ARGS) 2>&1 | grep "org.cloudsimplus"

check-gateway-jar:
	@JAR=$$(ls $(PAPER_DIR)/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-*.jar 2>/dev/null | head -1); \
	if [ -f "$$JAR" ]; then \
		echo "JAR: $$JAR"; \
		unzip -p "$$JAR" META-INF/maven/org.cloudsimplus/cloudsimplus/pom.properties 2>/dev/null | grep version; \
	else \
		echo "No JAR found - run 'make build-gateway' first"; \
	fi

.PHONY: build build-gateway build-tensorboard build-manager run-tensorboard run clean-gateway wipe-logs stop get-gradle clear-gradle check-gateway-deps check-gateway-jar
