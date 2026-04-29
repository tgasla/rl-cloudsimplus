# domain=vm-management or domain=job-placement — defaults to vm-management
DOMAIN ?= vm-management
DOMAIN_DIR := domain/$(DOMAIN)

CONFIG_FILE := $(DOMAIN_DIR)/config.yml
get_yaml_value = $(shell grep -A 20 '^globals:' $(CONFIG_FILE) | grep -m 1 '^ *$(1):' | sed 's/^ *$(1): //')

# Read versions from versions.gradle — single source of truth
GRADLE_VERSION := $(shell grep "gradleVersion" common/versions.gradle | sed "s/.*gradleVersion = '\([^']*\)'.*/\1/")
MANAGER_VERSION := $(shell grep "managerVersion" common/versions.gradle | sed "s/.*managerVersion = '\([^']*\)'.*/\1/")
GATEWAY_VERSION := $(shell grep "gatewayVersion" common/versions.gradle | sed "s/.*gatewayVersion = '\([^']*\)'.*/\1/")

build: build-gateway build-manager

# Canonical sources in common/
CANONICAL_PROTO = common/proto/unified/cloudsimplus.proto
CANONICAL_BUILD = common/cloudsimplus-gateway-shared/gateway-build.gradle

build-gateway:
	@mkdir -p $(DOMAIN_DIR)/cloudsimplus-gateway/src/main/proto
	@cp -p $(CANONICAL_PROTO) $(DOMAIN_DIR)/cloudsimplus-gateway/src/main/proto/
	@cp -p $(CANONICAL_BUILD) $(DOMAIN_DIR)/cloudsimplus-gateway/build.gradle
	cd common/cloudsimplus-gateway-shared && ./gradlew :$(DOMAIN):build $(ARGS)

build-manager:
	docker build -t manager:$(MANAGER_VERSION) -t gateway:$(GATEWAY_VERSION) \
		--build-arg PAPER_DIR=$(DOMAIN) \
		-f common/rl-manager/Dockerfile common/rl-manager/

build-tensorboard:
	docker build -t tensorboard $(DOMAIN_DIR)/tensorboard

run-tensorboard:
	docker run --rm --name tensorboard -d -v $(DOMAIN_DIR)/logs/:/logs/ -p 6006:6006 tensorboard

run:
	PAPER_DIR="$(DOMAIN)" ATTACHED=$(call get_yaml_value,attached) GPU=$(call get_yaml_value,gpu) PAPER=$(DOMAIN) bash common/scripts/run_docker.sh

clean-gateway:
	cd common/cloudsimplus-gateway-shared && ./gradlew :$(DOMAIN):clean 2>/dev/null || true

wipe-logs:
	rm -rf $(DOMAIN_DIR)/logs/*

stop:
	docker compose -f common/docker-compose.yml down --remove-orphans
# 	docker system prune -f
# 	docker system prune --volumes -f

# Clean all state: containers, volumes, images, gradle builds
clean-all: stop
	docker rmi -f $$(docker images -q 'manager*' 2>/dev/null) 2>/dev/null || true
	docker rmi -f $$(docker images -q 'gateway*' 2>/dev/null) 2>/dev/null || true
	docker rmi -f $$(docker images -q 'tensorboard*' 2>/dev/null) 2>/dev/null || true
	cd common/cloudsimplus-gateway-shared && ./gradlew :$(DOMAIN):clean 2>/dev/null || true
	rm -rf $(DOMAIN_DIR)/rl-manager/logs/*

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
	@echo "=== CloudSim Plus dependency for domain: $(DOMAIN) ==="
	@cd common/cloudsimplus-gateway-shared && ./gradlew :$(DOMAIN):dependencies $(ARGS) 2>&1 | grep "org.cloudsimplus"

check-gateway-jar:
	@JAR=$$(ls $(DOMAIN_DIR)/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-*.jar 2>/dev/null | head -1); \
	if [ -f "$$JAR" ]; then \
		echo "JAR: $$JAR"; \
		unzip -p "$$JAR" META-INF/maven/org.cloudsimplus/cloudsimplus/pom.properties 2>/dev/null | grep version; \
	else \
		echo "No JAR found - run 'make build-gateway' first"; \
	fi

.PHONY: build build-gateway build-tensorboard build-manager run-tensorboard run clean-gateway wipe-logs stop get-gradle clear-gradle check-gateway-deps check-gateway-jar
