MANAGER_VERSION=0.11
GRADLE_VERSION=9.1.0

CONFIG_FILE=config.yml
get_yaml_value = $(shell grep -A 20 '^globals:' $(CONFIG_FILE) | grep -m 1 '^ *$(1):' | sed 's/^ *$(1): //')

build: build-gateway build-manager

build-gateway:
	cd cloudsimplus-gateway && ./gradlew build --warning-mode all -Dlog.level=$(call get_yaml_value,java_log_level) -Dlog.destination=$(call get_yaml_value,java_log_destination) -Djunit.output.show=$(call get_yaml_value,junit_output_show)

build-manager:
	docker build -t manager:$(MANAGER_VERSION) -f rl-manager/Dockerfile .

build-tensorboard:
	docker build -t tensorboard tensorboard

run-tensorboard:
	docker run --rm --name tensorboard -d -v ./logs/:/logs/ -p 6006:6006 tensorboard

run:
	COMPOSE_BAKE=true ATTACHED=$(call get_yaml_value,attached) GPU=$(call get_yaml_value,gpu) scripts/run_docker.sh

clean-gateway:
	cd cloudsimplus-gateway && ./gradlew clean

wipe-logs:
	cd logs && rm -rf *

stop:
	docker compose down --remove-orphans
	docker system prune -f
	docker system prune --volumes -f

get-gradle:
	cd cloudsimplus-gateway && ./gradlew wrapper --gradle-version=${GRADLE_VERSION} --distribution-type=bin

clear-gradle:
	cd ~/.gradle && rm -rf *

.PHONY: build build-tensorboard build-gateway build-manager run-tensorboard clean-gateway wipe-logs stop
