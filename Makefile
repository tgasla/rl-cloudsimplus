MANAGER_VERSION=0.10
GATEWAY_VERSION=2.0.0
GRADLE_VERSION=8.5

CONFIG_FILE=config.yml
get_yaml_value = $(shell grep -A 20 '^globals:' $(CONFIG_FILE) | grep -m 1 '^ *$(1):' | sed 's/^ *$(1): //')

build: build-tensorboard build-gateway

build-tensorboard:
	docker build -t tensorboard tensorboard

build-gateway:
	cd cloudsimplus-gateway && ./gradlew build --warning-mode all -Dlog.level=$(call get_yaml_value,java_log_level) -Dlog.destination=$(call get_yaml_value,java_log_destination) -Djunit.output.show=$(call get_yaml_value,junit_output_show)
	cd cloudsimplus-gateway && ./gradlew dockerBuildImage

run-tensorboard:
	docker run --rm --name tensorboard -d -v ./logs/:/logs/ -p 80:6006 tensorboard

run:
	ATTACHED=$(call get_yaml_value,attached) GPU=$(call get_yaml_value,gpu) scripts/run_docker.sh

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

.PHONY: build build-compose-images build-tensorboard build-gateway \
	build-manager upgrade-gradle run-tensorboard run-cpu \
	run-gpu clean-gateway wipe-logs stop
