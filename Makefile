# TODO: I have to ditch this makefile in favor of a batch script.
# This would make much more sense.

MANAGER_VERSION=0.10
GATEWAY_VERSION=1.9.1

build-all: build-tensorboard build-compose-images

build-compose-images: build-gateway build-manager

build-tensorboard:
	docker build -t tensorboard tensorboard

build-gateway:
	cd cloudsimplus-gateway && ./gradlew build --warning-mode all
	cd cloudsimplus-gateway && ./gradlew dockerBuildImage

build-gateway-debug:
	cd cloudsimplus-gateway && ./gradlew build --warning-mode all -Dlog.level=DEBUG
	cd cloudsimplus-gateway && ./gradlew dockerBuildImage

build-manager:
	docker build -t manager:${MANAGER_VERSION} rl-manager

run-tensorboard:
	docker run --rm --name tensorboard -t -d -v ./tb-logs/:/tb-logs/ -p 80:6006 tensorboard

run-compose-detached:
	docker compose up -d

run-compose-detached-build:
	docker compose up -d --build

rmi-compose-images: rmi-gateway rmi-manager

rmi-tensorboard:
	docker rmi --force tensorboard:latest

rmi-gateway:
	docker rmi --force gateway:${GATEWAY_VERSION}

rmi-manager:
	docker stop manager && docker rmi --force manager:${MANAGER_VERSION}

clean-gateway:
	cd cloudsimplus-gateway && ./gradlew clean

prune-all:
	docker system prune -f
	docker volume prune -f
	docker container prune -f
	docker image prune -f

.PHONY: build-all build-compose-images build-tensorboard build-gateway \
build-gateway-debug build-manager run-tensorboard run-compose-detached \
run-compose-detached-build rmi-compose-images rmi-tensorboard rmi-gateway \
rmi-manager clean-gateway prune-all
 
