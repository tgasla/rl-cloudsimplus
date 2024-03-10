MANAGER_VERSION=0.10
GATEWAY_VERSION=1.9.1

build-all: build-tensorboard build-compose-images

build-compose-images: build-gateway build-manager

build-tensorboard:
	docker build -t tensorboard tensorboard

build-gateway:
	cd cloudsimplus-gateway && sudo ./gradlew build --warning-mode all
	cd cloudsimplus-gateway && sudo ./gradlew dockerBuildImage

build-gateway-debug:
	cd cloudsimplus-gateway && sudo ./gradlew build --warning-mode all -Dlog.level=DEBUG
	cd cloudsimplus-gateway && sudo ./gradlew dockerBuildImage

build-manager:
	docker build -t manager:${MANAGER_VERSION} rl-manager

run-tensorboard:
	docker run --rm --name tensorboard -t -d -v ./tb-logs/:/tb-logs/ -p 80:6006 tensorboard

run-compose:
	docker compose up -d

run-compose-build:
	docker compose up -d --build

rmi-compose-images: rmi-gateway rmi-manager

rmi-tensorboard:
	docker rmi --force tensorboard:latest

rmi-gateway:
	docker rmi --force gateway:${GATEWAY_VERSION}

rmi-manager:
	docker stop manager && docker rmi --force manager:${MANAGER_VERSION}

clean-gateway:
	cd cloudsimplus-gateway && sudo ./gradlew clean

prune-all:
	docker system prune -f
	docker volume prune -f
	docker container prune -f
	docker image prune -f

.PHONY: build-tensorboard build-gateway build-manager \
run-tensorboard run-compose rmi-tensorboard rmi-gateway \
rmi-manager prune-all rmi-compose-images clean-gateway
