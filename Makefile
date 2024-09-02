MANAGER_VERSION=0.10
GATEWAY_VERSION=2.0.0

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

upgrade-gradle:
	./gradlew wrapper --gradle-version=8.10 --distribution-type=bin

run-tensorboard:
	docker run --rm --name tensorboard -t -d -v ./logs/:/logs/ -p 80:6006 tensorboard

run-sim-cpu:
	docker compose up --build

run-sim-gpu:
	docker compose --profile cuda up --build

run-sim-gpu-d:
	docker compose --profile cuda up --build -d

rmi-compose-images: rmi-gateway rmi-manager

rmi-tensorboard:
	docker rmi --force tensorboard:latest

rmi-gateway:
	docker rmi --force gateway:${GATEWAY_VERSION}

rmi-manager:
	docker stop manager && docker rmi --force manager:${MANAGER_VERSION}

clean-gateway:
	cd cloudsimplus-gateway && ./gradlew clean

wipe-logs:
	cd logs && rm -rf *
stop:
	docker compose down
	docker system prune -f

.PHONY: build-all build-compose-images build-tensorboard build-gateway \
build-gateway-debug build-manager upgrade-gradle run-tensorboard run-sim-cpu \
run-sim-gpu rmi-compose-images rmi-tensorboard rmi-gateway \
rmi-manager clean-gateway stop
