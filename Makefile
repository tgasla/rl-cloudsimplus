MANAGER_VERSION=0.10
GATEWAY_VERSION=1.9.1

all: gateway manager tensorboard

run:
	docker compose up -d

gateway:
	cd cloudsimplus-gateway && sudo ./gradlew build --warning-mode all
	cd cloudsimplus-gateway && sudo ./gradlew dockerBuildImage

manager:
	docker build -t manager:${MANAGER_VERSION} rl-manager

tensorboard:
	docker build -t tensorboard tensorboard

.PHONY: clean-all clean-tensorboard clean-manager clean-gateway

clean-tensorboard:
	docker rmi --force tensorboard

clean-manager:
	docker rmi --force manager:${MANAGER_VERSION}

clean-gateway:
	docker rmi --force gateway:${GATEWAY_VERSION}

prune:
	docker system prune -f
	docker volume prune -f
	docker container prune -f
	docker image prune -f

clean-all:
	docker system prune -f
	docker volume prune -f
	docker container prune -f
	docker image prune -f
	cd cloudsimplus-gateway && ./gradlew clean
	docker rmi --force manager:${MANAGER_VERSION} gateway:${GATEWAY_VERSION}
