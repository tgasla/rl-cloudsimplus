MANAGER_VERSION=0.10
GATEWAY_VERSION=2.0.0

build: build-tensorboard build-compose-images

build-compose-images: build-gateway build-manager

build-tensorboard:
	docker build -t tensorboard tensorboard

build-gateway-debug:
	cd cloudsimplus-gateway && ./gradlew build --warning-mode all -Dlog.level=DEBUG
	cd cloudsimplus-gateway && ./gradlew dockerBuildImage

build-gateway:
	cd cloudsimplus-gateway && ./gradlew build --warning-mode all -Dlog.level=${LOG_LEVEL}
	cd cloudsimplus-gateway && ./gradlew dockerBuildImage

build-manager:
	docker build -t manager:${MANAGER_VERSION} rl-manager

upgrade-gradle:
	cd cloudsimplus-gateway && ./gradlew wrapper --gradle-version=${GRADLE_VERSION} --distribution-type=bin

run-tensorboard:
	docker run --rm --name tensorboard -d -v ./logs/:/logs/ -p 80:6006 tensorboard

run-cpu:
	scripts/run_docker_cpu.sh

run-gpu:
	scripts/run_docker_gpu.sh

run-gpu-attached:
	ATTACHED=true scripts/run_docker_gpu.sh

clean-gateway:
	cd cloudsimplus-gateway && ./gradlew clean

wipe-logs:
	cd logs && rm -rf *

stop:
	docker compose down --remove-orphans
	docker system prune -f
	docker system prune --volumes -f

clear-gradle:
	cd ~/.gradle && rm -rf *

.PHONY: build build-compose-images build-tensorboard build-gateway \
	build-manager upgrade-gradle run-tensorboard run-cpu \
	run-gpu clean-gateway wipe-logs stop
