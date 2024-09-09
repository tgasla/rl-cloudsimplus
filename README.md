# RL-CloudSimPlus

<div align="center">
<a href="https://app.codacy.com/gh/tgasla/rl-cloudsimplus/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/e22788c9fc3c488598520c7fa35840cc" alt="Codacy Badge"></a>
<a href="https://github.com/tgasla/rl-cloudsimplus/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tgasla/rl-cloudsimplus?" alt="GPLv3 License"></a>
</div>

## Requirements

Linux Debian >=10 or MacOS >=11

### 1. Install Docker

<https://docs.docker.com/get-docker/>

 > [!WARNING]
 > If you install Docker Desktop for MacOS, make sure you are giving enough memory in your containers by going to <b> settings.. > Resources </b> and increasing the Memory Limit

### 2. Install Docker Compose

<https://docs.docker.com/compose/install/>

### 3. Install Java 21 JDK and JRE

You can install OpenJDK 21 JDK and JRE

- For Linux Debian

```bash
sudo apt install openjdk-21-jdk openjdk-21-jre
```

- For MacOS using [brew](https://brew.sh/)

```bash
brew install openjdk@21
```

<!--
or you can also try Azul Zulu

`https://www.azul.com/downloads/?version=java-17-lts#zulu`

-->

### 4. Set the JAVA_HOME environment variable to the right path

> [!IMPORTANT]  
> The exact path may vary (different distro and different arch).

- For linux

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-<arch>
```

- For MacOS

```bash
export JAVA_HOME=/usr/libexec/java_home
```

<!--
### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`
-->

<!--
- For Zulu

    `export JAVA_HOME=/Library/Java/JavaVirtualMachines/zulu-17.jdk/Contents/Home`

- For OpenJDK downloaded using brew

  You can ask brew where OpenJDK Java was installed

  `brew info openjdk@21`

  and then add the given path to your shell profile
  
  `export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`
-->

<!--
### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`
-->

## Build images

### 1. Build gateway image

Use the following command to build the gateway image:

```bash
make build-gateway
```

To enable debugging and show DEBUG log messages, use the following command:

```bash
make build-gateway-debug
```

### 2. Build manager image

To build the manager image, use the following command:

```bash
make build-manager
```

### 3. Build TensorBoard image

```bash
make build-tensorboard
```

## Start application

First start TensorBoard:

```bash
make run-tensorboard
```

Use the following command:

```bash
docker compose [--profile cuda] up [--build] [-d | --detach]
```

- The `--profile cuda` flag enables Nvidia CUDA GPU access for the manager.
  - You need to have [CUDA](https://developer.nvidia.com/cuda-downloads) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your system.
  - Make sure to restart docker daemon if you just downloaded the cuda-container-toolkit.
- The `--build` flag also builds the manager image
- The `-d` flag runs the app in detached mode (runs in the background)

If, after running the app, you want to start a second manager (to run a second experiment simultaneously), you need to run:

```bash
docker compose run [--build] [-d | --detach] manager
```

## Stop application

To stop the application, use the following command:

```bash
docker compose down
```

If you also want to clear docker unused data, use the following command:

```bash
docker system prune [-f | --force]
```

## Acknowledgements

This project uses the [CloudSim Plus](http://cloudsimplus.org/) framework: a full-featured, highly extensible, easy-to-use Java 17+ framework for modeling and simulating cloud computing infrastructure and services. The source code is available [here](https://github.com/manoelcampos/cloudsim-plus).

The code was based on the work done by [pkoperek](https://github.com/pkoperek) in the following projects:

- [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
- [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
- [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
