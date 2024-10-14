# RL-CloudSimPlus

<div align="center">
<a href="https://app.codacy.com/gh/tgasla/rl-cloudsimplus/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/e22788c9fc3c488598520c7fa35840cc" alt="Codacy Badge"></a>
<a href="https://github.com/tgasla/rl-cloudsimplus/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tgasla/rl-cloudsimplus?" alt="GPLv3 License"></a>
</div>

## Installation


### 1. Install Docker

<https://docs.docker.com/get-docker/>

 > [!WARNING]
 > If you install Docker Desktop for MacOS, make sure you are giving enough memory in your containers by going to <b> Settings... > Resources </b> and increasing the Memory Limit

### 2. Install Docker Compose

<https://docs.docker.com/compose/install/>

### 3. Install Java OpenJDK 21

- For Debian, install the openjdk-21-jdk and openjdk-21-jre packages

```bash
sudo apt install openjdk-21-jdk openjdk-21-jre
```

- For MacOS, use [brew](https://brew.sh/)

```bash
brew install openjdk@21
```

<!--
or you can also try Azul Zulu

`https://www.azul.com/downloads/?version=java-21-lts#zulu`

-->

### 4. Set the JAVA_HOME environment variable to the right path

> [!IMPORTANT]  
> The exact path may vary (different distro, different arch, etc.)

- For Linux

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

`cloudsimplus_gateway/gradlew wrapper --gradle-version 8.6 --distribution-type all`
-->

<!--
- For Zulu

    `export JAVA_HOME=/Library/Java/JavaVirtualMachines/zulu-21.jdk/Contents/Home`

- For OpenJDK downloaded using brew

  You can ask brew where OpenJDK Java was installed

  `brew info openjdk@21`

  and then add the given path to your shell profile
  
  `export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home`
-->

<!--
### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 8.6 --distribution-type all`
-->

## Building the TensorBoard, Gateway, and RL Manager images

```bash
make build
```

> [!NOTE]  
> It is often useful to rebuild images one at a time, especially when a change is made only in a specific application part.
> For example, when we change the gateway code, we must rebuild the image before running the application.
> Optionally, when building, we can specify the log level of the messages printed in stdout and in a log file.
> If the log level environment variable is not set, the log level is set to INFO by default.

```bash
[LOG_LEVEL=TRACE|DEBUG|INFO|WARN|ERROR] make build-gateway
```

## Starting the TensorBoard dashboard

We have created three docker images. The gateway and manager images consist of the main application and are the docker compose services we need for every experiment we want to run.
The TensorBoard image is the UI endpoint and helps us keep track of the experiment's progress. Because we do not want to shut down the visualization dashboard every time we want to stop an experiment,
the TensorBoard image is not a docker compose service and can be started as a standalone docker container by using the following command (TODO: consider changing it):

```bash
make run-tensorboard
```

> [!NOTE]
> You can check that the TensorBoard dashboard is running by visiting [http://localhost](http://localhost).

## Editing the experiment configuration file

To run an experiment, first edit the configuration file located at [rl-manager/mnt/config.yml](https://github.com/tgasla/rl-cloudsimplus/blob/main/rl-manager/mnt/config.yml).

The configuration file contains two sections: the 'common' section and the 'experiment' section. This is because we may run multiple different experiments in parallel.
- The parameters that all experiments have in common are specified under the common section, and those that are unique among the experiments are defined under the experiment_{id} section
  - If a parameter is specified in both the common and experiment sections, the common one is ignored, and the experiment one takes effect.
- To run multiple experiments in parallel, add as many experiment areas as you want, specifying the corresponding parameters for each experiment.
- Each experiment should have a unique experiment id, and each section should be written as experiment_{id}. For simplicity, use ids starting by 1 and increment by 1.

There are three experiment modes: train, transfer, and test. When transfer or test mode are specified, an additional 'train_model_dir' key for an experiment should be defined, with the name of the directory in which the trained agent model should be used.

## Running an experiment

After you are ready editing the configuration file run the following command to start the experiment(s).

```bash
make run-cpu
```

> [!NOTE]
> This command runs all the docker containers in detached mode
> If you want to use attached mode try the following command:

```bash
make run-cpu-attached
```

## CUDA GPU support

There is also support to run the experiments in CUDA GPUs.
  - You need to have [CUDA](https://developer.nvidia.com/cuda-downloads) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your system.
  - Restart the docker daemon if you just downloaded the cuda-container-toolkit.

Run using all available CUDA devices by running the following command:

```bash
make run-gpu
```
for detached

or

```bash
make run-gpu-attached
```

for attached mode.

<!--
- The `--build` flag also builds the manager image
- The `-d` flag runs the app in detached mode (runs in the background)

If, after running the app, you want to start a second manager (to run a second experiment simultaneously), you need to run:

```bash
docker compose run [--build] [-d | --detach] manager
```
-->

## Stopping the application

If you want to stop the application and clear all the dangling containers and volumes run the following command:

```bash
make stop
```

<!--
If you also want to clear docker unused data, use the following command:

```bash
docker system prune [-f | --force]
```
-->

## Acknowledgements

- This project uses the [CloudSim Plus](http://cloudsimplus.org/) framework: a full-featured, highly extensible, easy-to-use Java 17+ framework for modeling and simulating cloud computing infrastructure and services. The source code is available [here](https://github.com/manoelcampos/cloudsim-plus).

- The code was based on the work done by [pkoperek](https://github.com/pkoperek) in the following projects:
  - [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
  - [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
  - [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
