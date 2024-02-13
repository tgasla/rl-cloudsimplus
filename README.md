# 1. Prerequisites

## <details><summary> Linux Debian 12/11/10 </summary>
### 1.1 Install Docker
https://docs.docker.com/get-docker/

### 1.2 Install Docker Compose
https://docs.docker.com/compose/install/

### 1.3 Install Java 17 or Java 21

You can install OpenJDK Java 17 or 21 JDK and JRE

`sudo apt-get install openjdk-17-jdk openjdk-17-jre`

For Java 21, replace 17 with 21 in the baove command.

### 1.4 Make sure that the environment variable JAVA_HOME is set to the right path

`export JAVA_HOME=/usr/lib/jvm/java-17-openjdk`

### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`

</details>

## MacOS 14/13/12/11
### 1.1 Install Docker
https://docs.docker.com/get-docker/

### 1.2 Install Docker Compose
https://docs.docker.com/compose/install/

### 1.3 Install Java 17
You can install OpenJDK Java 17 using brew

`brew install openjdk@17`

or you can also try Azul Zulu

`https://www.azul.com/downloads/?version=java-17-lts#zulu`

### 1.4 Make sure that the environment variable JAVA_HOME is set to the right path
- For Zulu

    `export JAVA_HOME=/Library/Java/JavaVirtualMachines/zulu-17.jdk/Contents/Home`

- For OpenJDK downloaded using brew

  You can ask brew where OpenJDK Java was installed

  `brew info openjdk@17`

  and then add the given path into your shell profile
  
  `export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`

### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`

# 2. Build images

## 2.1 Build gateway image
Make sure you are inside the `cloudsimplus_gateway` folder and then build the image using

`sudo make build`

## 2.2 Build manager image
Make sure you are inside the `rl_manager` folder and then build the image using

`sudo make build`

# 3. Start application
`docker compose up`

- You can also build manager image by adding the `--build` flag to the `docker compose` command

- If you want to see only the manager output use:
  `docker compose manager run`

- If you want the manager image to have GPU access:
  - For NVIDIA CUDA GPU you need to download the `nvidia-container-toolkit` by running `sudo apt-get install nvidia-container-toolkit` if you are using Debian and then add the `--profile gpu` flag to the `docker compose` command.

# 4. Stop application
`docker compose down`

# Acknowledgements

* This project uses the CloudSim Plus framework: a full-featured, highly extensible, and easy to use Java 17+ framework for
modeling and simulation of cloud computing infrastructure and services.

  You can find it [here](http://cloudsimplus.org/). The source code is available [here](https://github.com/manoelcampos/cloudsim-plus)

- The code was based on the work done by [pkoperek](https://github.com/pkoperek) in these following projects:
  - [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
  - [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
  - [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
