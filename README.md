# 1. Prerequisites

<details><summary><b>Linux Debian 12/11/10</b></summary>
    
### 1.1 Install Docker
https://docs.docker.com/get-docker/

### 1.2 Install Docker Compose
https://docs.docker.com/compose/install/

### 1.3 Install Java 21

You can install OpenJDK 21 JDK and JRE

`sudo apt-get install openjdk-21-jdk openjdk-21-jre`

### 1.4 Make sure that the environment variable JAVA_HOME is set to the right path

`export JAVA_HOME=/usr/lib/jvm/java-21-openjdk`

<!--
### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`
-->
</details>

<details><summary><b>MacOS 14/13/12/11</b></summary>

### 1.1 Install Docker
https://docs.docker.com/get-docker/

If you install Docker Desktop make sure you are giving enough memory in your containers by going to <b> Settings.. > Resources </b> and increasing the Memory Limit

### 1.2 Install Docker Compose
https://docs.docker.com/compose/install/

### 1.3 Install Java 21 JDK and JRE
You can install OpenJDK Java 21 using [brew](https://brew.sh/)

`brew install openjdk@21`

<!--
or you can also try Azul Zulu

`https://www.azul.com/downloads/?version=java-17-lts#zulu`

-->

### 1.4 Make sure that the environment variable JAVA_HOME is set to the right path

`export JAVA_HOME=/usr/libexec/java_home`

<!--
- For Zulu

    `export JAVA_HOME=/Library/Java/JavaVirtualMachines/zulu-17.jdk/Contents/Home`

- For OpenJDK downloaded using brew

  You can ask brew where OpenJDK Java was installed

  `brew info openjdk@21`

  and then add the given path into your shell profile
  
  `export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`

  -->
  
<!--
### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`
-->
</details>

# 2. Build images

## 2.1 Build gateway image
`make build-gateway`

or

`make build-gateway-debug` to enable debugging and show DEBUG log messages. 

## 2.2 Build manager image
`make build-manager`

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

- This project uses the CloudSim Plus framework: a full-featured, highly extensible, and easy to use Java 17+ framework for
modeling and simulation of cloud computing infrastructure and services.

  You can find it [here](http://cloudsimplus.org/). The source code is available [here](https://github.com/manoelcampos/cloudsim-plus)

- The code was based on the work done by [pkoperek](https://github.com/pkoperek) in these following projects:
  - [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
  - [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
  - [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
