<div align="center">
<a href="https://app.codacy.com/gh/tgasla/rl-cloudsimplus/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/e22788c9fc3c488598520c7fa35840cc" alt="Codacy Badge"></a>
<a href="https://github.com/tgasla/rl-cloudsimplus/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tgasla/rl-cloudsimplus?" alt="GPLv3 License"></a>
</div>

## Requirements

<details><summary><b>Linux Debian 12/11/10</b></summary>
    
### 1. Install Docker
https://docs.docker.com/get-docker/

### 2. Install Docker Compose
https://docs.docker.com/compose/install/

### 3. Install Java 21

You can install OpenJDK 21 JDK and JRE

```
sudo apt-get install openjdk-21-jdk openjdk-21-jre
```

### 4. Set the JAVA_HOME environment variable to the right path
```
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
```

<!--
### 1.5 Select the correct Gradle version

Head to the `cloudsimplus_gateway` that contains the `gradlew` file and run wrapper

`cloudsimplus_gateway/gradlew wrapper --gradle-version 7.3 --distribution-type all`
-->
</details>

<details><summary><b>MacOS 14/13/12/11</b></summary>

### 1. Install Docker
https://docs.docker.com/get-docker/

 > :warning: **Warning:**
 > If you install Docker Desktop make sure you are giving enough memory in your containers by going to <b> Settings.. > Resources </b> and increasing the Memory Limit

### 2. Install Docker Compose
https://docs.docker.com/compose/install/

### 3. Install Java 21 JDK and JRE
You can install OpenJDK Java 21 using [brew](https://brew.sh/)
```
brew install openjdk@21
```

<!--
or you can also try Azul Zulu

`https://www.azul.com/downloads/?version=java-17-lts#zulu`

-->

### 4. Make sure that the environment variable JAVA_HOME is set to the right path

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

## Build images

### 1. Build gateway image
Use the folloiwng command to build the gateway image:
```
make build-gateway
```

To enable debugging and show DEBUG log messages use the following command:
```
make build-gateway-debug
```

### 2. Build manager image
To build the manager image use the fowllowing command:
```
make build-manager
```

## Start application
Use the folloiwng command:
```
docker compose up
```

You can also build the manager image by using the folloiwng command:
```
docker compose up --build
```

If you want to see only the manager output use:
```
docker compose manager run
```

If you want the manager image to have NVIDIA CUDA GPU access you need to download the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and then use the following command:
```
docker compose --profile cuda up
```

If you want to use both the --profile and --build flags, use the following command:
```
docker compose --profile cuda up --build
```

## Stop application
To stop the application, use the following command:
```
docker compose down
```

If you also want to clear docker unused data, use the following command:
```
docker system prune -f
```

## Acknowledgements

This project uses the [CloudSim Plus](http://cloudsimplus.org/) framework: a full-featured, highly extensible, and easy to use Java 17+ framework for modeling and simulation of cloud computing infrastructure and services. The source code is available [here](https://github.com/manoelcampos/cloudsim-plus).

The code was based on the work done by [pkoperek](https://github.com/pkoperek) in these following projects:
  - [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
  - [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
  - [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
