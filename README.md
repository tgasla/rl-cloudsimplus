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
```
brew install openjdk@21
```

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
Use the folloiwng command to build the gateway image:
```
make build-gateway
```

To enable debugging and show DEBUG log messages use the following command:
```
make build-gateway-debug
```

## 2.2 Build manager image
To build the manager image use the fowllowing command:
```
make build-manager
```

# 3. Start application
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

If you want the manager image to have NVIDIA CUDA GPU access you need to download the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) then use the following command:
```
docker compose --profile cuda up
```

If you want to use both the --profile and --build flags, the correct syntax is the following:
```
docker compose --profile cuda up --build
```

# 4. Stop application
To stop the application and clear unused data, use the following commands:
```
docker compose down && docker system prune -f
```

# Acknowledgements

This project uses the [CloudSim Plus](http://cloudsimplus.org/) framework: a full-featured, highly extensible, and easy to use Java 17+ framework for modeling and simulation of cloud computing infrastructure and services. The source code is available [here](https://github.com/manoelcampos/cloudsim-plus).

The code was based on the work done by [pkoperek](https://github.com/pkoperek) in these following projects:
  - [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
  - [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
  - [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
