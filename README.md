## Running

### 1. Build gateway image
```
cd cloudsimplus_gateway
sudo make build
```

### 2. Build manager image
```
cd rl_manager
sudo make build
```
### 3. Start application
`docker compose up`

* You can also build manager image when starting application using:
`docker compose up --build`

* If you want the manager image to have NVIDIA GPU access use:
  `docker compose up --profile gpu

## Acknowledgements

Project uses the CloudSim Plus framework: a full-featured, highly extensible and easy to use Java 8+ framework for
modeling and simulation of cloud computing infrastructure and services.

You can find it [here](http://cloudsimplus.org/). Source code is available [here](https://github.com/manoelcampos/cloudsim-plus)

Code was based on the work done by [pkoperek](https://github.com/pkoperek) in these projects:
* [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
* [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
* [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
