## Running

### Environment variables

* `TEST_CASE` - denotes the test case which should be ran as experiment.
  Defaults to `model`. Available values: `model`, `dcnull`.

### docker-compose

Running a simple test with 
* `docker-compose run manager`

### kubernetes

* `kubectl create -f dqn.yml` - create/deploy
* `kubectl logs <POD>` - view logs of the pod
* `kubectl delete -f dqn.yml` - delete the deployment

## Acknowledgements

Project uses the CloudSim Plus framework: a full-featured, highly extensible and easy to use Java 8+ framework for
modeling and simulation of cloud computing infrastructure and services.

You can find it [here](http://cloudsimplus.org/). Source code is available [here](https://github.com/manoelcampos/cloudsim-plus)

Code was based on these projects:
* [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
* [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
* [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
