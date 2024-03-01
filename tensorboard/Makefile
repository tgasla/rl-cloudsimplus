build:
	docker build -t tensorboard .

run:
	docker run --rm --name tensorboard -it -v ~/tb_logs/:/tb/tb-logs/ -p 80:6006 tensorboard
