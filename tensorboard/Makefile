build:
	docker build -t tensorboard .

run:
	docker run --rm -it -v ~/tb_logs/:/tb/tb_logs/ -p 80:6006 tensorboard
