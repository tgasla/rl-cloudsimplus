run:
	docker compose up -d

.PHONY: clean
clean:
	docker system prune -f
	docker volume prune -f

