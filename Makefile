.PHONY: build up down

PORT ?= 8000

build:
	docker build -t cat-gesture-meme-tracker .

up: build
	docker run --rm --name cat-gesture-meme-tracker -p $(PORT):80 cat-gesture-meme-tracker

down:
	docker stop cat-gesture-meme-tracker 2>/dev/null || true
