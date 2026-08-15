.PHONY: build up

build:
	docker build -t cat-gesture-meme-tracker .

up: build
	docker run --rm -p 8000:80 cat-gesture-meme-tracker
