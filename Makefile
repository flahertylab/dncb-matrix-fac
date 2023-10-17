

# Define variables
DOCKER_IMAGE = dncb-fac-image
CONTAINER_NAME = dncb-fac-container
DATA_PATH = /data/projects/dncbtd

USER = $(shell id -un)
GROUP = $(shell id -gn)
UID = $(shell id -u)
GID = $(shell id -g)

.PHONY: build run clean tests bash

# Build the Docker image
build:
	docker build -t $(DOCKER_IMAGE) .

# Run the Docker container
run:
	docker run --rm -it -v $(DATA_PATH):/work/data --name $(CONTAINER_NAME) $(DOCKER_IMAGE)

bash:
	docker run --rm -it -v $(DATA_PATH):/work/data --name $(CONTAINER_NAME) $(DOCKER_IMAGE) /bin/bash

# Clean up the Docker container
clean:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

# Save requirements.txt
requirements.txt :
	conda env export > environment.yml --no-builds
	pip list --format=freeze > requirements.txt

tests:
	docker run --rm -it --name $(CONTAINER_NAME) $(DOCKER_IMAGE) python run_tests.py