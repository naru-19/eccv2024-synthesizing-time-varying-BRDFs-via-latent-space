

.PHONY: build
build:
	docker build -t synthesize_tvbrdf:latest .

.PHONY: run
run: build
	docker run -it --rm --name synthesize_tvbrdf \
		-v $(shell pwd):/app \
		-v $(shell pwd)/model_weights:/app/model_weights \
		synthesize_tvbrdf:latest	

.PHONY: in
in:
	docker exec -it synthesize_tvbrdf /bin/bash