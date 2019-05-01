IMAGE_NAME = keras-facenet
NOTEBOOK_PORT = 5001
VOLUMES = -v "$(PWD):/usr/src"
JUPYTER_OPTIONS := --ip=0.0.0.0 --port=$(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

.PHONY: build

build:
	docker build -t $(IMAGE_NAME) .
lab-server:
	docker run -it $(VOLUMES) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME) jupyter lab $(JUPYTER_OPTIONS)
bash:
	docker run -it $(VOLUMES) $(IMAGE_NAME) bash
test:
	docker run -it $(VOLUMES) $(IMAGE_NAME) pytest
clean:
	rm -rf build
	rm -rf dist
	rm -rf tests/facenet
	rm -rf *.egg-info
package:
	python setup.py sdist bdist_wheel
