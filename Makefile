DOCKER       ?= docker
IMAGE_NAME   ?= novasplat
IMAGE_TAG    ?= cuda128
DATA_DIR     ?= $(HOME)/data
OUTPUT_DIR   ?= $(CURDIR)/output
IPC_MODE     ?= host
SHM_SIZE     ?= 8g

FULL_IMAGE   := $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: build shell

build:
	$(DOCKER) build -t $(FULL_IMAGE) .

shell:
	$(DOCKER) run --rm -it --privileged --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e XDG_RUNTIME_DIR=/tmp/runtime-dir --ipc=$(IPC_MODE) --shm-size=$(SHM_SIZE) \
		-v $(CURDIR):/workspace/novasplat \
		-v $(DATA_DIR):/root/data \
		-v $(OUTPUT_DIR):/workspace/novasplat/output \
		-w /workspace/novasplat \
		$(FULL_IMAGE) \
		bash
