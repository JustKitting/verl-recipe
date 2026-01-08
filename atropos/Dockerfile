# Atropos + VeRL Training
#
# Build: docker build -t atropos-verl .
# Run:   docker run --gpus all --shm-size=32g --pid=host atropos-verl --steps 100

ARG BASE_IMAGE=verl:blackwell
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential cmake ninja-build numactl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
COPY requirements.txt /workspace/atropos/requirements.txt
RUN pip install --no-cache-dir -r /workspace/atropos/requirements.txt

COPY . /workspace/atropos/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV NCCL_CUMEM_ENABLE=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV RAY_memory_usage_threshold=0.99

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--help"]
