# Grand Challenge algorithm image for ToothFairy4 (ODIN 2026 Task 1).
#
# Base image matches the organizer's template so the CUDA runtime is known to
# work on the evaluation hardware. The build installs no packages from the
# network at run time and the container never opens a socket: Grand Challenge
# executes algorithms without internet access.
FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS toothfairy4_algorithm_amd64

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY --chown=user:user submission/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# Package source, installed without dependencies because the base image and the
# requirements file already pin everything.
COPY --chown=user:user pyproject.toml README.md LICENSE /opt/app/
COPY --chown=user:user src /opt/app/src
RUN python -m pip install --no-cache-dir --no-deps /opt/app

# The trained bundle. Grand Challenge also mounts a model tarball at this path;
# baking it in keeps the image self-contained either way.
COPY --chown=user:user submission/model /opt/ml/model

COPY --chown=user:user submission/inference.py /opt/app/inference.py

USER user

ENTRYPOINT ["python", "inference.py"]
