FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY submit.py /workspace/submit.py
COPY RL.py /workspace/RL.py

RUN pip install --no-cache-dir -e .
