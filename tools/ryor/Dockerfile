# Run-your-own-rescue engine: the headless simulator with the real brain (tools/ryor/engine) and the replay streamer.
# Built by tools/ryor/build_image.sh from a staged context (only the files below), linux/arm64 for Fargate Graviton.
FROM mcr.microsoft.com/dotnet/runtime:8.0-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-boto3 python3-httpx && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 ryor
WORKDIR /app
COPY --chown=ryor ctx/ /app/
RUN mkdir -p /app/unity/SmokeRescue/Assets && chown -R ryor /app/unity
USER ryor
ARG TAG=local
ENV ENGINE_TAG=$TAG ENGINE_GIT=$TAG DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_gcServer=0
ENTRYPOINT ["python3", "-u", "/app/tools/ryor/entry.py"]
