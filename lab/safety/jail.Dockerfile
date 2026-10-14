# The jail image: an empty Ubuntu with what agent code needs to build and run. The engine's venv, the interpreter,
# weights and CUDA are bind-mounted read-only from the host; the NVIDIA runtime adds the driver.
FROM ubuntu:24.04
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      build-essential python3 python3-dev git socat ca-certificates >/dev/null \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 755 /root
