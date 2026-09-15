# NVIDIA Isaac Sim 6.0.1, pinned to the source environment's image digest.
ARG ISAAC_BASE=nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9
FROM ${ISAAC_BASE}
USER root
COPY requirements-runtime.txt /opt/isaac-compat/requirements-runtime.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /opt/isaac-compat/requirements-runtime.txt
COPY sitecustomize/sitecustomize.py /opt/isaac-compat/sitecustomize.py
COPY entrypoint.sh /opt/isaac-compat/entrypoint.sh
RUN chmod 0755 /opt/isaac-compat/entrypoint.sh \
    && mkdir -p /workspace/PhyRC_Sim /workspace/RCareWorld-2.0 \
    && chown -R isaac-sim:isaac-sim /workspace
USER isaac-sim
WORKDIR /workspace/PhyRC_Sim
ENV PYTHONPATH=/opt/isaac-compat PYTHONDONTWRITEBYTECODE=1 \
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
ENTRYPOINT ["/opt/isaac-compat/entrypoint.sh"]
