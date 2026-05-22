FROM node:20-alpine

# `node --check` only — no candidate execution.
RUN adduser -D -s /sbin/nologin sandbox
USER sandbox
WORKDIR /work
