FROM node:20-alpine

# Strict tsc --noEmit only.  Pin to TS 5.x.
RUN npm install -g typescript@5
RUN adduser -D -s /sbin/nologin sandbox
USER sandbox
WORKDIR /work
