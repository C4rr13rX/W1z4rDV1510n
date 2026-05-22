FROM python:3.12-slim

# Validation-only image — no candidate execution.  Python's stdlib is
# sufficient for `py_compile` so we install nothing extra.

# Drop privileges so candidate code (mounted read-only) can't be
# tampered with by the validator itself.
RUN useradd --create-home --shell /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /work
