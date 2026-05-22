FROM rust:1.80-slim

# Scratch crate baked into the image so `cargo check` finds a Cargo.toml.
# The candidate is mounted at /work/src/lib.rs by the sandbox harness.
WORKDIR /work
RUN cargo init --lib --quiet
# Pre-fetch a small std-only dep tree so first check isn't a giant net hit.
# Candidate code replaces src/lib.rs at validation time.
RUN cargo check --quiet || true
USER nobody
