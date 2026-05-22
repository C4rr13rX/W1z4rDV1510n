# Sandbox Dockerfiles

One Dockerfile per language. The Docker sandbox backend builds these
lazily on first use of each language. Naming convention: `{lang}.Dockerfile`.

| File | Image tag |
|---|---|
| `python.Dockerfile` | `w1z4rd-sb-python:3.12-slim` |
| `javascript.Dockerfile` | `w1z4rd-sb-node:20-alpine` |
| `typescript.Dockerfile` | `w1z4rd-sb-ts:5-alpine` |
| `rust.Dockerfile` | `w1z4rd-sb-rust:1.80-slim` |
| `bash.Dockerfile` | `w1z4rd-sb-bash:5` |

Public images (powershell, go, java, cpp, csharp) are pulled directly,
no Dockerfile needed.

Each image must:
- Run with `--network=none --read-only --tmpfs /tmp:size=64m`
- Validate syntax/types only — never execute the candidate's main
- Exit non-zero on validation failure

Build is triggered automatically by `DockerSandbox._ensure_image`. To
pre-warm all images:

```bash
for f in *.Dockerfile; do
  lang="${f%.Dockerfile}"
  docker build -t "w1z4rd-sb-${lang}:latest" -f "$f" .
done
```
