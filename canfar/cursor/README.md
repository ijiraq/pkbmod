# Cursor on CANFAR (containerized)

This directory provides a container image that runs Cursor in a virtual X11 desktop and publishes it through noVNC (web browser access).

## What this gives you

- A self-contained Cursor runtime suitable for CANFAR custom images.
- Browser access to Cursor at `http://<pod-or-service>:8080/vnc.html`.
- Similar operational model to the existing VS Code containerized workflow on CANFAR.

## Build and push

```bash
docker build -t <registry>/pkbmod/cursor:latest canfar/cursor

docker push <registry>/pkbmod/cursor:latest
```

If the default Cursor download endpoint changes, override it while building:

```bash
docker build \
  --build-arg CURSOR_APPIMAGE_URL="https://<new-cursor-url>" \
  -t <registry>/pkbmod/cursor:latest \
  canfar/cursor
```

## Run locally (sanity check)

```bash
docker run --rm -p 8080:8080 <registry>/pkbmod/cursor:latest
```

Open:

- `http://localhost:8080/vnc.html`

## Deploy on CANFAR

1. Publish the image to a registry reachable from CANFAR.
2. In the CANFAR workload form (same pattern as your VS Code app):
   - Image: `<registry>/pkbmod/cursor:latest`
   - Container port: `8080`
   - Command/args: leave default (entrypoint handles startup)
   - Storage: mount your project volume(s) (e.g., to `/home/cursor/workspace`)
3. Expose the service/route and open `/vnc.html`.

## Notes and caveats

- Cursor licensing/sign-in still applies.
- This image launches Cursor with `--no-sandbox` because containers commonly run without the Chromium sandbox setup required by Electron.
- GPU is disabled by default (`--disable-gpu`) for broad compatibility in shared/cloud environments.
- If Cursor fails to start, inspect logs from:
  - `/tmp/cursor.log`
  - `/tmp/x11vnc.log`
  - `/tmp/websockify.log`

## Optional hardening

- Add authentication in front of noVNC (Ingress auth, OAuth proxy, or network policy).
- Pin `CURSOR_APPIMAGE_URL` to a known-good, versioned artifact URL in CI.
