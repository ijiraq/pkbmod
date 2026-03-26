FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    VNC_RESOLUTION=1920x1080 \
    VNC_COL_DEPTH=24 \
    NOVNC_PORT=8080 \
    VNC_PORT=5901

ARG CURSOR_APPIMAGE_URL="https://download.cursor.sh/linux/appImage/x64"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dbus-x11 \
    fluxbox \
    libasound2t64 \
    libgtk-3-0t64 \
    libnss3 \
    libx11-xcb1 \
    novnc \
    python3-websockify \
    tini \
    x11vnc \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash cursor

WORKDIR /opt/cursor
RUN curl -fsSL "$CURSOR_APPIMAGE_URL" -o /opt/cursor/Cursor.AppImage \
    && chmod +x /opt/cursor/Cursor.AppImage

COPY --chown=cursor:cursor entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER cursor
WORKDIR /home/cursor

EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
