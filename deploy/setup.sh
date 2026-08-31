#!/usr/bin/env bash
# Run this ON the Oracle Cloud VM (via SSH), as whatever default user your
# image gives you (opc for Oracle Linux, ubuntu for Canonical Ubuntu — this
# script detects the package manager and adapts, including firewalld/SELinux
# steps that only apply to Oracle Linux).
#
# Idempotent — safe to re-run to pick up a new git push (it'll pull latest
# and restart the service).
#
# Usage:
#   ssh -i <your-key.pem> <user>@<vm-public-ip>
#   curl -fsSL https://raw.githubusercontent.com/sprihaapandey/Design-Map/main/deploy/setup.sh | bash

set -euo pipefail

REPO_URL="https://github.com/sprihaapandey/Design-Map.git"
APP_DIR="/opt/tastemap"
APP_USER="$(whoami)"

echo "==> detecting OS package manager"
if command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER=dnf
elif command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER=apt
else
    echo "unsupported OS: no dnf or apt-get found" >&2
    exit 1
fi
echo "    using $PKG_MANAGER, running as $APP_USER"

echo "==> installing system packages"
if [ "$PKG_MANAGER" = "apt" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-venv python3-pip git nginx
else
    sudo dnf install -y -q python3 python3-pip git nginx firewalld
    # python3-venv isn't a separate package on most RHEL-family distros —
    # the venv module ships in the base python3 package. Try anyway, don't
    # fail the script if the package name doesn't exist here.
    sudo dnf install -y -q python3-venv 2>/dev/null || true
fi

echo "==> setting up $APP_DIR"
if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p "$APP_DIR"
    sudo chown "$APP_USER:$APP_USER" "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
else
    cd "$APP_DIR" && git pull
fi

cd "$APP_DIR"

echo "==> python venv + server-only deps (excludes scraping/labeling tools)"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-server.txt

echo "==> installing systemd service (running as $APP_USER)"
sed "s/^User=.*/User=$APP_USER/" deploy/tastemap.service | sudo tee /etc/systemd/system/tastemap.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable tastemap
sudo systemctl restart tastemap

# If certbot has already issued a cert, nginx's config has been rewritten
# in place by the certbot --nginx plugin (adds the 443 server block + the
# 80->443 redirect). Overwriting it with our plain-HTTP checked-in version
# on every redeploy would silently undo TLS every time this script re-runs.
# Once a cert exists, leave nginx config alone — deploy/nginx-tastemap.conf
# only matters for the very first (pre-TLS) run.
CERT_GLOB=(/etc/letsencrypt/live/*/fullchain.pem)
if [ -e "${CERT_GLOB[0]}" ]; then
    echo "==> TLS cert already present (${CERT_GLOB[0]}) — leaving nginx config as certbot set it up"
    sudo nginx -t
    sudo systemctl reload nginx
else
    echo "==> installing nginx reverse proxy config (HTTP only, pre-TLS)"
    # /etc/nginx/conf.d/ is included by the stock nginx.conf on both Debian/Ubuntu
    # and RHEL-family distros, so this one path works everywhere without needing
    # the Debian-specific sites-available/sites-enabled dance.
    sudo cp deploy/nginx-tastemap.conf /etc/nginx/conf.d/tastemap.conf
    sudo rm -f /etc/nginx/sites-enabled/default   # harmless no-op if this dir doesn't exist
    sudo nginx -t
    sudo systemctl enable nginx
    sudo systemctl restart nginx
fi

if [ "$PKG_MANAGER" = "dnf" ]; then
    echo "==> Oracle Linux specifics: firewalld + SELinux"
    if systemctl is-active --quiet firewalld; then
        sudo firewall-cmd --permanent --add-service=http
        sudo firewall-cmd --permanent --add-service=https
        sudo firewall-cmd --reload
        echo "    opened ports 80 + 443 in firewalld"
    fi
    if command -v setsebool >/dev/null 2>&1; then
        sudo setsebool -P httpd_can_network_connect 1
        echo "    allowed nginx -> local proxy under SELinux"
    fi
fi

echo
echo "==> done. checking service status:"
sleep 2
sudo systemctl status tastemap --no-pager -l | head -15
echo
echo "visit: http://$(curl -s ifconfig.me)/"
echo
echo "NOTE: this only opens the OS-level firewall (firewalld/ufw)."
echo "You still need Oracle Cloud Security List ingress rules for TCP 80"
echo "and TCP 443 (0.0.0.0/0) — see the earlier setup instructions if not done."
