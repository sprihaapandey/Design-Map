#!/usr/bin/env bash
# Run this ON the Oracle Cloud VM (via SSH), as the default 'ubuntu' user.
# Idempotent — safe to re-run to pick up a new git push (it'll pull latest
# and restart the service).
#
# Usage:
#   ssh -i <your-key.pem> ubuntu@<vm-public-ip>
#   curl -fsSL https://raw.githubusercontent.com/sprihaapandey/Design-Map/main/deploy/setup.sh | bash
#   (or scp this file over and run it directly, if the repo isn't public)

set -euo pipefail

REPO_URL="https://github.com/sprihaapandey/Design-Map.git"
APP_DIR="/opt/tastemap"

echo "==> installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git nginx

echo "==> setting up $APP_DIR"
if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p "$APP_DIR"
    sudo chown "$USER:$USER" "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
else
    cd "$APP_DIR" && git pull
fi

cd "$APP_DIR"

echo "==> python venv + server-only deps (excludes scraping/labeling tools)"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-server.txt

echo "==> installing systemd service"
sudo cp deploy/tastemap.service /etc/systemd/system/tastemap.service
sudo systemctl daemon-reload
sudo systemctl enable tastemap
sudo systemctl restart tastemap

echo "==> installing nginx reverse proxy config"
sudo cp deploy/nginx-tastemap.conf /etc/nginx/sites-available/tastemap
sudo ln -sf /etc/nginx/sites-available/tastemap /etc/nginx/sites-enabled/tastemap
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo
echo "==> done. checking service status:"
sleep 2
sudo systemctl status tastemap --no-pager -l | head -15
echo
echo "visit: http://$(curl -s ifconfig.me)/"
