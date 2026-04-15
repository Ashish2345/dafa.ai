# Deployment — Azure VPS (Ubuntu 22.04)

Step-by-step guide to deploy `dafa.ai` (FastAPI backend) to an Azure VPS alongside the already-deployed frontend at `/var/www/merodafa/dafa-ai-frontend`.

The flow:
1. **One-time server setup** — SSH user, firewall, Python/uv, MongoDB, Nginx, TLS, systemd.
2. **First deploy** — clone repo, write `.env`, start service, configure Nginx.
3. **GitHub Actions** — wire up the automated redeploy on push to `main`.
4. **Subsequent deploys** — just `git push origin main`.

---

## 0. Assumptions

Adjust if different:

| Thing | Assumed value |
|---|---|
| VPS OS | Ubuntu 22.04 LTS |
| SSH user | `azureuser` (the default Azure admin) — or whatever you set |
| Backend path on VPS | `/var/www/merodafa/dafa.ai` |
| Frontend path on VPS | `/var/www/merodafa/dafa-ai-frontend` (already deployed) |
| API domain | `api.merodafa.com` (add an A record → VPS public IP) |
| Frontend domain | `app.merodafa.com` and `merodafa.com` |
| MongoDB | Self-hosted on the same VPS (listen on `127.0.0.1:27017`) |
| API internal port | `8000` (uvicorn behind Nginx) |

If MongoDB is instead MongoDB Atlas or another host, skip the Mongo install section and use the Atlas connection string in `.env`.

---

## 1. Connect to the VPS

From your laptop:

```bash
ssh azureuser@<VPS_PUBLIC_IP>
```

If this is the first connection, accept the host key.

---

## 2. Base OS packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  curl \
  git \
  ca-certificates \
  gnupg \
  unzip \
  nginx \
  ufw
```

---

## 3. Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'      # opens 80 + 443
sudo ufw enable
sudo ufw status
```

Do **not** open 8000 or 27017 to the public — Nginx fronts the API, MongoDB listens on localhost only.

---

## 4. Install `uv` (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Re-source your shell to pick up uv
source ~/.bashrc || source ~/.profile
uv --version   # sanity check
```

`uv` is installed to `~/.local/bin/uv`. The GitHub Actions workflow adds this to `PATH` explicitly.

---

## 5. Install MongoDB 7 (self-hosted, localhost only)

Skip this section if using Atlas.

```bash
# MongoDB public signing key
curl -fsSL https://pgp.mongodb.com/server-7.0.asc \
  | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
  | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

sudo apt update
sudo apt install -y mongodb-org

sudo systemctl enable --now mongod
sudo systemctl status mongod --no-pager | head -10
```

MongoDB listens on `127.0.0.1:27017` by default (see `/etc/mongod.conf`). Keep it that way.

**Create an application user** (optional but recommended for a prod):

```bash
mongosh
```

In the mongo shell:

```javascript
use dafa
db.createUser({
  user: "dafa_app",
  pwd: "<CHOOSE_A_STRONG_PASSWORD>",
  roles: [{ role: "readWrite", db: "dafa" }],
})
exit
```

Your `MONGODB_URL` will then be:
```
mongodb://dafa_app:<PASSWORD>@127.0.0.1:27017/dafa?authSource=dafa
```

---

## 6. Prepare the deploy directory

```bash
sudo mkdir -p /var/www/merodafa
sudo chown -R "$USER":"$USER" /var/www/merodafa
cd /var/www/merodafa
```

---

## 7. Clone the backend repo

You need a read-only GitHub deploy key so `git pull` works without password prompts from GitHub Actions.

**On the VPS:**

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/dafa_deploy -C "dafa-vps-deploy"
cat ~/.ssh/dafa_deploy.pub
```

Copy the printed public key.

**On GitHub** (`Ashish2345/dafa.ai` repo):
- Settings → Deploy keys → **Add deploy key**
- Title: `azure-vps`, paste the key, leave "Allow write access" **unchecked**.

**On the VPS**, configure SSH to use this key for GitHub:

```bash
cat >> ~/.ssh/config <<'EOF'

Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/dafa_deploy
  IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
ssh -T git@github.com    # expect: "Hi Ashish2345/dafa.ai! You've successfully authenticated..."
```

Clone:

```bash
cd /var/www/merodafa
git clone git@github.com:Ashish2345/dafa.ai.git
cd dafa.ai
```

---

## 8. Install Python deps

```bash
cd /var/www/merodafa/dafa.ai
uv sync --frozen --no-dev
```

This creates `.venv/` with all packages. Test it boots:

```bash
uv run python -c "from app.main import app; print('imports ok')"
```

---

## 9. Create the `.env` file

```bash
sudo install -m 600 -o "$USER" -g "$USER" /dev/null /var/www/merodafa/dafa.ai/.env
nano /var/www/merodafa/dafa.ai/.env
```

Fill in (adjust every value):

```dotenv
# --- FastAPI / App ---
APP_ENV=production
LOG_LEVEL=INFO
API_V1_PREFIX=/api/v1

# CORS — the domains your frontend is served from
CORS_ORIGINS=https://app.merodafa.com,https://merodafa.com

# --- Secrets ---
JWT_SECRET_KEY=<generate-with-`openssl rand -hex 32`>
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080

# --- MongoDB ---
MONGODB_URL=mongodb://dafa_app:<PASSWORD>@127.0.0.1:27017/dafa?authSource=dafa
MONGODB_DB=dafa

# --- LLM ---
GOOGLE_API_KEY=<your-gemini-api-key>
GEMINI_MODEL=gemini-2.5-flash

# --- Optional: OCR provider creds ---
# GOOGLE_APPLICATION_CREDENTIALS=/var/www/merodafa/dafa.ai/gcp-sa.json
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
```

Permissions matter — `chmod 600` so only the deploy user can read it:

```bash
chmod 600 /var/www/merodafa/dafa.ai/.env
```

---

## 10. Systemd service

Create `/etc/systemd/system/dafa-api.service`:

```bash
sudo tee /etc/systemd/system/dafa-api.service > /dev/null <<'EOF'
[Unit]
Description=dafa.ai FastAPI backend (uvicorn)
After=network.target mongod.service
Wants=mongod.service

[Service]
Type=simple
User=azureuser
Group=azureuser
WorkingDirectory=/var/www/merodafa/dafa.ai
Environment="PATH=/home/azureuser/.local/bin:/usr/bin"
EnvironmentFile=/var/www/merodafa/dafa.ai/.env
ExecStart=/home/azureuser/.local/bin/uv run uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 2 \
  --proxy-headers \
  --forwarded-allow-ips=127.0.0.1
Restart=on-failure
RestartSec=3
# Keep a sane log buffer
StandardOutput=journal
StandardError=journal
# Mild hardening
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now dafa-api
sudo systemctl status dafa-api --no-pager | head -20
```

Tail logs while you verify:

```bash
sudo journalctl -u dafa-api -f
```

Probe locally:

```bash
curl -i http://127.0.0.1:8000/api/v1/health || true
```

If `workers` > 1, consider raising to `4` once you confirm RAM usage is fine (each worker loads the app, OCR pipeline can eat memory).

---

## 11. Allow the CI deploy user to restart the service without a password

The GitHub Actions workflow runs `sudo systemctl restart dafa-api`. Add a narrow sudoers rule:

```bash
sudo tee /etc/sudoers.d/dafa-api-restart > /dev/null <<'EOF'
azureuser ALL=(root) NOPASSWD: /bin/systemctl restart dafa-api, /bin/systemctl status dafa-api
EOF
sudo chmod 440 /etc/sudoers.d/dafa-api-restart
# Sanity check: no syntax errors
sudo visudo -c
```

---

## 12. Nginx reverse proxy + TLS

Create `/etc/nginx/sites-available/dafa-api`:

```bash
sudo tee /etc/nginx/sites-available/dafa-api > /dev/null <<'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name api.merodafa.com;

    # Certbot ACME challenge
    location /.well-known/acme-challenge/ { root /var/www/letsencrypt; }

    # Upload size — PDFs can be large (backend bumps as needed too)
    client_max_body_size 200m;

    # Proxy everything to uvicorn
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # RAG queries can take 30-60s (LLM + OCR + synthesis)
        proxy_read_timeout 180s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 180s;

        # Let SSE / streaming endpoints flush without buffering
        proxy_buffering off;
        proxy_cache off;
    }
}
EOF

sudo mkdir -p /var/www/letsencrypt
sudo ln -sf /etc/nginx/sites-available/dafa-api /etc/nginx/sites-enabled/dafa-api
sudo nginx -t && sudo systemctl reload nginx
```

Point the DNS **A record** for `api.merodafa.com` to the VPS public IP. Wait for propagation (check with `dig api.merodafa.com`).

### TLS with Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.merodafa.com \
  --non-interactive --agree-tos -m ashish.rayamajhi@docsumo.com \
  --redirect
# Auto-renewal is installed by default — verify:
sudo systemctl list-timers | grep certbot
```

Certbot rewrites the Nginx config to add HTTPS + redirects 80→443.

Probe from your laptop:

```bash
curl -i https://api.merodafa.com/api/v1/health
```

---

## 13. Wire up GitHub Actions

You need these secrets on the `Ashish2345/dafa.ai` repo (Settings → Secrets → Actions). The frontend repo already has them; same values work for backend.

| Secret | Value |
|---|---|
| `VM_HOST` | VPS public IP or hostname |
| `VM_USER` | `azureuser` |
| `VM_PORT` | `22` |
| `VM_SSH_KEY` | Private key of a dedicated CI user (**not** the deploy key from §7; that's for GitHub→VPS reads, this is for GitHub-Actions→VPS SSH) |

### Creating the CI SSH key (one-time)

On your laptop (or any machine):

```bash
ssh-keygen -t ed25519 -N "" -f ./dafa_ci -C "dafa-github-actions"
# public key → authorized_keys on VPS
# private key → GitHub secret VM_SSH_KEY
```

On the VPS, append the public key to `~/.ssh/authorized_keys`:

```bash
cat dafa_ci.pub | ssh azureuser@<VPS_IP> 'cat >> ~/.ssh/authorized_keys'
```

Copy the **contents of `dafa_ci`** (the private key, entire file including header/footer) and paste into GitHub secret `VM_SSH_KEY`. Delete the local copy afterward.

### Verify

```bash
# From your laptop
ssh -i ./dafa_ci azureuser@<VPS_IP> 'echo ok'
```

Then go to GitHub → Actions → **Deploy backend** → **Run workflow**. Should complete within a couple of minutes.

---

## 14. Subsequent deploys

```bash
# on your laptop
git push origin main
```

GitHub Actions fires, SSHes in, `git pull`, `uv sync`, `systemctl restart dafa-api`. Done.

To deploy manually without a push:

- GitHub → Actions → **Deploy backend** → **Run workflow**
- Or on the VPS: `cd /var/www/merodafa/dafa.ai && git pull && uv sync && sudo systemctl restart dafa-api`

---

## 15. Frontend config update

Once the API is live at `https://api.merodafa.com`, the frontend needs its `VITE_API_URL` (or equivalent) env var pointing to it. Update in the frontend repo's `.env.production`, rebuild, and the frontend deploy workflow redeploys.

---

## 16. Operations cheat sheet

```bash
# Logs
sudo journalctl -u dafa-api -f                  # live
sudo journalctl -u dafa-api --since "10 min ago"
sudo journalctl -u dafa-api -n 200 --no-pager

# Service control
sudo systemctl status dafa-api
sudo systemctl restart dafa-api
sudo systemctl stop dafa-api
sudo systemctl start dafa-api

# Nginx
sudo nginx -t && sudo systemctl reload nginx
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# MongoDB
mongosh mongodb://127.0.0.1:27017
sudo systemctl status mongod

# Disk usage (PDFs + images in GridFS can grow)
df -h
du -sh /var/lib/mongodb
du -sh /var/www/merodafa/dafa.ai/uploads
```

---

## 17. Troubleshooting

**`502 Bad Gateway` from Nginx**  
→ uvicorn is down. `sudo journalctl -u dafa-api -n 100`. Usually a missing env var, bad Python import, or Mongo connection refused.

**Backend restarts but API returns old behavior**  
→ Python bytecode cache. `sudo systemctl restart dafa-api` should clear in-process state; if it doesn't, check that `git pull` actually updated `app/` (`git log -1` on the VPS).

**GitHub Actions deploy fails at `systemctl restart`**  
→ sudoers rule from §11 missing or wrong user. Check `sudo -n -u azureuser systemctl restart dafa-api`.

**CORS errors from frontend**  
→ `CORS_ORIGINS` in `.env` doesn't include the exact frontend origin (protocol + host + port match). Check `app/settings.py:40-44`.

**MongoDB auth errors**  
→ User created in `admin` vs `dafa` DB, or wrong `authSource` in URL. `mongosh` → `use dafa` → `db.getUsers()`.

**TLS fails to renew**  
→ Port 80 blocked, or ACME-challenge path doesn't exist. Test manually: `sudo certbot renew --dry-run`.

---

## 18. Things to set up post-deploy (recommended, not blocking)

- **Log rotation** for app logs if you redirect to files (default journald already rotates).
- **Backups** for MongoDB:
  ```
  mongodump --db dafa --out /var/backups/mongo/$(date +%F)
  ```
  Run via cron daily, prune old ones weekly.
- **Monitoring**: `htop`, or UptimeRobot against `https://api.merodafa.com/api/v1/health` for external uptime checks.
- **Log cost tracking**: the LLM service already logs per-call `cost=$0.xxxx`. If you want this in a dashboard, pipe journald → Loki/Grafana later.
- **Harden SSH**: disable password login (`PasswordAuthentication no` in `/etc/ssh/sshd_config`).
- **Fail2ban** for SSH brute-force protection.
