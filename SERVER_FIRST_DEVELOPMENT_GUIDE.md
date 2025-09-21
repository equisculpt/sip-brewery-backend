# 🚀 SIP Brewery - Server-First Development Guide (Hetzner Cloud)

## 🎯 Philosophy: Develop on Server, Not Locally

**Why Server-First Development?**
- ✅ **24/7 Availability**: Server stays online when laptop is off
- ✅ **Production Environment**: Develop in the same environment as production
- ✅ **Resource Efficiency**: Use server's CPU/RAM instead of laptop battery
- ✅ **Team Collaboration**: Everyone works on the same environment
- ✅ **No Local Dependencies**: No need to install heavy software locally

---

## 📋 Prerequisites (Local Machine - Minimal Setup)

### Required Software (Lightweight)
1. **Git for Windows** (includes Git Bash) - [Download](https://git-scm.com/download/win)
2. **VS Code** with Remote-SSH extension - [Download](https://code.visualstudio.com/)
3. **Hetzner Cloud CLI** (hcloud) - For server management

### Optional (for convenience)
- **Cursor** with SSH support - [Download](https://cursor.sh/)
- **PuTTY** - Alternative SSH client

---

## 🌩️ Server Setup & Management

### 1. Install Hetzner Cloud CLI (Local)
```bash
# Download and install hcloud
curl -L "https://github.com/hetznercloud/cli/releases/latest/download/hcloud-windows-amd64.zip" -o hcloud.zip
powershell.exe -Command "Expand-Archive -Path 'hcloud.zip' -DestinationPath '/c/Program Files/hcloud' -Force"
export PATH="/c/Program Files/hcloud:$PATH"
echo 'export PATH="/c/Program Files/hcloud:$PATH"' >> ~/.bashrc
rm hcloud.zip
```

### 2. Configure Hetzner Cloud
```bash
# Create context with your API token
hcloud context create sipbrewery

# Create SSH key for server access
ssh-keygen -t rsa -b 4096 -f ~/.ssh/sipbrewery_key -N ""
hcloud ssh-key create --name sipbrewery-dev-key --public-key-from-file ~/.ssh/sipbrewery_key.pub
```

### 3. Create Development Server
```bash
# Create a powerful development server
hcloud server create \
  --name sipbrewery-dev \
  --type cx31 \
  --image ubuntu-22.04 \
  --ssh-key sipbrewery-dev-key \
  --location nbg1

# Get server IP
hcloud server list
```

### 4. Configure SSH Access
```bash
# Add to SSH config for easy access
cat >> ~/.ssh/config << EOF
Host sipbrewery-dev
    HostName YOUR_SERVER_IP
    User root
    IdentityFile ~/.ssh/sipbrewery_key
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

# Test connection
ssh sipbrewery-dev
```

---

## 🛠️ Server Environment Setup

### 1. Initial Server Configuration
```bash
# SSH to your development server
ssh sipbrewery-dev

# Update system
apt update && apt upgrade -y

# Install essential tools
apt install -y curl wget git vim nano htop tree unzip

# Install Node.js (latest LTS)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
apt install -y nodejs

# Install Python and pip
apt install -y python3 python3-pip python3-venv

# Install PM2 for process management
npm install -g pm2

# Install Docker (optional, for containerization)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### 2. Setup Development Environment
```bash
# Create development directory
mkdir -p /var/www
cd /var/www

# Clone your repository
git clone git@github.com:equisculpt/sip-brewery-backend.git
cd sip-brewery-backend

# Setup Git configuration
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Install Project Dependencies
```bash
# Install Node.js dependencies
npm install

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
nano .env  # Edit with your configuration
```

---

## 💻 Remote Development Workflow

### 1. VS Code Remote Development
```bash
# Install Remote-SSH extension in VS Code
# Then connect to server:
# Ctrl+Shift+P -> "Remote-SSH: Connect to Host" -> sipbrewery-dev
```

### 2. Daily Development Routine
```bash
# Connect to server
ssh sipbrewery-dev
cd /var/www/sip-brewery-backend

# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Start development services
npm run dev
# OR
pm2 start ecosystem.config.js --env development
```

### 3. Development with PM2 (Process Manager)
```bash
# Create PM2 ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'sipbrewery-backend',
    script: 'server.js',
    instances: 1,
    autorestart: true,
    watch: true,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'development',
      PORT: 3001
    },
    env_production: {
      NODE_ENV: 'production',
      PORT: 3001
    }
  }]
};
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 logs  # View logs
pm2 monit # Monitor processes
```

---

## 🔄 Git Workflow (Server-Based)

### 1. Feature Development
```bash
# On server
cd /var/www/sip-brewery-backend

# Start new feature
git checkout main
git pull origin main
git checkout -b feature/new-feature

# Make changes, test on server
# ... development work ...

# Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Create PR on GitHub web interface
```

### 2. Useful Git Aliases (Server)
```bash
# Add to ~/.bashrc on server
cat >> ~/.bashrc << EOF
# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gb='git branch'
alias gco='git checkout'

# Project aliases
alias sip='cd /var/www/sip-brewery-backend'
alias logs='pm2 logs'
alias restart='pm2 restart all'
alias status='pm2 status'
EOF

source ~/.bashrc
```

---

## 🚀 Production Deployment

### 1. Create Production Server
```bash
# Create production server (more powerful)
hcloud server create \
  --name sipbrewery-prod \
  --type cx41 \
  --image ubuntu-22.04 \
  --ssh-key sipbrewery-dev-key \
  --location nbg1

# Setup production environment
ssh sipbrewery-prod
# ... repeat server setup steps ...
```

### 2. Automated Deployment
```bash
# Create deployment script
cat > deploy.sh << EOF
#!/bin/bash
cd /var/www/sip-brewery-backend
git pull origin main
npm install
source venv/bin/activate
pip install -r requirements.txt
pm2 restart all
echo "Deployment completed!"
EOF

chmod +x deploy.sh

# Deploy
./deploy.sh
```

---

## 🔧 Server Management Commands

### Essential Server Commands
```bash
# Server status
hcloud server list
hcloud server describe sipbrewery-dev

# Server actions
hcloud server start sipbrewery-dev
hcloud server stop sipbrewery-dev
hcloud server reboot sipbrewery-dev

# Resize server (upgrade/downgrade)
hcloud server-type list
hcloud server change-type sipbrewery-dev --type cx41

# Create snapshot
hcloud server create-image sipbrewery-dev --description "Dev environment backup"
```

### Process Management
```bash
# PM2 commands
pm2 list          # List all processes
pm2 restart all   # Restart all processes
pm2 stop all      # Stop all processes
pm2 delete all    # Delete all processes
pm2 logs          # View logs
pm2 monit         # Monitor dashboard

# System monitoring
htop              # System resources
df -h             # Disk usage
free -h           # Memory usage
```

---

## 🐛 Troubleshooting

### Common Issues

**1. SSH Connection Issues**
```bash
# Test connection
ssh -v sipbrewery-dev

# Fix permissions
chmod 600 ~/.ssh/sipbrewery_key
chmod 644 ~/.ssh/sipbrewery_key.pub
```

**2. Server Out of Memory**
```bash
# Check memory usage
free -h
pm2 monit

# Restart services
pm2 restart all

# Upgrade server if needed
hcloud server change-type sipbrewery-dev --type cx41
```

**3. Port Already in Use**
```bash
# Find process using port
lsof -i :3001
kill -9 PID

# Or restart PM2
pm2 restart all
```

---

## 💰 Cost Optimization

### Server Sizing Guide
- **Development**: cx31 (2 vCPU, 8GB RAM) - ~€8.93/month
- **Production**: cx41 (4 vCPU, 16GB RAM) - ~€17.86/month
- **High Traffic**: cx51 (8 vCPU, 32GB RAM) - ~€35.71/month

### Cost-Saving Tips
```bash
# Stop development server when not in use
hcloud server stop sipbrewery-dev

# Start when needed
hcloud server start sipbrewery-dev

# Create snapshots before major changes
hcloud server create-image sipbrewery-dev --description "Before update"
```

---

## 🎯 Advantages of Server-First Development

### ✅ Benefits
1. **Always Online**: Your application runs 24/7
2. **Real Environment**: Develop in production-like conditions
3. **Team Collaboration**: Shared development environment
4. **Resource Efficiency**: Use server resources, not laptop battery
5. **Easy Scaling**: Upgrade server specs as needed
6. **Backup & Recovery**: Server snapshots for quick recovery

### 🔄 Workflow Summary
1. **Local**: Only Git, SSH, and code editor
2. **Server**: All development, testing, and running
3. **GitHub**: Code repository and collaboration
4. **Hetzner**: Infrastructure and hosting

---

## 📚 Quick Reference

### Daily Commands
```bash
# Connect to server
ssh sipbrewery-dev

# Navigate to project
cd /var/www/sip-brewery-backend

# Development workflow
git pull origin main
git checkout -b feature/new-feature
# ... make changes ...
git add .
git commit -m "feat: description"
git push origin feature/new-feature

# Process management
pm2 restart all
pm2 logs
pm2 monit
```

### Server Management
```bash
# Server status
hcloud server list

# Start/stop server
hcloud server start sipbrewery-dev
hcloud server stop sipbrewery-dev

# SSH to server
ssh sipbrewery-dev
```

---

**This approach eliminates the need for local dependencies while providing a robust, scalable development environment that mirrors production!** 🚀
