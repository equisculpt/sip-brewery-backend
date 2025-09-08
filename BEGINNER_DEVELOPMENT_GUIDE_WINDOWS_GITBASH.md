# 🚀 SIP Brewery - Beginner Development Guide (Windows + Git Bash)

## 📋 Prerequisites

### Required Software
1. **Git for Windows** (includes Git Bash) - [Download](https://git-scm.com/download/win)
2. **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
3. **Python** (v3.8 or higher) - [Download](https://python.org/)
4. **VS Code** or **Cursor** - [VS Code](https://code.visualstudio.com/) | [Cursor](https://cursor.sh/)
5. **Hetzner Cloud CLI** (hcloud) - We'll install this together

---

## 🛠️ Initial Setup

### 1. Verify Git Bash Installation
Open **Git Bash** (right-click in any folder → "Git Bash Here" or search "Git Bash" in Start menu)

```bash
# Check Git version
git --version

# Check if you have basic Unix tools
ls --version
curl --version
```

### 2. Configure Git (First Time Only)
```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### 3. Set Up SSH Keys for GitHub
```bash
# Generate SSH key (replace with your email)
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# When prompted, press Enter to use default location
# Set a passphrase or press Enter for no passphrase

# Start SSH agent
eval "$(ssh-agent -s)"

# Add SSH key to agent
ssh-add ~/.ssh/id_rsa

# Copy public key to clipboard (Git Bash)
cat ~/.ssh/id_rsa.pub | clip

# Or view the key to copy manually
cat ~/.ssh/id_rsa.pub
```

**Add SSH Key to GitHub:**
1. Go to GitHub.com → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste your public key and save

### 4. Test SSH Connection
```bash
ssh -T git@github.com
```

---

## 📁 Project Setup

### 1. Clone the Repository
```bash
# Navigate to your development folder
cd /c/Users/MILINRAIJADA/

# Clone the repository
git clone git@github.com:yourusername/sip-brewery-backend.git

# Navigate to project directory
cd sip-brewery-backend
```

### 2. Install Dependencies
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (if requirements.txt exists)
pip install -r requirements.txt

# Or if using virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Git Bash syntax
pip install -r requirements.txt
```

---

## 🌩️ Hetzner Cloud Setup

### 1. Install Hetzner Cloud CLI
```bash
# Create directory for hcloud
mkdir -p /c/Program\ Files/hcloud

# Download hcloud CLI
curl -L "https://github.com/hetznercloud/cli/releases/latest/download/hcloud-windows-amd64.zip" -o hcloud.zip

# Extract (using PowerShell from Git Bash)
powershell.exe -Command "Expand-Archive -Path 'hcloud.zip' -DestinationPath '/c/Program Files/hcloud' -Force"

# Add to PATH for current session
export PATH="/c/Program Files/hcloud:$PATH"

# Make permanent by adding to ~/.bashrc
echo 'export PATH="/c/Program Files/hcloud:$PATH"' >> ~/.bashrc

# Clean up
rm hcloud.zip

# Verify installation
hcloud version
```

### 2. Configure Hetzner Cloud
```bash
# Create context (will prompt for API token)
hcloud context create sipbrewery

# Test connection
hcloud server list
```

**Get API Token:**
1. Go to [Hetzner Cloud Console](https://console.hetzner-cloud.com/)
2. Select your project
3. Go to Security → API Tokens
4. Create new token with Read & Write permissions

### 3. Create SSH Key for Servers
```bash
# Generate SSH key for servers (separate from GitHub)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/sipbrewery_key -N ""

# Add SSH key to Hetzner Cloud
hcloud ssh-key create --name sipbrewery-key --public-key-from-file ~/.ssh/sipbrewery_key.pub

# List SSH keys to verify
hcloud ssh-key list
```

---

## 🏗️ Development Workflow

### 1. Daily Development Routine
```bash
# Start your day
cd /c/Users/MILINRAIJADA/sip-brewery-backend

# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Start development server
npm start
# OR
npm run dev
```

### 2. Working with Git
```bash
# Check status
git status

# Add files
git add .
# OR add specific files
git add filename.js

# Commit changes
git commit -m "feat: add new feature description"

# Push to remote
git push origin feature/your-feature-name

# Switch branches
git checkout main
git checkout feature/other-branch

# Merge branch (after PR approval)
git checkout main
git pull origin main
git merge feature/your-feature-name
```

### 3. Common Git Bash Commands
```bash
# Navigation
pwd                    # Show current directory
ls -la                 # List files with details
cd /c/Users/MILINRAIJADA  # Navigate to directory
cd ..                  # Go up one directory
cd ~                   # Go to home directory

# File operations
mkdir folder-name      # Create directory
touch filename.js      # Create empty file
rm filename.js         # Delete file
rm -rf folder-name     # Delete folder and contents
cp source dest         # Copy file
mv old-name new-name   # Rename/move file

# Text operations
cat filename.js        # View file contents
head filename.js       # View first 10 lines
tail filename.js       # View last 10 lines
grep "search" file.js  # Search in file

# Process management
ps aux                 # List running processes
kill PID               # Kill process by ID
Ctrl+C                 # Stop current process
```

---

## 🔧 Development Environment

### 1. VS Code/Cursor Extensions
Install these extensions for better development experience:
- **Git Graph** - Visualize git history
- **GitLens** - Enhanced git capabilities
- **Thunder Client** - API testing
- **Prettier** - Code formatting
- **ESLint** - JavaScript linting
- **Python** - Python development
- **Remote - SSH** - Connect to servers

### 2. Useful Git Bash Aliases
Add these to your `~/.bashrc` file:
```bash
# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gb='git branch'
alias gco='git checkout'

# Navigation aliases
alias ll='ls -la'
alias la='ls -la'
alias ..='cd ..'
alias ...='cd ../..'

# Project aliases
alias sip='cd /c/Users/MILINRAIJADA/sip-brewery-backend'
alias start='npm start'
alias dev='npm run dev'

# Reload bashrc
alias reload='source ~/.bashrc'
```

Apply aliases:
```bash
source ~/.bashrc
```

### 3. Environment Variables
Create `.env` file in project root:
```bash
# Copy example environment file
cp .env.example .env

# Edit with your values
nano .env
# OR
code .env
```

---

## 🚀 Server Deployment

### 1. Create Server
```bash
# List available server types
hcloud server-type list

# List available images
hcloud image list --type system

# Create server
hcloud server create \
  --name sipbrewery-server \
  --type cx11 \
  --image ubuntu-22.04 \
  --ssh-key sipbrewery-key \
  --location nbg1

# Get server IP
hcloud server list
```

### 2. Connect to Server
```bash
# SSH to server (replace with actual IP)
ssh -i ~/.ssh/sipbrewery_key root@YOUR_SERVER_IP

# Or add to SSH config for easier access
cat >> ~/.ssh/config << EOF
Host sipbrewery
    HostName YOUR_SERVER_IP
    User root
    IdentityFile ~/.ssh/sipbrewery_key
EOF

# Then connect simply with:
ssh sipbrewery
```

### 3. Deploy Application
```bash
# On your local machine - push code
git push origin main

# On server - pull and deploy
ssh sipbrewery
cd /var/www/sip-brewery-backend
git pull origin main
npm install
npm run build
pm2 restart all
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. Permission Denied (SSH)**
```bash
# Fix SSH key permissions
chmod 600 ~/.ssh/sipbrewery_key
chmod 644 ~/.ssh/sipbrewery_key.pub
```

**2. Git Push Rejected**
```bash
# Pull latest changes first
git pull origin main
# Resolve conflicts if any, then push
git push origin main
```

**3. Node Modules Issues**
```bash
# Clear npm cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**4. Python Virtual Environment**
```bash
# Activate virtual environment
source venv/Scripts/activate

# Deactivate
deactivate

# Recreate if corrupted
rm -rf venv
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

**5. Hetzner CLI Issues**
```bash
# Reset hcloud configuration
rm -rf ~/.config/hcloud
hcloud context create sipbrewery

# Check current context
hcloud context list
hcloud context use sipbrewery
```

---

## 📚 Learning Resources

### Git & GitHub
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

### Node.js & JavaScript
- [Node.js Documentation](https://nodejs.org/en/docs/)
- [JavaScript.info](https://javascript.info/)
- [MDN Web Docs](https://developer.mozilla.org/)

### Python
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)

### Hetzner Cloud
- [Hetzner Cloud Documentation](https://docs.hetzner.cloud/)
- [CLI Documentation](https://github.com/hetznercloud/cli)

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Project setup
git clone <repo-url>
cd project-directory
npm install
cp .env.example .env

# Daily workflow
git pull origin main
git checkout -b feature/new-feature
# ... make changes ...
git add .
git commit -m "feat: description"
git push origin feature/new-feature

# Server management
hcloud server list
hcloud server create --name server --type cx11 --image ubuntu-22.04
ssh -i ~/.ssh/key root@server-ip

# Debugging
git status
git log --oneline
npm run test
tail -f logs/app.log
```

### File Paths in Git Bash
- Windows drives: `/c/`, `/d/`, etc.
- User directory: `/c/Users/MILINRAIJADA/`
- Project directory: `/c/Users/MILINRAIJADA/sip-brewery-backend/`

---

## 🆘 Getting Help

1. **Check documentation** in the project README
2. **Use Git Bash help**: `command --help`
3. **Check logs**: `tail -f logs/error.log`
4. **Ask team members** or create GitHub issues
5. **Stack Overflow** for specific technical problems

---

**Happy Coding! 🎉**

Remember: Git Bash gives you a Unix-like environment on Windows, making development more consistent across platforms. Use it for all your command-line operations for the best experience.
