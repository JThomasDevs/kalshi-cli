#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const CLI_DIR = path.join(os.homedir(), '.local', 'share', 'kalshi-cli');
const VENV_DIR = path.join(CLI_DIR, '.venv');
const REPO_URL = 'https://github.com/JThomasDevs/kalshi-cli.git';
const KALSHI_DIR = path.join(os.homedir(), '.kalshi');

console.log('🎰 Setting up kalshi-cli...');

// Clone or update repo
function setupRepo() {
  return new Promise((resolve) => {
    if (fs.existsSync(path.join(CLI_DIR, '.git'))) {
      console.log('📦 Updating kalshi-cli...');
      const git = spawn('git', ['pull'], { cwd: CLI_DIR, stdio: 'inherit' });
      git.on('close', resolve);
    } else {
      console.log('📦 Cloning kalshi-cli...');
      fs.mkdirSync(CLI_DIR, { recursive: true });
      const git = spawn('git', ['clone', REPO_URL, CLI_DIR], { stdio: 'inherit' });
      git.on('close', resolve);
    }
  });
}

// Create venv and install deps
function setupVenv() {
  return new Promise((resolve) => {
    const venvExists = fs.existsSync(VENV_DIR);

    if (!venvExists) {
      console.log('📦 Creating virtual environment...');
      const venv = spawn('python3', ['-m', 'venv', VENV_DIR], { stdio: 'inherit' });
      venv.on('close', (code) => {
        if (code !== 0) {
          console.error('❌ Failed to create venv');
          process.exit(code);
        }
        installDeps(resolve);
      });
    } else {
      // Always reinstall deps on update to pick up new requirements
      installDeps(resolve);
    }
  });
}

function installDeps(resolve) {
  console.log('📦 Installing Python dependencies...');
  const pip = spawn(path.join(VENV_DIR, 'bin', 'pip'), ['install', '-r', path.join(CLI_DIR, 'requirements.txt')], {
    stdio: 'inherit'
  });
  pip.on('close', resolve);
}

// Set up ~/.kalshi directory with .env template
function setupKalshiConfig() {
  console.log('📄 Setting up ~/.kalshi configuration...');
  
  fs.mkdirSync(KALSHI_DIR, { recursive: true });
  
  const envPath = path.join(KALSHI_DIR, '.env');
  if (!fs.existsSync(envPath)) {
    fs.writeFileSync(envPath, `# Kalshi API Configuration
# Get credentials at: https://kalshi.com/api
KALSHI_ACCESS_KEY=your_access_key_here
`);
    console.log('📄 Created ~/.kalshi/.env — edit it with your API key');
  } else {
    console.log('✅ ~/.kalshi/.env already exists');
  }
  
  const keyPath = path.join(KALSHI_DIR, 'private_key.pem');
  if (!fs.existsSync(keyPath)) {
    fs.writeFileSync(keyPath, `# Place your RSA private key here
# Get it from: https://kalshi.com/api
-----BEGIN RSA PRIVATE KEY-----
your_private_key_here
-----END RSA PRIVATE KEY-----
`);
    console.log('📄 Created ~/.kalshi/private_key.pem — place your RSA private key here');
  } else {
    console.log('✅ ~/.kalshi/private_key.pem already exists');
  }
}

async function main() {
  await setupRepo();
  await setupVenv();
  setupKalshiConfig();
  console.log('');
  console.log('✅ kalshi-cli is ready!');
  console.log('');
  console.log('📋 Next steps:');
  console.log('  1. Edit ~/.kalshi/.env with your KALSHI_ACCESS_KEY');
  console.log('  2. Paste your RSA private key into ~/.kalshi/private_key.pem');
  console.log('  3. Run: kalshi --help');
}

main().catch(err => {
  console.error('❌ Setup failed:', err.message);
  process.exit(1);
});
