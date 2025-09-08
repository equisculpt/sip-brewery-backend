"""
Installation and Setup Script for ASI Finance Search Engine
Installs all dependencies and sets up the complete system
"""
import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASIFinanceEngineInstaller:
    """Installer for ASI Finance Search Engine"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.requirements_installed = False
        
    def run_command(self, command, description):
        """Run a command and handle errors"""
        logger.info(f"🔄 {description}")
        logger.info(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ {description} - SUCCESS")
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def install_python_packages(self):
        """Install all required Python packages"""
        logger.info("📦 Installing Python packages...")
        
        # Core packages
        core_packages = [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "pydantic>=2.0.0",
            "aiohttp>=3.8.0",
            "aiofiles>=23.0.0",
            "python-multipart>=0.0.6"
        ]
        
        # Data processing packages
        data_packages = [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "beautifulsoup4>=4.12.0",
            "lxml>=4.9.0"
        ]
        
        # ML/AI packages
        ml_packages = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "spacy>=3.6.0"
        ]
        
        # Database and caching
        db_packages = [
            "redis>=4.5.0",
            "aioredis>=2.0.0",
            "elasticsearch>=8.8.0"
        ]
        
        # Web scraping and automation
        scraping_packages = [
            "playwright>=1.35.0",
            "selenium>=4.10.0",
            "scrapy>=2.9.0"
        ]
        
        # Testing packages
        test_packages = [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0"
        ]
        
        # Additional utility packages
        utility_packages = [
            "python-dotenv>=1.0.0",
            "schedule>=1.2.0",
            "websockets>=11.0.0",
            "joblib>=1.3.0",
            "python-dateutil>=2.8.0"
        ]
        
        all_packages = (
            core_packages + data_packages + ml_packages + 
            db_packages + scraping_packages + test_packages + utility_packages
        )
        
        # Install packages in batches to avoid conflicts
        batches = [
            ("Core packages", core_packages),
            ("Data processing", data_packages),
            ("Database packages", db_packages),
            ("Utility packages", utility_packages),
            ("Testing packages", test_packages),
            ("Web scraping", scraping_packages),
            ("ML/AI packages", ml_packages)  # Install ML packages last
        ]
        
        for batch_name, packages in batches:
            logger.info(f"Installing {batch_name}...")
            package_list = " ".join(packages)
            success = self.run_command(
                f"pip install {package_list}",
                f"Installing {batch_name}"
            )
            if not success:
                logger.warning(f"Some packages in {batch_name} failed to install. Continuing...")
        
        self.requirements_installed = True
        return True
    
    def install_spacy_model(self):
        """Install spaCy English model"""
        logger.info("🧠 Installing spaCy English model...")
        return self.run_command(
            "python -m spacy download en_core_web_sm",
            "Installing spaCy English model"
        )
    
    def install_playwright_browsers(self):
        """Install Playwright browsers"""
        logger.info("🌐 Installing Playwright browsers...")
        return self.run_command(
            "playwright install chromium",
            "Installing Playwright Chromium browser"
        )
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directories...")
        
        directories = [
            "logs",
            "data",
            "models",
            "cache",
            "temp"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        return True
    
    def create_config_files(self):
        """Create configuration files"""
        logger.info("⚙️ Creating configuration files...")
        
        # Environment configuration
        env_config = """# ASI Finance Search Engine Configuration
# Database Configuration
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Crawling Configuration
MAX_CONCURRENT_REQUESTS=10
REQUEST_DELAY=1.0
USER_AGENT_ROTATION=True

# ML Configuration
USE_GPU=False
MODEL_CACHE_DIR=./models
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/asi_finance_engine.log

# Security Configuration
API_KEY_REQUIRED=False
CORS_ORIGINS=*
"""
        
        env_file = self.base_dir / ".env"
        with open(env_file, "w") as f:
            f.write(env_config)
        logger.info(f"Created environment config: {env_file}")
        
        # Logging configuration
        logging_config = """version: 1
disable_existing_loggers: False

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: ./logs/asi_finance_engine.log
    mode: a

loggers:
  asi:
    level: DEBUG
    handlers: [console, file]
    propagate: False

root:
  level: INFO
  handlers: [console]
"""
        
        logging_file = self.base_dir / "logging_config.yaml"
        with open(logging_file, "w") as f:
            f.write(logging_config)
        logger.info(f"Created logging config: {logging_file}")
        
        return True
    
    def create_startup_script(self):
        """Create startup script"""
        logger.info("🚀 Creating startup script...")
        
        startup_script = """#!/usr/bin/env python3
\"\"\"
Startup script for ASI Finance Search Engine
\"\"\"
import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from asi.enhanced_finance_engine_api import app
import uvicorn

async def main():
    \"\"\"Main startup function\"\"\"
    logging.info("🚀 Starting ASI Finance Search Engine...")
    
    # Start the server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
"""
        
        startup_file = self.base_dir / "start_asi_engine.py"
        with open(startup_file, "w") as f:
            f.write(startup_script)
        logger.info(f"Created startup script: {startup_file}")
        
        # Make it executable on Unix systems
        if os.name != 'nt':
            os.chmod(startup_file, 0o755)
        
        return True
    
    def create_test_script(self):
        """Create test runner script"""
        logger.info("🧪 Creating test script...")
        
        test_script = """#!/usr/bin/env python3
\"\"\"
Test runner for ASI Finance Search Engine
\"\"\"
import subprocess
import sys
from pathlib import Path

def run_tests():
    \"\"\"Run all tests\"\"\"
    test_files = [
        "test_complete_system.py",
        "test_symbol_resolution.py"
    ]
    
    base_dir = Path(__file__).parent
    
    for test_file in test_files:
        test_path = base_dir / test_file
        if test_path.exists():
            print(f"\\n🧪 Running {test_file}...")
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_path), "-v"
            ], cwd=base_dir)
            
            if result.returncode != 0:
                print(f"❌ Tests failed in {test_file}")
            else:
                print(f"✅ Tests passed in {test_file}")
        else:
            print(f"⚠️ Test file not found: {test_file}")

if __name__ == "__main__":
    run_tests()
"""
        
        test_file = self.base_dir / "run_tests.py"
        with open(test_file, "w") as f:
            f.write(test_script)
        logger.info(f"Created test script: {test_file}")
        
        return True
    
    def verify_installation(self):
        """Verify that installation was successful"""
        logger.info("🔍 Verifying installation...")
        
        # Test imports
        test_imports = [
            "fastapi",
            "uvicorn", 
            "pandas",
            "numpy",
            "sklearn",
            "aiohttp",
            "beautifulsoup4"
        ]
        
        failed_imports = []
        for package in test_imports:
            try:
                __import__(package)
                logger.info(f"✅ {package} - OK")
            except ImportError:
                logger.error(f"❌ {package} - FAILED")
                failed_imports.append(package)
        
        # Test optional imports
        optional_imports = [
            "torch",
            "transformers", 
            "spacy",
            "redis",
            "elasticsearch"
        ]
        
        for package in optional_imports:
            try:
                __import__(package)
                logger.info(f"✅ {package} (optional) - OK")
            except ImportError:
                logger.warning(f"⚠️ {package} (optional) - Not available")
        
        if failed_imports:
            logger.error(f"❌ Installation incomplete. Failed imports: {failed_imports}")
            return False
        
        logger.info("✅ Installation verification successful!")
        return True
    
    def run_installation(self):
        """Run complete installation process"""
        logger.info("🚀 Starting ASI Finance Search Engine Installation...")
        logger.info("=" * 60)
        
        steps = [
            ("Installing Python packages", self.install_python_packages),
            ("Creating directories", self.create_directories),
            ("Creating configuration files", self.create_config_files),
            ("Creating startup script", self.create_startup_script),
            ("Creating test script", self.create_test_script),
            ("Installing spaCy model", self.install_spacy_model),
            ("Installing Playwright browsers", self.install_playwright_browsers),
            ("Verifying installation", self.verify_installation)
        ]
        
        failed_steps = []
        
        for step_name, step_function in steps:
            logger.info(f"\n📋 Step: {step_name}")
            logger.info("-" * 40)
            
            try:
                success = step_function()
                if success:
                    logger.info(f"✅ {step_name} - COMPLETED")
                else:
                    logger.warning(f"⚠️ {step_name} - COMPLETED WITH WARNINGS")
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"❌ {step_name} - FAILED: {e}")
                failed_steps.append(step_name)
        
        # Installation summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 INSTALLATION SUMMARY")
        logger.info("=" * 60)
        
        if not failed_steps:
            logger.info("🎉 Installation completed successfully!")
            logger.info("\n🚀 To start the ASI Finance Search Engine:")
            logger.info("   python start_asi_engine.py")
            logger.info("\n🧪 To run tests:")
            logger.info("   python run_tests.py")
            logger.info("\n📚 API Documentation will be available at:")
            logger.info("   http://localhost:8000/docs")
        else:
            logger.warning(f"⚠️ Installation completed with {len(failed_steps)} warnings:")
            for step in failed_steps:
                logger.warning(f"   - {step}")
            logger.info("\n💡 You may need to install some packages manually.")
        
        logger.info("\n🔗 For support and documentation:")
        logger.info("   Check the README.md file or contact the development team")
        
        return len(failed_steps) == 0

def main():
    """Main installation function"""
    installer = ASIFinanceEngineInstaller()
    success = installer.run_installation()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
