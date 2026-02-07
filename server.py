"""
Server entry point with explicit .env file loading.

This file ensures environment variables are loaded before the application starts.
Run this file to start the server: python server.py
"""

from pathlib import Path

# Try to load .env file using python-dotenv if available
try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print(f"⚠ .env file not found at {env_path}")
        # Try to load from current directory
        load_dotenv(override=False)
        print("✓ Attempted to load .env from current directory")
except ImportError:
    print("⚠ python-dotenv not installed. Install it with: pip install python-dotenv")
    print("⚠ Environment variables must be set manually or via system environment")

# Environment variables are loaded, ready to start

# Now import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    from app.settings import settings

    print(f"🚀 Starting {settings.app_name}...")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"📚 Docs: http://{settings.host}:{settings.port}/docs")
    print(f"🌍 Environment: {settings.environment}\n")

    log_level = str(settings.log_level).lower() if hasattr(settings.log_level, "lower") else "info"
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=log_level,
    )
else:
    # When imported as a module, just ensure .env is loaded
    pass
