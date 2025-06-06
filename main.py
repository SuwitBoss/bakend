#!/usr/bin/env python3
"""
Main entry point for the FaceSocial backend.
This file imports the app from the app module and exposes it for Gunicorn.
"""

from app.main import app

# This allows Gunicorn to find the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
