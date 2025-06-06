"""
Gunicorn config file for FastAPI applications
"""

# Number of workers - adjust based on CPU cores
import multiprocessing
workers = min(multiprocessing.cpu_count() * 2, 8)

# Use Uvicorn worker for ASGI compatibility with FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Socket binding
bind = "0.0.0.0:8000"

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Timeout configuration
timeout = 120
keepalive = 5

# Prevent Gunicorn from using threads
threads = 1

# Restart workers periodically to prevent memory leaks
max_requests = 1000
max_requests_jitter = 200

# Worker connections
worker_connections = 1000

# Process naming
proc_name = "facesocial_backend"

# Automatically reload on code changes
reload = True

# Pre-load application to catch errors before forking
preload_app = True

# Exit workers gracefully
graceful_timeout = 30

# Specify ASGI app
wsgi_app = "app.main:app"