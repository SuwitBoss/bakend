import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - Limited for AI/ML memory usage
workers = min(multiprocessing.cpu_count(), 4)
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeouts - Critical for AI/ML workloads
timeout = 300  # 5 minutes for heavy ML processing
keepalive = 5
graceful_timeout = 60

# Memory management
worker_tmp_dir = "/dev/shm"  # Use RAM for temporary files
preload_app = True  # Essential for ML models - load once, share across workers

# Restart workers periodically to prevent memory leaks
max_requests = 100  # Restart after 100 requests for ML workloads
max_requests_jitter = 10

# Logging
loglevel = "info"
accesslog = "/app/logs/access.log"
errorlog = "/app/logs/error.log"

# Worker lifecycle hooks for ML models
def on_starting(server):
    """Called before workers start"""
    server.log.info("Starting Gunicorn server...")

def post_fork(server, worker):
    """Called after worker forked - Initialize ML models here"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)
