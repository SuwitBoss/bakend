# FaceSocial Backend API

Face recognition application backend with AI capabilities for social authentication and glasses detection.

## Features

- Face detection and recognition
- Advanced glasses detection
- User authentication and registration
- AI-powered face analysis
- Docker support for deployment

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- Docker and Docker Compose (for containerized deployment)

### Setup with Docker

1. Clone the repository
2. Navigate to the project directory
3. Create `.env` file from `.env.example`
4. Run with Docker Compose:

```bash
docker-compose up
```

### Local Development Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env` file

4. Run the application:

```bash
uvicorn app.main:app --reload
```

## API Documentation

When the application is running, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

This project is licensed under the MIT License - see the LICENSE file for details.
