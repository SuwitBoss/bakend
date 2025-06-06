-- FaceSocial Database Initialization Script
-- This script creates the initial database structure for production PostgreSQL

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database if it doesn't exist (this will be handled by docker-compose)
-- The database is created automatically via POSTGRES_DB environment variable

-- Set timezone
SET timezone = 'UTC';

-- Create users table with proper constraints
-- Note: SQLAlchemy will handle table creation, this is just a backup/reference
-- The actual tables will be created by the application using Alembic migrations

-- You can add any additional database setup here
-- such as initial data, stored procedures, triggers, etc.

-- Example: Create a simple function for generating timestamps
CREATE OR REPLACE FUNCTION updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'FaceSocial database initialization completed successfully!';
END $$;
