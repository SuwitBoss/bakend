-- FaceSocial Database Initialization Script
-- This script creates the initial database structure for production PostgreSQL

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Set timezone
SET timezone = 'UTC';

-- Create users table with proper constraints
-- Note: SQLAlchemy will handle table creation, this is just a backup/reference
-- The actual tables will be created by the application using Alembic migrations

-- Create vector-based functions for face similarity search
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector) 
RETURNS float AS $$
  SELECT 1 - (a <=> b) as similarity;
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

-- Example: Create a simple function for generating timestamps
CREATE OR REPLACE FUNCTION updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create face_embeddings table for storing face vectors with pgvector
CREATE TABLE IF NOT EXISTS face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    embedding vector(512) NOT NULL,  -- 512-dimensional face embedding vector
    photo_id UUID,  -- Reference to the photo this embedding came from
    is_primary BOOLEAN DEFAULT false,  -- Whether this is the primary face for the user
    confidence FLOAT NOT NULL,  -- Detection confidence score
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for fast vector similarity search
CREATE INDEX IF NOT EXISTS face_embedding_vector_idx ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'FaceSocial database initialization completed successfully!';
END $$;
