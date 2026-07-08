#!/bin/bash
# scripts/bootstrap.sh — First-time setup

set -e

echo "Starting infrastructure..."
docker compose up -d redpanda redis postgres

echo "Waiting for PostgreSQL..."
until docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do
    sleep 1
done

echo "Seeding database..."
docker compose --profile seed run --rm db-seed

echo "Hydrating Redis..."
docker compose --profile seed run --rm hydrate-redis

echo "Starting pipeline..."
docker compose up -d