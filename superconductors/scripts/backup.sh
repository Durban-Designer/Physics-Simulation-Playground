#!/bin/bash
# Backup PostgreSQL database to Google Cloud Storage

set -e  # Exit on any error

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Configuration
DB_CONTAINER="research-postgres"
DB_NAME="experiments"
DB_USER="postgres"
GCS_BUCKET="${GCS_BUCKET:-superconductor-research-backups}"
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="experiments_backup_${DATE}.sql"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 Starting database backup...${NC}"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Check if PostgreSQL container is running
if ! docker ps | grep -q "${DB_CONTAINER}"; then
    echo -e "${RED}❌ PostgreSQL container '${DB_CONTAINER}' is not running${NC}"
    exit 1
fi

# Create database backup
echo -e "${YELLOW}📊 Creating PostgreSQL dump...${NC}"
docker exec "${DB_CONTAINER}" pg_dump -U "${DB_USER}" "${DB_NAME}" > "${BACKUP_DIR}/${BACKUP_FILE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database dump created: ${BACKUP_FILE}${NC}"
else
    echo -e "${RED}❌ Failed to create database dump${NC}"
    exit 1
fi

# Compress the backup
echo -e "${YELLOW}🗜️  Compressing backup...${NC}"
gzip "${BACKUP_DIR}/${BACKUP_FILE}"
COMPRESSED_FILE="${BACKUP_FILE}.gz"

# Check if gcloud CLI is available and configured
if command -v gsutil &> /dev/null; then
    echo -e "${YELLOW}☁️  Uploading to Google Cloud Storage...${NC}"
    
    # Upload to GCS
    gsutil cp "${BACKUP_DIR}/${COMPRESSED_FILE}" "gs://${GCS_BUCKET}/backups/${COMPRESSED_FILE}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Backup uploaded to GCS: gs://${GCS_BUCKET}/backups/${COMPRESSED_FILE}${NC}"
        
        # Clean up old backups (keep last 30 days)
        echo -e "${YELLOW}🧹 Cleaning up old backups...${NC}"
        gsutil ls -l "gs://${GCS_BUCKET}/backups/" | \
            grep -E "experiments_backup_[0-9]{8}_[0-9]{6}\.sql\.gz" | \
            sort -k2 | \
            head -n -30 | \
            awk '{print $3}' | \
            xargs -r gsutil rm
            
        echo -e "${GREEN}✅ Old backups cleaned up${NC}"
    else
        echo -e "${RED}❌ Failed to upload backup to GCS${NC}"
        echo -e "${YELLOW}💾 Backup saved locally: ${BACKUP_DIR}/${COMPRESSED_FILE}${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  gcloud CLI not found. Backup saved locally only.${NC}"
    echo -e "${YELLOW}💾 Local backup: ${BACKUP_DIR}/${COMPRESSED_FILE}${NC}"
fi

# Clean up old local backups (keep last 7 days)
echo -e "${YELLOW}🧹 Cleaning up old local backups...${NC}"
find "${BACKUP_DIR}" -name "experiments_backup_*.sql.gz" -mtime +7 -delete

# Display backup information
BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${COMPRESSED_FILE}" | cut -f1)
echo -e "${GREEN}📦 Backup completed successfully!${NC}"
echo -e "   File: ${COMPRESSED_FILE}"
echo -e "   Size: ${BACKUP_SIZE}"
echo -e "   Location: ${BACKUP_DIR}/"

# Optional: Send notification (if configured)
if [ ! -z "${SLACK_WEBHOOK_URL}" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🔄 Database backup completed: ${COMPRESSED_FILE} (${BACKUP_SIZE})\"}" \
        "${SLACK_WEBHOOK_URL}" &> /dev/null
fi

echo -e "${GREEN}✨ Backup process completed at $(date)${NC}"