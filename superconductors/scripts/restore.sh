#!/bin/bash
# Restore PostgreSQL database from backup

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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Restore PostgreSQL database from backup"
    echo ""
    echo "Options:"
    echo "  -f, --file FILENAME     Restore from local backup file"
    echo "  -c, --cloud FILENAME    Download and restore from GCS"
    echo "  -l, --list             List available backups"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list                                    # List available backups"
    echo "  $0 --file experiments_backup_20231201_143022.sql.gz"
    echo "  $0 --cloud experiments_backup_20231201_143022.sql.gz"
}

# Function to list available backups
list_backups() {
    echo -e "${YELLOW}📋 Available local backups:${NC}"
    if [ -d "${BACKUP_DIR}" ]; then
        ls -la "${BACKUP_DIR}"/experiments_backup_*.sql.gz 2>/dev/null | \
            awk '{print "  " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}' || \
            echo "  No local backups found"
    else
        echo "  No backup directory found"
    fi
    
    echo ""
    
    if command -v gsutil &> /dev/null; then
        echo -e "${YELLOW}☁️  Available cloud backups (latest 10):${NC}"
        gsutil ls -l "gs://${GCS_BUCKET}/backups/" | \
            grep -E "experiments_backup_[0-9]{8}_[0-9]{6}\.sql\.gz" | \
            sort -k2 -r | \
            head -10 | \
            awk '{
                split($3, path, "/");
                filename = path[length(path)];
                size = $1;
                date = $2;
                print "  " filename " (" size " bytes, " date ")"
            }' || echo "  No cloud backups found or no access"
    else
        echo -e "${YELLOW}⚠️  gcloud CLI not available - cannot list cloud backups${NC}"
    fi
}

# Function to restore from local file
restore_local() {
    local backup_file="$1"
    local full_path="${BACKUP_DIR}/${backup_file}"
    
    if [ ! -f "${full_path}" ]; then
        echo -e "${RED}❌ Backup file not found: ${full_path}${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}🔄 Restoring from local backup: ${backup_file}${NC}"
    
    # Check if file is compressed
    if [[ "${backup_file}" == *.gz ]]; then
        echo -e "${YELLOW}🗜️  Decompressing backup...${NC}"
        gunzip -c "${full_path}" | docker exec -i "${DB_CONTAINER}" psql -U "${DB_USER}" "${DB_NAME}"
    else
        cat "${full_path}" | docker exec -i "${DB_CONTAINER}" psql -U "${DB_USER}" "${DB_NAME}"
    fi
}

# Function to restore from cloud
restore_cloud() {
    local backup_file="$1"
    local temp_file="/tmp/${backup_file}"
    
    if ! command -v gsutil &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not available${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}☁️  Downloading from GCS: ${backup_file}${NC}"
    gsutil cp "gs://${GCS_BUCKET}/backups/${backup_file}" "${temp_file}"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to download backup from GCS${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}🔄 Restoring from cloud backup...${NC}"
    
    # Check if file is compressed
    if [[ "${backup_file}" == *.gz ]]; then
        echo -e "${YELLOW}🗜️  Decompressing backup...${NC}"
        gunzip -c "${temp_file}" | docker exec -i "${DB_CONTAINER}" psql -U "${DB_USER}" "${DB_NAME}"
    else
        cat "${temp_file}" | docker exec -i "${DB_CONTAINER}" psql -U "${DB_USER}" "${DB_NAME}"
    fi
    
    # Clean up temp file
    rm -f "${temp_file}"
}

# Function to confirm dangerous operation
confirm_restore() {
    echo -e "${RED}⚠️  WARNING: This will OVERWRITE the existing database!${NC}"
    echo -e "${YELLOW}Current database '${DB_NAME}' will be completely replaced.${NC}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        echo -e "${YELLOW}❌ Restore cancelled${NC}"
        exit 0
    fi
}

# Main script logic
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Check if PostgreSQL container is running
if ! docker ps | grep -q "${DB_CONTAINER}"; then
    echo -e "${RED}❌ PostgreSQL container '${DB_CONTAINER}' is not running${NC}"
    echo -e "${YELLOW}💡 Run: docker-compose -f docker-compose.simple.yml up -d${NC}"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--list)
            list_backups
            exit 0
            ;;
        -f|--file)
            backup_file="$2"
            if [ -z "${backup_file}" ]; then
                echo -e "${RED}❌ Please specify a backup file${NC}"
                exit 1
            fi
            confirm_restore
            restore_local "${backup_file}"
            shift 2
            ;;
        -c|--cloud)
            backup_file="$2"
            if [ -z "${backup_file}" ]; then
                echo -e "${RED}❌ Please specify a backup file${NC}"
                exit 1
            fi
            confirm_restore
            restore_cloud "${backup_file}"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

echo -e "${GREEN}✅ Database restore completed successfully!${NC}"
echo -e "${YELLOW}💡 You may want to restart your application containers${NC}"