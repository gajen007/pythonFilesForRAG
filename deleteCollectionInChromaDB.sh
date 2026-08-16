#!/bin/bash
# Usage: ./delete_collection.sh <collection_name>
# Example: ./delete_collection.sh docs

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <collection_name>"
  exit 1
fi

COLLECTION_NAME="$1"
DB_DIR="$HOME/rag-test/chroma_db"

python3 - "$COLLECTION_NAME" "$DB_DIR" <<'EOF'
import sys
import chromadb

collection_name = sys.argv[1]
db_dir = sys.argv[2]

client = chromadb.PersistentClient(path=db_dir)

try:
    client.delete_collection(collection_name)
    print(f"Deleted collection: {collection_name}")
except Exception as e:
    print(f"Could not delete collection '{collection_name}': {e}")
    sys.exit(1)
EOF