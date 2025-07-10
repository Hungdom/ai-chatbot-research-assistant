#!/bin/bash
ES_URL="http://localhost:9200"

# Đợi Elasticsearch sẵn sàng
until curl -s -XGET "$ES_URL" | grep -q '"cluster_name"'; do
  echo "$(date) - Waiting for Elasticsearch to be ready..."
  sleep 5
done

echo "Elasticsearch is ready."

create_index_if_not_exists() {
  local INDEX_NAME=$1
  local INDEX_MAPPING=$2

  index_exists=$(curl -s -o /dev/null -w "%{http_code}" "$ES_URL/$INDEX_NAME")

  if [ "$index_exists" -eq 200 ]; then
    echo "Index '$INDEX_NAME' already exists. Keeping old data."
  else
    echo "Index '$INDEX_NAME' does not exist. Creating..."
    curl -X PUT "$ES_URL/$INDEX_NAME" -H 'Content-Type: application/json' -d"$INDEX_MAPPING"
    echo "Index '$INDEX_NAME' created."
  fi
}

arxiv_mapping='
{
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 1
  },
  "mappings": {
    "dynamic": false,
    "properties": {
      "id": { "type": "long" },
      "arxiv_id": { "type": "keyword" },
      "title": {
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "authors": {
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "abstract": { "type": "text" },
      "categories": { "type": "keyword" },
      "published_date": { "type": "date" },
      "comment": { "type": "text" },
      "journal_ref": {
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "created_at": { "type": "date" }
    }
  }
}'

create_index_if_not_exists "arxiv_index" "$arxiv_mapping"
