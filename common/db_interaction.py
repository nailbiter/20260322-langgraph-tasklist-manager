import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
from dotenv import load_dotenv

from common.utils.logging import get_configured_logger

load_dotenv()

# --- Logging ---
_log_file = os.path.join(
    ".logs", f"db_interaction-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log.txt"
)
logger = get_configured_logger("db_interaction", log_to_file=_log_file)

# --- Configuration & Clients ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client.get_database("gstasks")
tasks_col = db.get_collection("tasks")
tags_col = db.get_collection("tags")


def get_tag_map() -> (dict, dict):
    """Maps UUID -> Name and Name -> UUID from the tags collection."""
    tags = list(tags_col.find({}, {"uuid": 1, "name": 1}))
    uuid_to_name = {t["uuid"]: t["name"] for t in tags}
    name_to_uuid = {t["name"]: t["uuid"] for t in tags}
    return uuid_to_name, name_to_uuid


def normalize_date(date_val) -> str:
    """Handles both BSON $date objects and ISO strings for readability."""
    _logger = logger.getChild("normalize_date")
    if isinstance(date_val, dict) and "$date" in date_val:
        res = date_val["$date"]
    else:
        res = str(date_val)
    _logger.debug(dict(x=date_val, y=res))
    return res


def fetch_mongo_tasks(limit=100):
    _logger = logger.getChild("fetch_mongo_tasks")
    """Fetches and normalizes tasks from MongoDB for the demo state."""
    u_to_n, _ = get_tag_map()
    raw_tasks = list(
        tasks_col.find(
            {
                "$and": [
                    {"scheduled_date": {"$gte": datetime(2026, 2, 7)}},
                    {"status": {"$ne": "DONE"}},
                    {"status": {"$ne": "FAILED"}},
                    {"status": {"$ne": "CANCELLED"}},
                ]
            }
        )
        .sort("_id", -1)
        .limit(limit)
    )
    _logger.debug(f"downloaded {len(raw_tasks)} tasks")
    normalized = []
    for t in raw_tasks:
        # Ensure every task has a uuid (fallback to stringified _id)
        task_uuid = t.get("uuid") or str(t["_id"])

        # Resolve tag UUIDs to names for the LLM/Studio context
        resolved_tags = [
            u_to_n.get(tag_uuid, tag_uuid) for tag_uuid in t.get("tags", [])
        ]

        normalized.append(
            {
                "uuid": task_uuid,
                "name": t.get("name", "Untitled Task"),
                "status": t.get("status", "OPEN"),
                "scheduled_date": normalize_date(t.get("scheduled_date")),
                "URL": t.get("URL"),
                "tags": resolved_tags,
                "comment": t.get("comment"),
            }
        )
    _logger.debug(f"normalized: {normalized}")
    return normalized


def _resolve_tags(tags: List[str]) -> List[str]:
    """Helper to map human-readable names back to UUIDs."""
    if not tags:
        return []
    _, n_to_u = get_tag_map()
    return [n_to_u.get(tag, tag) for tag in tags]


def insert_task(task_data: dict):
    """Inserts a new task into MongoDB, resolving tag names to UUIDs."""
    if "tags" in task_data:
        task_data["tags"] = _resolve_tags(task_data["tags"])
    return tasks_col.insert_one(task_data)


def update_task_by_uuid(task_uuid: str, updates: dict):
    """Updates a task in MongoDB by its custom 'uuid' field, mapping tag names back to UUIDs."""
    _logger = logger.getChild("fetch_mongo_tasks")
    if "tags" in updates:
        updates["tags"] = _resolve_tags(updates["tags"])
    if "scheduled_date" in updates:
        updates["scheduled_date"] = pd.to_datetime(
            normalize_date(updates["scheduled_date"])
        )
    _logger.debug(dict(task_uuid=task_uuid, updates=updates))
    return tasks_col.update_one({"uuid": task_uuid}, {"$set": updates})


# def update_task_by_id(task_id: str, updates: dict):
#     """Updates a task in MongoDB by its BSON '_id'."""
#     return tasks_col.update_one({"_id": ObjectId(task_id)}, {"$set": updates})


# def find_tasks(query_filter: dict, limit: int = 50):
#     """Generic find wrapper for tasks."""
#     return list(tasks_col.find(query_filter).limit(limit))
