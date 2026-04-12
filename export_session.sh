#!/bin/bash

# export_session.sh <thread_id> [project_name]
# Example: ./export_session.sh 1256e69b-c5bd-4425-a70c-8c0726534296

if [ -z "$1" ]; then
    echo "Usage: $0 <thread_id> [project_name]"
    exit 1
fi

THREAD_ID=$1

# 1. Try to get Project Name from argument, then .env
PROJECT_NAME=$2
if [ -z "$PROJECT_NAME" ]; then
    PROJECT_NAME=$(grep "LANGSMITH_PROJECT" .env | sed 's/.*=//' | tr -d '"' | tr -d "'")
fi

# 2. Map LANGSMITH_API_KEY to LANGCHAIN_API_KEY for the SDK
if [ -z "$LANGCHAIN_API_KEY" ]; then
    export LANGCHAIN_API_KEY=$(grep "LANGSMITH_API_KEY" .env | sed 's/.*=//' | tr -d '"' | tr -d "'")
fi

echo "Exporting session $THREAD_ID from project '$PROJECT_NAME'..." >&2

uv run python -c "
import json
import os
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()
client = Client()

try:
    # Use read_thread to fetch the conversation history
    thread = client.read_thread(thread_id='$THREAD_ID', project_name='$PROJECT_NAME')
    
    output = []
    # thread is an iterable of Run objects
    for run in thread:
        output.append({
            'run_id': str(run.id),
            'name': run.name,
            'start_time': str(run.start_time),
            'inputs': run.inputs,
            'outputs': run.outputs,
            'metadata': run.extra.get('metadata', {})
        })
    print(json.dumps(output, indent=2, default=str))
except Exception as e:
    print(json.dumps({'error': str(e)}, indent=2))
"
