#!/bin/bash
# EMBench grader: reads the server-side tool-call log and /app/answer.md, writes /logs/verifier/reward.txt.
mkdir -p /logs/verifier
python3 /tests/grade.py --task-root /tests/embench_tasks --task-id "{{TASK_ID}}" --answer /app/answer.md --out /logs/verifier
