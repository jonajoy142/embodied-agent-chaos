# Week 7 deliverable: reproducible container for the public repo.
# Minimal stub for now — flesh out with a headless PyBullet base image
# and pinned deps once the experiment matrix (Week 6) is finalized.

FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/run_scripted_pick_place.py"]
