# SenTree GNN Playback

Standalone React app for the GNN training animation.

## Data export

From the repo root:

```bash
.venv/bin/python scripts/export_gnn_playback_data.py
```

This writes:

```text
apps/gnn-playback/public/data/gnn_training_history.json
```

## Run the app

```bash
cd apps/gnn-playback
npm install
npm run dev
```

Default dev URL:

```text
http://127.0.0.1:4173
```

## Build

```bash
npm run build
npm run preview
```
