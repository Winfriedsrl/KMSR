# winfried-text-to-cpi

Retrieval of the *rubrica imputativa* from a natural-language description: the Winfried platform
and its evaluation tooling. Home of **KMSR** (Knowledge-Guided Multi-Stage Retrieval), the
knowledge-guided pipeline presented in:

> **The Impact of Domain-Specific Knowledge in Legal AI: A Case Study on Italian Criminal Law**

📄 Reproduce it → **[paper_experiments/](paper_experiments/README.md)**

This README covers **operating the system**: Docker, Streamlit, Elasticsearch and the generic CLI.

---

Two supported modes:
- `VM`: Nginx on the host with password (protected external access)
- `Local`: no Nginx, direct access from localhost

## Compose files

- `docker-compose.yaml`: `VM` mode (Kibana behind `/kibana`)
- `docker-compose.local.yaml`: `Local` mode (Kibana directly on port `5602`)

## Common prerequisites

- Docker + Docker Compose plugin
- A Python environment for Streamlit

## Starting Docker

`VM` mode:

```bash
docker compose -f docker-compose.yaml up -d
```

`Local` mode:

```bash
docker compose -f docker-compose.local.yaml up -d
```

Check:

```bash
docker compose ps
```

## Streamlit

Before starting Streamlit, create a `.env` file in the project root with ALL of these variables:

```bash
OPENAI_API_KEY="sk-..."
ES_HOST="http://localhost:9201"
ES_INDEX_NAME="periodi_index_name"
ES_INDEX_NAME_COMBO="combo_index_name"
```

### VM mode (behind Nginx on `/streamlit`)

```bash
streamlit run src/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.baseUrlPath /streamlit
```

### Local mode (no Nginx)

```bash
streamlit run src/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

## Nginx on the host (VM mode only)

Install and configure:

```bash
sudo apt update
sudo apt install -y nginx apache2-utils
sudo cp nginx/default.conf /etc/nginx/sites-available/winfried
sudo ln -sf /etc/nginx/sites-available/winfried /etc/nginx/sites-enabled/winfried
sudo rm -f /etc/nginx/sites-enabled/default
sudo htpasswd -c /etc/nginx/.htpasswd admin
sudo nginx -t
sudo systemctl restart nginx
```

Changing the password later:

```bash
sudo htpasswd /etc/nginx/.htpasswd admin
sudo systemctl restart nginx
```

## URLs

`VM` mode:
- `http://<VM_IP>/kibana/`
- `http://<VM_IP>/streamlit/`

`Local` mode:
- `http://localhost:5602`
- `http://localhost:8501`

## UFW firewall (VM mode only)

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw deny 8501/tcp
sudo ufw deny 5602/tcp
sudo ufw deny 9201/tcp
sudo ufw enable
sudo ufw status
```

## Stop

```bash
docker compose -f docker-compose.yaml down
# or
# docker compose -f docker-compose.local.yaml down
```

## Indexing (CLI)

> ⚠️ Generic reference for the indexing command. **These are not the paper's indices**: corpora,
> file paths and index names differ. To build the paper's corpora use
> [paper_experiments/build_indices.sh](paper_experiments/build_indices.sh) — running the commands
> below instead will silently produce indices the paper configs do not point at.

Prerequisites:
- Elasticsearch running (port `9201`)
- a Python environment with the dependencies installed (`requirements.txt`)
- a complete `.env` file (see the Streamlit section)

Minimum required:

```bash
OPENAI_API_KEY="sk-..."
ES_HOST="http://localhost:9201"
ES_INDEX_NAME="periodi_index_name"
ES_INDEX_NAME_COMBO="combo_index_name"
```

Base command:

```bash
python3 src/main.py index \
  --name_prefix wf_periodi \
  --data_source data \
  --dataset_type periodi \
  --embedders text-embedding-3-large
```

`combo` dataset:

```bash
python3 src/main.py index \
  --name_prefix wf_combo \
  --data_source data \
  --dataset_type combo \
  --embedders text-embedding-3-large
```

`reati` dataset:

```bash
python3 src/main.py index \
  --name_prefix wf_reati \
  --data_source data \
  --dataset_type reati \
  --embedders text-embedding-3-large
```

`reati_circostanziati_full` dataset:

```bash
python3 src/main.py index \
  --name_prefix wf_reati_circ_full \
  --data_source data \
  --dataset_type reati_circostanziati_full \
  --embedders text-embedding-3-large
```

`reati_circostanziati_light` dataset:

```bash
python3 src/main.py index \
  --name_prefix wf_reati_circ_light \
  --data_source data \
  --dataset_type reati_circostanziati_light \
  --embedders text-embedding-3-large
```

BM25 only (no embeddings):

```bash
python3 src/main.py index \
  --name_prefix bm25idx \
  --data_source data \
  --dataset_type periodi
```

Notes:
- The index is named after `--name_prefix` as given. If that name is already taken,
  `make_index_name()` appends a counter (`wf_periodi_2`, `wf_periodi_3`, …) instead of failing —
  so re-indexing on a non-empty Elasticsearch creates a *new* index and leaves the old one in
  place. Delete first (`curl -X DELETE "$ES_HOST/<name>"`) if you meant to rebuild.
- To list the indices that exist:

```bash
curl -s "http://localhost:9201/_cat/indices?v"
```

## Evaluation (CLI)

> ⚠️ This is the `evaluate` command, for ad-hoc evaluation on a GT csv. The paper uses a
> different subcommand (`evaluate-paper --hard`), with its own ground truth and metrics: see
> [paper_experiments/README.md](paper_experiments/README.md) §4.

The same engine as the Streamlit `Valutazione CSV` view, runnable from the terminal.

Ground-truth input format:
- a CSV like `data/gt/wf_training_set.csv`
- columns: `query`, `result`, `type`

Model config format:
- a `.json` or `.yaml` file holding a list of model objects
- same structure as the GUI configuration

Ready-made example config:
- `data/model_configs/example_eval_models.yaml`

Base command:

```bash
python3 src/main.py evaluate \
  --gt_csv data/gt/wf_training_set.csv \
  --models_config data/model_configs/example_eval_models.yaml
```

Default output:
- prints the summary to the terminal
- always saves a new run under `runs/evaluations/<timestamp>/`
- files written: `summary.csv`, `details.csv`, `meta.json`

Saving the summary with a redirect as well:

```bash
python3 src/main.py evaluate \
  --gt_csv data/gt/wf_training_set.csv \
  --models_config data/model_configs/example_eval_models.yaml \
  > out/eval_summary.csv
```

## Elasticsearch snapshot/restore

```bash
curl -X PUT "http://localhost:9201/_snapshot/local_fs" \
  -H "Content-Type: application/json" \
  -d '{"type":"fs","settings":{"location":"/usr/share/elasticsearch/snapshots"}}'

curl -X PUT "http://localhost:9201/_snapshot/local_fs/snap_001?wait_for_completion=true"
```

Then copy `es_snapshots/` to the new VM and restore it there.
