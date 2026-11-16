# RUN.md — Lancer le service SIE en local

Guide reproduit de bout en bout sur **macOS 24.6.0 (Darwin, Apple Silicon)** le
2026-11-16. Chaque commande ci-dessous a été exécutée. Les cas non testés sont
signalés explicitement.

Le service est un endpoint FastAPI : il reçoit un (ou plusieurs) document(s)
scanné(s) + un schéma YAML, convertit le document en Markdown avec
[docling](https://github.com/DS4SD/docling) (OCR + structure de tables, forcé
sur CPU), puis demande à un LLM local servi par **Ollama** de remplir un
`BaseModel` Pydantic généré à la volée à partir du schéma.

## 1. Prérequis

| Composant | Version testée | Comment vérifier |
|---|---|---|
| macOS   | 24.6.0 / Apple Silicon | `uname -a` |
| Python  | 3.13.7 (le repo cible ≥ 3.11, `.python-version` fixe 3.11) | `python3 --version` |
| uv      | 0.9.21 | `uv --version` |
| Ollama  | 0.34.2 | `ollama --version` |
| RAM     | ≥ 16 Go recommandé (llama3.1 8B Q4 ≈ 5 Go + modèles docling 1–2 Go) | |
| Disque  | ≈ 10 Go (venv + modèles docling + poids Ollama) | |

Installer uv si absent :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Installer Ollama depuis <https://ollama.com/download>.

## 2. Installation

Cloner le repo puis, à la racine :

```bash
uv sync
```

`uv sync` crée `.venv/` et installe les versions pinnées par `uv.lock`. Aucune
activation manuelle n'est nécessaire — on utilise `uv run …` partout.

Vérification rapide :

```bash
uv run python -c "import docling, fastapi, openai, pydantic, yaml, tenacity, uvicorn; print('ok')"
# → ok
```

## 3. Configuration

L'URL d'Ollama est codée en dur (`base_url="http://localhost:11434/v1"` dans
`client.py`). Le nom du modèle est **configurable par variable
d'environnement** :

| Variable       | Défaut            | Rôle                                                                                          |
|----------------|-------------------|-----------------------------------------------------------------------------------------------|
| `OLLAMA_MODEL` | `llama3.1:latest` | Tag Ollama passé tel quel au client OpenAI-compatible (`chat.completions.create(model=…)`).  |

Exemples pour le benchmark multi-modèles :

```bash
OLLAMA_MODEL=mistral:7b uv run uvicorn api:app --host 127.0.0.1 --port 8000
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

Récupérer le modèle par défaut (≈ 4.9 Go) :

```bash
ollama pull llama3.1
```

Vérifier qu'il est bien là :

```bash
ollama list | grep llama3.1
# llama3.1:latest    8.0B    Q4_K_M    ~4.9 GB
```

> Note historique : le code public référençait `llama3.1:8b`. Avec la variable
> d'environnement, n'importe quel tag Ollama installé localement convient.
> Pour les pièges de résolution de tag, voir § 6 « Problèmes fréquents ».

## 4. Lancement

Deux processus doivent tourner.

### 4.1 Ollama

Sur macOS l'application Ollama (ou `brew services start ollama`) démarre un
serveur qui écoute sur `127.0.0.1:11434`. Test :

```bash
curl -s http://localhost:11434/api/tags | head -c 120
# → {"models":[{"name":"llama3.1:latest", ...
```

Si rien ne répond, lancer le daemon à la main :

```bash
ollama serve
```

(Garder ce terminal ouvert, ou le lancer en background.)

### 4.2 API FastAPI

Depuis la racine du repo :

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

Logs attendus :

```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Test de santé :

```bash
curl -s http://127.0.0.1:8000/health
# → {"status":"ok"}
```

Swagger : <http://127.0.0.1:8000/docs>.

### 4.3 Premier appel : download des modèles docling

Le **premier** appel à `/extract` déclenche le téléchargement des modèles
docling (layout, structure de tables, poids OCR). Dans notre run : **2 min 16 s
pour un BL d'une page**. Les appels suivants, modèles en cache, prennent
**≈ 24 s** pour le même document sur le même poste.

## 5. Utilisation

### 5.1 Endpoints

| Méthode | Chemin     | Description |
|---------|------------|-------------|
| GET     | `/health`  | `{"status":"ok"}` |
| POST    | `/extract` | `multipart/form-data`. Champs : `files` (1 ou plusieurs documents) + `schema_file` (un YAML unique). |

Formats de documents acceptés : `.pdf .png .jpg .jpeg .tiff .tif .bmp .webp`.
Extensions schéma acceptées : `.yaml .yml`.

**Limites :**
- 10 Mo max par fichier (code : `MAX_FILE_BYTES` dans `api.py:30`).
- 5 documents traités en parallèle (`MAX_CONCURRENT_DOCS`), les suivants sont
  mis en file d'attente par le `ThreadPoolExecutor`.

**Codes HTTP :**
- 200 : au moins un document traité avec succès.
- 400 : schéma invalide (YAML mal formé, extension erronée, fichier vide) **ou
  tous** les documents en erreur.
- Le rapport YAML conserve en sortie un item par document, avec `status:
  success | error`.

### 5.2 Exemple curl réel

Schéma utilisé : `schemas/base.yaml` du repo (connaissement maritime, voir
§ 5.3).

```bash
curl -sS -X POST http://127.0.0.1:8000/extract \
  -F "files=@bl.png;type=image/png" \
  -F "schema_file=@schemas/base.yaml;type=application/x-yaml" \
  -o response.yaml -w "HTTP=%{http_code}\n"
```

Réponse (`cat response.yaml`) obtenue ici avec un BL fictif généré pour les
tests (`bl.png`) :

```yaml
- source_file: bl.png
  status: success
  extracted_data:
    company_name: OCEANLINE SHIPPING CO.
    bl_number: OLSC-2026-00428175
    shipper: ACME Exports Ltd., 12 Harbour Rd, Singapore 049318
    port_of_loading: Singapore (SGSIN)
    port_of_discharge: Rotterdam (NLRTM)
    place_of_acceptance: Singapore CY
    place_of_delivery: Rotterdam CY
    shipped_on_board_date: 14 OCT 2026
    vessel_name: MV NORTHERN STAR
    voyage_number: V-114E
    mark: ACME/RTM/LOT-A
    container_numbers:
    - MRKU7061784
    - MSKU4412309
    - TGHU5567201
    lots:
    - 20 pallets electronic components
    - 18 pallets textile g0ods   # artefact OCR réel (voir §6)
    - 22 pallets automotive parts
```

`HTTP=200`.

### 5.3 Format du schéma YAML

Chaque clé top-level = un champ à extraire. Sa valeur est un dict `{type,
description}`.

- `type` ∈ `string | integer | float | boolean | list` (mapping dans
  `models.py:TYPE_MAPPING`).
- `description` est injectée dans la prompt système envoyée au LLM via le YAML
  complet ; c'est votre seul levier de pilotage de l'extraction.
- Tous les champs deviennent `Optional` avec `None` par défaut. Si **tous** les
  champs ressortent à `None`, `client.py` lève une `ValueError` et le retry
  tenacity relance jusqu'à 3 tentatives.

Exemple complet (reprise de `schemas/base.yaml`) :

```yaml
company_name:
  type: string
  description: "Nom officiel de la compagnie maritime, du transporteur ou de la ligne."

bl_number:
  type: string
  description: "Numéro de B/L (Bill of Lading)."

shipper:
  type: string
  description: "Expéditeur / consignor."

port_of_loading:
  type: string
  description: "Port d'embarquement."

port_of_discharge:
  type: string
  description: "Port de débarquement."

vessel_name:
  type: string
  description: "Nom du navire."

container_numbers:
  type: list
  description: "Liste des numéros de conteneur (format : 4 lettres + 7 chiffres, ex. MRKU7061784). Un élément par conteneur."
```

### 5.4 Un cas d'erreur (vérifié)

Fichier YAML vide → HTTP 400 :

```bash
: > empty.yaml
curl -sS -X POST http://127.0.0.1:8000/extract \
  -F "files=@bl.png;type=image/png" \
  -F "schema_file=@empty.yaml;type=application/x-yaml" \
  -w "\nHTTP=%{http_code}\n"
# {"detail":"Schema error, please check the schema file you provided."}
# HTTP=400
```

Document non supporté (seul fichier) → HTTP 400, mais la cause apparaît dans le
corps YAML :

```bash
echo hello > doc.txt
curl -sS -X POST http://127.0.0.1:8000/extract \
  -F "files=@doc.txt;type=text/plain" \
  -F "schema_file=@schemas/base.yaml;type=application/x-yaml" \
  -w "\nHTTP=%{http_code}\n"
# - source_file: doc.txt
#   status: error
#   error: 'Format non supporté : ''doc.txt''.'
# HTTP=400
```

Fichier > 10 Mo → HTTP 400 (`"<nom>" dépasse 10 Mo.`). Vérifié avec un fichier
zero-padded de 11 Mo.

## 6. Problèmes fréquents (réellement rencontrés)

### 6.1 `model 'llama3.1:8b' not found` (statut HTTP 200 côté API mais `status: error` dans le YAML)

La version publiée de `client.py:36` référençait `llama3.1:8b`. Ollama n'alias
pas automatiquement `llama3.1:latest` vers `llama3.1:8b`, donc même si vous
avez le modèle 8B installé sous le tag `latest`, le serveur retourne
`not_found_error`. Trois solutions :

- **Patché dans ce repo** : `client.py` pointe désormais sur `llama3.1:latest`.
- Sinon, créer un alias local sans télécharger :
  ```bash
  ollama cp llama3.1:latest llama3.1:8b
  ```
- Ou tirer le tag exact :
  ```bash
  ollama pull llama3.1:8b
  ```

### 6.2 L'API ne répond pas pendant ~2 minutes au premier `/extract`

Attendu. Docling télécharge ses modèles (layout analysis, table structure,
OCR). À froid sur Apple Silicon M2 : 2 min 16 s pour un BL d'une page ; à
chaud : 24 s. Augmenter le timeout curl en conséquence (`--max-time 300`).

### 6.3 Le polling `/health` échoue immédiatement après `uvicorn`

Le process démarre en quelques secondes, mais les toutes premières
sollicitations peuvent tomber en "Connection refused". Boucler sur une
poll-check courte (ex. toutes les 3 s pendant 30 s) suffit.

### 6.4 Artéfacts OCR dans la sortie

Observé : `textile g0ods` (0 à la place de "o") dans la liste `lots`. C'est
une caractéristique du moteur OCR docling sur image rastérisée, pas un bug
applicatif. Pour des documents de production, préférer des PDFs natifs (texte
sélectionnable) aux scans quand c'est possible.

### 6.5 Tous les champs sortent à `null`

`client.py:90` lève une `ValueError` quand `all(value is None for ...)` et
`tenacity` retente. Si l'erreur persiste après 3 tentatives, c'est généralement
que le Markdown rendu par docling est vide (image illisible, PDF endommagé) ou
que le schéma décrit des champs introuvables dans ce document.

### 6.6 Le code demande Python ≥ 3.11 mais `.python-version` dit 3.11

`.python-version` indique à uv quel interpréteur créer dans `.venv/`. Tout
Python ≥ 3.11 fonctionne. Testé ici avec Python 3.13.7 installé par uv sans
anicroche.

## 7. Ce qui n'est pas testé dans ce RUN

- **PDF natif multi-pages** (seul un PNG d'une page a été testé). L'extracteur
  est configuré pour PDF et images avec le même pipeline OCR + tables ; aucune
  raison à priori qu'il échoue, mais non vérifié ici.
- **Batch de plusieurs documents dans le même appel** (jusqu'à 5 en parallèle).
  Le code gère `list[UploadFile]`, mais notre test se limitait à un document.
- **Linux x86_64 / WSL.** Non testé.
- **Le provider `CLOUD = "groq"`** de l'enum `LLMProvider` (`client.py:23`). Le
  code ne configure *aucun* client pour cette valeur — `_configure_client`
  lève `ValueError` si on la passe. C'est donc du code mort aujourd'hui.
