# MindSpace Knowledge App

MindSpace is a mobile-first knowledge space prototype for the software innovation competition.

## Structure

- `mobile/`: Expo + React Native + TypeScript app prototype.
- `backend/`: FastAPI backend with SQLite, document ingestion, and keyword search.

## V1 Goal

1. Create knowledge spaces.
2. Upload PDF, Word, or text files.
3. Extract text into searchable chunks with source metadata.
4. Search inside one space.
5. Return matched snippets with file name and page/paragraph location.

## Run Backend

```powershell
cd D:\firstmodel\knowledge-space-app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Local Embedding Search

If you want semantic search to run locally, install `FlagEmbedding` in the backend environment and keep the default model:

```powershell
pip install FlagEmbedding
```

Optional environment variables:

```powershell
$env:EMBEDDING_MODEL_NAME="BAAI/bge-m3"
$env:EMBEDDING_BATCH_SIZE="16"
$env:EMBEDDING_MAX_LENGTH="8192"
```

The first run may download model weights. If `FlagEmbedding` is not installed, the app still works with keyword and same-page search.

## Run Mobile

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://你的电脑局域网IP:8000"
cd D:\firstmodel\knowledge-space-app\mobile
npm install
npx expo install expo-document-picker
npm run start
```

The app will auto-detect the Metro host in most LAN setups. If the phone still cannot reach the backend, set `EXPO_PUBLIC_API_BASE_URL` to your computer's LAN IP, such as `http://192.168.1.222:8000`.
