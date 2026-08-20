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
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Run Mobile

```powershell
cd D:\firstmodel\knowledge-space-app\mobile
npm install
npx expo install expo-document-picker
npm run start
```

The app expects the API at `http://127.0.0.1:8000` for local preview.
