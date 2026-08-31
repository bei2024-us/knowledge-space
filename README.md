# MindSpace 知识空间

MindSpace 是一个面向学习资料和工作资料管理的移动优先知识空间应用。用户可以创建知识空间，上传 PDF、Word、TXT、Markdown、音频和视频资料，系统会自动解析内容并切分成可搜索片段，搜索时返回命中文本、来源文件和页码、段落或音视频时间戳，帮助用户快速定位原始资料。

## 核心功能

- 知识空间管理：创建多个知识空间，并在空间内按文件夹组织资料。
- 多格式资料解析：支持 `.pdf`、`.docx`、`.txt`、`.md`，以及常见音频和视频格式。
- 可追溯搜索：搜索结果展示命中文本、文件名、文件夹和原文位置。
- 中文友好检索：结合中文分词、关键词扩展、近义词扩展、同页命中和 SQLite FTS。
- 扫描件处理：PDF 文本不足时可通过 OCRmyPDF 尝试 OCR。
- 音视频转写：通过 faster-whisper 将音频或视频内容转写为带时间戳的片段。
- 语义搜索增强：可选安装 FlagEmbedding，使用 BGE-M3 进行本地向量检索。
- 移动端体验：提供 Expo + React Native 移动端原型，也提供浏览器 Web 演示页。

## 技术栈

- Backend：FastAPI、SQLite、PyMuPDF、python-docx、jieba、OCRmyPDF、faster-whisper、NumPy。
- Optional AI Search：FlagEmbedding、BAAI/bge-m3。
- Frontend：Expo、React Native、TypeScript、react-native-web。
- Storage：本地 SQLite 数据库和本地上传文件目录。

## 项目结构

```text
knowledge-space/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 接口、搜索、预览、上传逻辑
│   │   ├── db.py            # SQLite 表结构和存储路径
│   │   ├── parsers.py       # 文档解析、OCR、音视频 ASR
│   │   └── embeddings.py    # 可选语义向量检索
│   ├── static/
│   │   ├── app-ui.html      # Web 演示页面
│   │   └── upload-ui.html
│   └── requirements.txt
├── mobile/
│   ├── App.tsx              # Expo / React Native 移动端
│   ├── package.json
│   └── app.json
├── docs/
│   └── product-plan.md
└── README.md
```

## 快速运行

### 1. 克隆仓库

```bash
git clone https://github.com/bei2024-us/knowledge-space.git
cd knowledge-space
```

### 2. 启动后端

Windows PowerShell：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

macOS / Linux：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动后可以打开：

```text
http://127.0.0.1:8000/health
```

如果返回 `{"status":"ok"}`，说明后端运行成功。

### 3. 打开 Web 演示页

后端启动后，在浏览器访问：

```text
http://127.0.0.1:8000/app-ui
```

Web 演示页可以创建空间、创建文件夹、上传资料、搜索片段、查看文档预览和词云，适合快速评审和录制 Demo。

### 4. 启动移动端

请确保手机和运行后端的电脑在同一个局域网内，并把 `YOUR_LAN_IP` 换成电脑的局域网 IP，例如 `192.168.1.222`。

Windows PowerShell：

```powershell
cd mobile
npm install
$env:EXPO_PUBLIC_API_BASE_URL="http://YOUR_LAN_IP:8000"
npm run start
```

macOS / Linux：

```bash
cd mobile
npm install
export EXPO_PUBLIC_API_BASE_URL="http://YOUR_LAN_IP:8000"
npm run start
```

然后用手机上的 Expo Go 扫描终端或浏览器里的二维码，即可在手机上打开 MindSpace。

## 可选增强能力

### 语义搜索

基础搜索无需额外模型。如果需要启用本地语义搜索，可以在后端虚拟环境中安装：

```bash
pip install FlagEmbedding
```

可选环境变量：

```bash
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=16
EMBEDDING_MAX_LENGTH=8192
```

首次运行会下载模型权重，耗时取决于网络和机器性能。未安装 FlagEmbedding 时，系统仍可使用关键词、中文分词、近义词和同页命中搜索。

### 扫描版 PDF OCR

OCR 依赖 OCRmyPDF、Tesseract、Ghostscript 和 QPDF。Ubuntu / Debian 可参考：

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim ghostscript qpdf
```

如果 OCR 组件不可用，普通文本型 PDF、Word、TXT 和 Markdown 仍可正常解析。

### 音视频转写

音视频转写使用 faster-whisper。首次上传音视频时可能需要下载模型并加载到本地，耗时可能较长。可通过环境变量调整：

```bash
ASR_MODEL=small
ASR_DEVICE=cpu
ASR_DTYPE=int8
ASR_LANG=zh
```

## 主要接口

- `GET /health`：后端健康检查。
- `GET /spaces`：获取知识空间列表。
- `POST /spaces`：创建知识空间。
- `DELETE /spaces/{space_id}`：删除知识空间。
- `GET /spaces/{space_id}/folders`：获取文件夹列表。
- `POST /spaces/{space_id}/folders`：创建文件夹。
- `POST /spaces/{space_id}/files`：上传并解析文件。
- `GET /spaces/{space_id}/documents`：获取空间内文档列表。
- `GET /documents/{document_id}`：查看文档解析片段。
- `GET /documents/{document_id}/viewer`：查看文档预览页。
- `GET /documents/{document_id}/file`：获取原始文件。
- `POST /spaces/{space_id}/search`：在空间内搜索资料片段。
- `GET /spaces/{space_id}/word-cloud`：生成空间词云。
- `GET /documents/{document_id}/word-cloud`：生成单个文档词云。

## 数据与安全

运行时数据默认保存在 `backend/data/`，包括 SQLite 数据库和上传文件。该目录已在 `.gitignore` 中排除，不会提交到仓库。

仓库已排除常见敏感和构建产物：

- `.env`、`.env.*`
- `backend/data/`
- `*.sqlite3`、`*.db`
- `mobile/node_modules/`
- `mobile/android/`、`mobile/ios/`
- `*.apk`、`*.aab`、`*.ipa`

提交前仍建议再次检查是否误放 API Key、密码、私有数据或真实用户资料。

## 当前状态

MindSpace 当前实现了从资料上传、文本解析、片段入库、空间搜索、来源定位、文档预览、词云展示到移动端展示的完整原型链路。适合用于黑客松初赛 Demo，后续可继续扩展 RAG 问答、多用户账号、云端同步和更完整的移动端发布流程。
