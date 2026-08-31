import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { StatusBar } from 'expo-status-bar';
import React, { Component, ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  Platform,
  NativeModules,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

const API_PORT = 8000;
const API_BASE = resolveApiBase();
const APP_VERSION = 'vision-summary-v10';

function resolveApiBase() {
  const configured = getConfiguredApiBase();
  if (configured) return configured;

  if (Platform.OS === 'web') {
    return `http://127.0.0.1:${API_PORT}`;
  }

  const host = getPackagerHost();
  if (host) {
    if ((host === 'localhost' || host === '127.0.0.1') && Platform.OS === 'android') {
      return `http://10.0.2.2:${API_PORT}`;
    }
    return `http://${host}:${API_PORT}`;
  }

  return Platform.OS === 'android' ? `http://10.0.2.2:${API_PORT}` : `http://127.0.0.1:${API_PORT}`;
}

function getConfiguredApiBase() {
  const value = process.env.EXPO_PUBLIC_API_BASE_URL?.trim();
  if (!value) return null;
  return value.replace(/\/+$/, '');
}

function getPackagerHost() {
  const scriptURL = (NativeModules as { SourceCode?: { scriptURL?: string } }).SourceCode?.scriptURL;
  if (!scriptURL) return null;
  const match = String(scriptURL).match(/^(?:https?|exp):\/\/([^:/?#]+)/i);
  return match?.[1] || null;
}

type Space = {
  id: number;
  name: string;
  description: string;
  document_count: number;
  chunk_count: number;
};

type Folder = {
  id: number;
  space_id: number;
  name: string;
  document_count: number;
  created_at: string;
};

type DocumentItem = {
  id: number;
  folder_id: number | null;
  filename: string;
  file_type: string;
  chunk_count: number;
  created_at: string;
};

type PreviewChunk = {
  chunk_id: number;
  location_label: string;
  text: string;
};

type DocumentPreview = DocumentItem & {
  chunks: PreviewChunk[];
};

type SearchResult = {
  chunk_id: number;
  document_id: number;
  filename: string;
  folder_name?: string;
  location_label: string;
  snippet: string;
  text: string;
  /** PDF 片段所在页码（后端从 location_label 解析）；非 PDF 为 null */
  page_number?: number | null;
};

type AiCitation = {
  n: number;
  chunk_id: number;
  filename: string;
  location_label: string;
};

type AiSummary = {
  usable: boolean;
  answer: string;
  citations?: AiCitation[];
  message?: string;
  /** AI 实际看了几页原图（代码截图/公式页的真实内容只能从原图读到） */
  used_images?: number;
};

type CloudWord = {
  text: string;
  weight: number;
};

type ViewMode = 'home' | 'space' | 'upload';

const emptySpace: Space = {
  id: 0,
  name: '知识空间',
  description: '把 PDF、Word、笔记或音频资料放进来，按知识点检索原文片段。',
  document_count: 0,
  chunk_count: 0,
};

// 解析音视频 chunk 的 location_label，提取媒体类型与起止时间戳
// 格式：Video 00:23 -> 01:12, segment 3  或  Audio 01:05:00 -> 01:06:30, segment 12
type MediaClip = {
  kind: 'audio' | 'video';
  startSec: number;
  endSec: number;
};

function parseMediaLabel(label: string): MediaClip | null {
  const m = label.match(/^(Audio|Video)\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*->\s*(\d{1,2}:\d{2}(?::\d{2})?)/);
  if (!m) return null;
  return {
    kind: m[1].toLowerCase() as 'audio' | 'video',
    startSec: tsToSec(m[2]),
    endSec: tsToSec(m[3]),
  };
}

function tsToSec(ts: string): number {
  const parts = ts.split(':').map(Number);
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return 0;
}

function fmtSec(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

// 音视频片段播放器：Web 端用原生 HTML5 audio/video，点击"播放此片段"seek 到起止时间
function MediaClipPlayer({ documentId, clip }: { documentId: number; clip: MediaClip }) {
  const ref = useRef<any>(null);
  const [playing, setPlaying] = useState(false);

  const seekAndPlay = () => {
    const el = ref.current;
    if (!el) return;
    try {
      el.currentTime = clip.startSec;
      void el.play();
      setPlaying(true);
    } catch {
      // 某些浏览器需要用户手势，忽略
    }
  };

  const handleTimeUpdate = () => {
    const el = ref.current;
    if (!el || !playing) return;
    if (el.currentTime >= clip.endSec) {
      el.pause();
      setPlaying(false);
    }
  };

  if (Platform.OS !== 'web') {
    return (
      <Text style={{ marginTop: 8, color: '#64748b', fontSize: 12 }}>
        音视频播放仅在 Web 端可用
      </Text>
    );
  }

  const src = `${API_BASE}/documents/${documentId}/file`;
  const mediaEl = React.createElement(clip.kind, {
    ref,
    src,
    controls: true,
    preload: 'metadata',
    onTimeUpdate: handleTimeUpdate,
    style: {
      width: '100%',
      maxWidth: 640,
      marginTop: 8,
      height: clip.kind === 'video' ? 'auto' : 48,
      borderRadius: 8,
    },
  });

  return (
    <View style={{ marginTop: 8 }}>
      <Pressable
        onPress={seekAndPlay}
        style={{
          flexDirection: 'row',
          alignSelf: 'flex-start',
          paddingVertical: 6,
          paddingHorizontal: 12,
          backgroundColor: playing ? '#dcfce7' : '#eff6ff',
          borderRadius: 16,
          borderWidth: 1,
          borderColor: playing ? '#16a34a' : '#2563eb',
        }}
      >
        <Text style={{ color: playing ? '#16a34a' : '#2563eb', fontSize: 13, fontWeight: '600' }}>
          {playing ? '播放中' : '播放此片段'}（{fmtSec(clip.startSec)} - {fmtSec(clip.endSec)}）
        </Text>
      </Pressable>
      {mediaEl}
    </View>
  );
}

// 把 AI 文本里的 **加粗** 渲染成真正的加粗；其余原样返回。
function renderInlineBold(text: string, keyPrefix: string): ReactNode[] {
  const segments = text.split(/(\*\*[^*]+\*\*)/g).filter(seg => seg.length > 0);
  return segments.map((seg, i) => {
    if (seg.length >= 4 && seg.startsWith('**') && seg.endsWith('**')) {
      return (
        <Text key={`${keyPrefix}-b${i}`} style={styles.aiBold}>
          {seg.slice(2, -2)}
        </Text>
      );
    }
    return <Text key={`${keyPrefix}-t${i}`}>{seg}</Text>;
  });
}

// 去掉行首的 Markdown 记号（#、>、- * 列表、行内 ` `），列表符号换成圆点。
function cleanMarkdownLine(line: string): string {
  let s = line.replace(/^\s{0,3}#{1,6}\s*/, '');
  s = s.replace(/^\s{0,3}>\s?/, '');
  s = s.replace(/^\s{0,3}[-*+]\s+/, '• ');
  s = s.replace(/`([^`]+)`/g, '$1');
  return s;
}

// 朴素渲染 AI 回答：按行清掉 Markdown 记号，把 **加粗** 变成真正加粗，避免满屏 ** 和 #。
function AiAnswer({ text }: { text: string }) {
  const lines = text.replace(/\r/g, '').split('\n');
  return (
    <View>
      {lines.map((raw, idx) => {
        const line = cleanMarkdownLine(raw);
        if (line.trim().length === 0) {
          return <View key={`aigap-${idx}`} style={styles.aiGap} />;
        }
        return (
          <Text key={`ailn-${idx}`} style={styles.aiText}>
            {renderInlineBold(line, `ailn-${idx}`)}
          </Text>
        );
      })}
    </View>
  );
}

class AppErrorBoundary extends Component<{ children: ReactNode }, { message: string | null }> {
  state = { message: null };

  static getDerivedStateFromError(error: Error) {
    return { message: error.message || '未知错误' };
  }

  render() {
    if (this.state.message) {
      return (
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.errorPanel}>
            <Text style={styles.errorTitle}>应用遇到一个问题</Text>
            <Text style={styles.errorText}>{this.state.message}</Text>
          </View>
        </SafeAreaView>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppErrorBoundary>
        <KnowledgeSpaceApp />
      </AppErrorBoundary>
    </SafeAreaProvider>
  );
}

function KnowledgeSpaceApp() {
  const [view, setView] = useState<ViewMode>('home');
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [activeFolderId, setActiveFolderId] = useState<number | null>(null);
  const [activeSpace, setActiveSpace] = useState<Space>(emptySpace);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [preview, setPreview] = useState<DocumentPreview | null>(null);
  const [cloudWords, setCloudWords] = useState<CloudWord[]>([]);
  const [spaceName, setSpaceName] = useState('');
  const [folderName, setFolderName] = useState('');
  const [query, setQuery] = useState('');
  const [expandedTerms, setExpandedTerms] = useState<string[]>([]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [aiSummary, setAiSummary] = useState<AiSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [expandedResultId, setExpandedResultId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [apiOnline, setApiOnline] = useState(false);

  const activeStats = useMemo(
    () => [
      { label: '资料', value: activeSpace.document_count },
      { label: '片段', value: activeSpace.chunk_count },
      { label: '命中', value: results.length },
    ],
    [activeSpace, results.length],
  );

  const activeFolder = folders.find(folder => folder.id === activeFolderId) || null;
  const shownDocuments = activeFolderId
    ? documents.filter(document => document.folder_id === activeFolderId)
    : documents;

  useEffect(() => {
    loadSpaces();
  }, []);

  async function loadSpaces(preferredSpaceId?: number) {
    try {
      const response = await fetch(`${API_BASE}/spaces`);
      if (!response.ok) throw new Error('API unavailable');
      const data: Space[] = await response.json();
      setApiOnline(true);
      setSpaces(data);

      const nextActive = data.find(space => space.id === preferredSpaceId) || data[0] || emptySpace;
      setActiveSpace(nextActive);
      if (nextActive.id) {
        await loadSpaceData(nextActive.id);
      } else {
        clearSpaceData();
      }
    } catch {
      setApiOnline(false);
      clearSpaceData();
      setSpaces([]);
    }
  }

  async function loadSpaceData(spaceId: number, preferredFolderId: number | null = activeFolderId) {
    const [nextFolders] = await Promise.all([loadFolders(spaceId), loadDocuments(spaceId)]);
    if (preferredFolderId && nextFolders.some(folder => folder.id === preferredFolderId)) {
      setActiveFolderId(preferredFolderId);
    } else {
      setActiveFolderId(null);
    }
  }

  function clearSpaceData() {
    setFolders([]);
    setActiveFolderId(null);
    setDocuments([]);
    setCloudWords([]);
    setResults([]);
    setExpandedTerms([]);
  }

  async function loadFolders(spaceId: number) {
    try {
      const response = await fetch(`${API_BASE}/spaces/${spaceId}/folders`);
      if (!response.ok) throw new Error('Failed to load folders');
      const data: Folder[] = await response.json();
      setFolders(data);
      return data;
    } catch {
      setFolders([]);
      return [];
    }
  }

  async function loadDocuments(spaceId: number) {
    try {
      const response = await fetch(`${API_BASE}/spaces/${spaceId}/documents`);
      if (!response.ok) throw new Error('Failed to load documents');
      setDocuments(await response.json());
    } catch {
      setDocuments([]);
    }
  }

  async function loadDocumentWordCloud(documentId: number) {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/word-cloud`);
      if (!response.ok) throw new Error('Failed to load word cloud');
      const data = await response.json();
      setCloudWords(data.words || []);
    } catch {
      setCloudWords([]);
    }
  }

  async function createSpace() {
    const name = spaceName.trim();
    if (!name || !apiOnline) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/spaces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description: '新的资料空间' }),
      });
      if (!response.ok) throw new Error('Create failed');
      const created: Space = await response.json();
      setSpaceName('');
      setView('space');
      await loadSpaces(created.id);
    } catch {
      Alert.alert('创建失败', '电脑端服务暂时没有响应。');
    } finally {
      setLoading(false);
    }
  }

  async function createFolder() {
    const name = folderName.trim();
    if (!name || !activeSpace.id) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/spaces/${activeSpace.id}/folders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!response.ok) throw new Error('Create folder failed');
      const created: Folder = await response.json();
      setFolderName('');
      await loadSpaceData(activeSpace.id, created.id);
    } catch {
      Alert.alert('创建文件夹失败', '请确认电脑端服务正在运行。');
    } finally {
      setLoading(false);
    }
  }

  async function deleteFolder(folder: Folder) {
    Alert.alert('删除文件夹', `确定删除“${folder.name}”吗？里面的文件也会一起删除。`, [
      { text: '取消', style: 'cancel' },
      {
        text: '删除',
        style: 'destructive',
        onPress: async () => {
          setLoading(true);
          try {
            const response = await fetch(`${API_BASE}/folders/${folder.id}`, { method: 'DELETE' });
            if (!response.ok) throw new Error('Delete folder failed');
            setPreview(null);
            setResults([]);
            await loadSpaces(activeSpace.id);
          } catch {
            Alert.alert('删除失败', '文件夹没有删除成功，请确认电脑端服务正在运行。');
          } finally {
            setLoading(false);
          }
        },
      },
    ]);
  }

  async function openSpace(space: Space) {
    setActiveSpace(space);
    setPreview(null);
    setResults([]);
    setExpandedTerms([]);
    setView('space');
    await loadSpaceData(space.id, null);
  }

  async function openDocument(documentId: number) {
    setLoading(true);
    setCloudWords([]);
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}`);
      if (!response.ok) throw new Error('Preview failed');
      setPreview(await response.json());
      void loadDocumentWordCloud(documentId);
    } catch {
      Alert.alert('预览失败', '文件已经上传，但暂时无法读取预览文本。');
    } finally {
      setLoading(false);
    }
  }

  async function deleteDocument(documentId: number) {
    Alert.alert('删除文件', '确定删除这个文件和它的搜索片段吗？', [
      { text: '取消', style: 'cancel' },
      {
        text: '删除',
        style: 'destructive',
        onPress: async () => {
          setLoading(true);
          try {
            const response = await fetch(`${API_BASE}/documents/${documentId}`, { method: 'DELETE' });
            if (!response.ok) throw new Error('Delete document failed');
            setPreview(null);
            await loadSpaces(activeSpace.id);
          } catch {
            Alert.alert('删除失败', '文件没有删除成功。');
          } finally {
            setLoading(false);
          }
        },
      },
    ]);
  }

  async function openOriginalFile(documentId: number) {
    const url = `${API_BASE}/documents/${documentId}/file`;
    const supported = await Linking.canOpenURL(url);
    if (supported) {
      await Linking.openURL(url);
    } else {
      Alert.alert('无法打开原文件', '系统没有找到可以打开这个文件的应用。');
    }
  }

  /** 定位到搜索结果所在的原文页：打开阅读器并自动滚动、蓝框标出那一页。 */
  async function locateInDocument(documentId: number, pageNumber?: number | null) {
    const url = pageNumber
      ? `${API_BASE}/documents/${documentId}/viewer?focus_page=${pageNumber}`
      : `${API_BASE}/documents/${documentId}/viewer`;
    try {
      const supported = await Linking.canOpenURL(url);
      if (!supported) throw new Error('unsupported');
      await Linking.openURL(url);
    } catch {
      Alert.alert('无法打开', '系统没有找到可以打开网页的应用。');
    }
  }

  async function searchSpace() {
    const text = query.trim();
    if (!text || !activeSpace.id) return;
    setLoading(true);
    setExpandedResultId(null);
    setAiSummary(null);

    try {
      const response = await fetch(`${API_BASE}/spaces/${activeSpace.id}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, limit: 30 }),
      });
      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      setExpandedTerms(data.expanded_terms || []);
      setResults(data.results || []);
      void summarizeResults(text, data.results || []);
    } catch {
      Alert.alert('搜索失败', '没有连上电脑端服务，或文档还没有解析完成。');
    } finally {
      setLoading(false);
    }
  }

  async function summarizeResults(text: string, list: SearchResult[]) {
    if (!text || !activeSpace.id || list.length === 0) return;
    setSummaryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/spaces/${activeSpace.id}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          chunk_ids: list.slice(0, 8).map(item => item.chunk_id),
        }),
      });
      if (!response.ok) throw new Error('Summarize failed');
      const data = await response.json();
      setAiSummary(data);
    } catch {
      setAiSummary({ usable: false, answer: '', message: 'AI 总结失败，请检查电脑端服务。' });
    } finally {
      setSummaryLoading(false);
    }
  }

  async function pickAndUploadFile() {
    if (!apiOnline || !activeSpace.id) {
      Alert.alert('服务未连接', '请先保持电脑端服务运行，并进入一个知识空间。');
      return;
    }

    const picked = await DocumentPicker.getDocumentAsync({
      copyToCacheDirectory: Platform.OS !== 'web',
      multiple: false,
      type: [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown',
        // 音频
        'audio/*',
        'audio/mpeg',
        'audio/wav',
        'audio/x-wav',
        'audio/mp4',
        'audio/x-m4a',
        'audio/aac',
        'audio/flac',
        'audio/ogg',
        'audio/opus',
        'audio/webm',
        // 视频
        'video/*',
        'video/mp4',
        'video/quicktime',
        'video/x-matroska',
        'video/x-msvideo',
        'video/webm',
        'video/x-flv',
        'video/x-ms-wmv',
        'video/mpeg',
        'video/3gpp',
      ],
    });

    if (picked.canceled || picked.assets.length === 0) return;

    const asset = picked.assets[0];
    const isMedia = /\.(mp3|wav|m4a|aac|flac|ogg|opus|mp4|mov|mkv|avi|webm|flv|wmv|mpg|mpeg|3gp)$/i.test(asset.name || '');
    setLoading(true);
    setUploadStatus(isMedia ? '正在上传并转写音视频，首次加载模型需要 1-2 分钟，请耐心等待...' : '正在上传文件...');
    try {
      const uploaded = await uploadWithFallback(asset);
      setUploadStatus(`已解析 ${uploaded.chunk_count} 个片段。`);
      Alert.alert('上传完成', `已解析 ${uploaded.chunk_count} 个片段。`);
      await loadSpaces(activeSpace.id);
      setView('space');
    } catch (error) {
      const msg = error instanceof Error ? error.message : '文件没有上传成功。';
      setUploadStatus(`上传失败：${msg}`);
      Alert.alert('上传失败', msg);
    } finally {
      setLoading(false);
    }
  }

  async function uploadWithFallback(asset: DocumentPicker.DocumentPickerAsset) {
    const folderId = activeFolderId || folders[0]?.id || null;

    // Web 平台：FileSystem API 不支持 blob URL，必须用浏览器原生 FormData + fetch
    if (Platform.OS === 'web') {
      return await uploadViaWebForm(asset, folderId);
    }

    // Native 平台：保持原 base64 → multipart 回退逻辑
    const readableUri = await prepareReadableUri(asset);
    try {
      const uploadUrl = folderId
        ? `${API_BASE}/spaces/${activeSpace.id}/files?folder_id=${folderId}`
        : `${API_BASE}/spaces/${activeSpace.id}/files`;
      const response = await FileSystem.uploadAsync(uploadUrl, readableUri, {
        fieldName: 'file',
        httpMethod: 'POST',
        mimeType: asset.mimeType || inferMimeType(asset.name),
        uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      });
      const body = JSON.parse(response.body);
      if (response.status < 200 || response.status >= 300) {
        throw new Error(body.detail || 'Native upload failed');
      }
      return body;
    } catch (nativeError) {
      const contentBase64 = await FileSystem.readAsStringAsync(readableUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const response = await fetch(`${API_BASE}/spaces/${activeSpace.id}/files/base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: asset.name,
          content_base64: contentBase64,
          folder_id: folderId,
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.detail || (nativeError instanceof Error ? nativeError.message : 'Base64 upload failed'));
      }
      return body;
    }
  }

  async function uploadViaWebForm(asset: DocumentPicker.DocumentPickerAsset, folderId: number | null) {
    // Web 下 expo-document-picker 可能直接给 File，也可能只给 blob URL；两种都支持
    const anyAsset = asset as unknown as { file?: File };
    let file: File | null = anyAsset.file ?? null;
    if (!file) {
      if (!asset.uri) throw new Error('无法读取选中的文件');
      const resp = await fetch(asset.uri);
      const blob = await resp.blob();
      file = new File([blob], asset.name || 'upload', { type: asset.mimeType || inferMimeType(asset.name) });
    }
    const formData = new FormData();
    formData.append('file', file, asset.name || 'upload');

    const uploadUrl = folderId
      ? `${API_BASE}/spaces/${activeSpace.id}/files?folder_id=${folderId}`
      : `${API_BASE}/spaces/${activeSpace.id}/files`;

    const response = await fetch(uploadUrl, { method: 'POST', body: formData });
    const text = await response.text();
    let body: any = {};
    try { body = JSON.parse(text); } catch { body = { detail: text }; }
    if (!response.ok) throw new Error(body.detail || `上传失败（HTTP ${response.status}）`);
    return body;
  }

  async function prepareReadableUri(asset: DocumentPicker.DocumentPickerAsset) {
    if (!FileSystem.cacheDirectory) return asset.uri;
    const safeName = asset.name.replace(/[^\w.-]/g, '_');
    const targetUri = `${FileSystem.cacheDirectory}${Date.now()}-${safeName}`;
    try {
      await FileSystem.copyAsync({ from: asset.uri, to: targetUri });
      return targetUri;
    } catch {
      return asset.uri;
    }
  }

  function renderHome() {
    return (
      <ScrollView contentContainerStyle={styles.screen}>
        <View style={styles.header}>
          <Text style={styles.kicker}>MindSpace</Text>
          <Text style={styles.title}>把资料变成可搜索的知识空间</Text>
          <Text style={styles.subtitle}>上传 PDF、Word、笔记、音频或视频，按知识点检索原文片段，并支持近义词搜索。</Text>
        </View>

        <View style={styles.createBox}>
          <TextInput
            style={styles.input}
            placeholder="新建空间，比如：高数"
            placeholderTextColor="#9aa3af"
            value={spaceName}
            onChangeText={setSpaceName}
          />
          <Pressable style={[styles.primaryButton, !apiOnline && styles.disabledButton]} onPress={createSpace}>
            <Text style={styles.primaryButtonText}>创建</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>最近空间</Text>
        {spaces.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>{apiOnline ? '还没有空间' : '服务未连接'}</Text>
            <Text style={styles.emptyText}>{apiOnline ? '先创建一个空间，再上传资料。' : '请保持电脑端后端服务运行。'}</Text>
          </View>
        ) : (
          spaces.map(space => (
            <Pressable key={space.id} style={styles.spaceCard} onPress={() => openSpace(space)}>
              <View style={styles.spaceCopy}>
                <Text style={styles.spaceTitle}>{space.name}</Text>
                <Text style={styles.spaceDescription}>{space.description || '知识空间'}</Text>
              </View>
              <View style={styles.spaceMeta}>
                <Text style={styles.metaNumber}>{space.document_count}</Text>
                <Text style={styles.metaLabel}>files</Text>
              </View>
            </Pressable>
          ))
        )}
      </ScrollView>
    );
  }

  function renderSpace() {
    return (
      <ScrollView contentContainerStyle={styles.screen}>
        <Pressable style={styles.backButton} onPress={() => setView('home')}>
          <Text style={styles.backButtonText}>返回空间</Text>
        </Pressable>

        <View style={styles.spaceHero}>
          <Text style={styles.kicker}>Knowledge Space</Text>
          <Text style={styles.title}>{activeSpace.name}</Text>
          <Text style={styles.subtitle}>{activeSpace.description || '在多份资料里统一检索知识点。'}</Text>
          <View style={styles.statsRow}>
            {activeStats.map(item => (
              <View key={item.label} style={styles.statItem}>
                <Text style={styles.statValue}>{item.value}</Text>
                <Text style={styles.statLabel}>{item.label}</Text>
              </View>
            ))}
          </View>
        </View>

        <Text style={styles.sectionTitle}>文件夹</Text>
        <View style={styles.folderTools}>
          <TextInput
            style={styles.folderInput}
            placeholder="新建文件夹"
            placeholderTextColor="#9aa3af"
            value={folderName}
            onChangeText={setFolderName}
            onSubmitEditing={createFolder}
          />
          <Pressable style={styles.addFolderButton} onPress={createFolder}>
            <Text style={styles.addFolderText}>添加</Text>
          </Pressable>
        </View>

        <View style={styles.folderList}>
          <Pressable
            style={[styles.folderChip, activeFolderId === null && styles.folderChipActive]}
            onPress={() => setActiveFolderId(null)}
          >
            <Text style={[styles.folderChipText, activeFolderId === null && styles.folderChipTextActive]}>
              全部 {documents.length}
            </Text>
          </Pressable>
          {folders.map(folder => (
            <View key={folder.id} style={[styles.folderChip, activeFolderId === folder.id && styles.folderChipActive]}>
              <Pressable style={styles.folderMain} onPress={() => setActiveFolderId(folder.id)}>
                <Text style={[styles.folderChipText, activeFolderId === folder.id && styles.folderChipTextActive]}>
                  {folder.name} {folder.document_count}
                </Text>
              </Pressable>
              <Pressable style={styles.folderDeleteButton} onPress={() => deleteFolder(folder)}>
                <Text style={styles.folderDeleteText}>删</Text>
              </Pressable>
            </View>
          ))}
        </View>

        <View style={styles.searchBox}>
          <TextInput
            style={styles.searchInput}
            placeholder="搜索知识点、题型、概念"
            placeholderTextColor="#9aa3af"
            value={query}
            onChangeText={setQuery}
            onSubmitEditing={searchSpace}
          />
          <Pressable style={styles.searchButton} onPress={searchSpace}>
            <Text style={styles.searchButtonText}>搜索</Text>
          </Pressable>
        </View>
        {expandedTerms.length > 1 && (
          <Text style={styles.synonymText}>已扩展：{expandedTerms.slice(0, 8).join('、')}</Text>
        )}

        <View style={styles.actionRow}>
          <Pressable style={styles.secondaryButton} onPress={() => setView('upload')}>
            <Text style={styles.secondaryButtonText}>上传到{activeFolder ? `「${activeFolder.name}」` : '文件夹'}</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>{activeFolder ? `${activeFolder.name}里的资料` : '已上传资料'}</Text>
        {shownDocuments.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>还没有资料</Text>
            <Text style={styles.emptyText}>上传成功后，文件会显示在这里。</Text>
          </View>
        ) : (
          shownDocuments.map(document => (
            <View key={document.id} style={styles.documentRow}>
              <Pressable style={styles.documentOpenArea} onPress={() => openDocument(document.id)}>
                <View style={styles.fileBadge}>
                  <Text style={styles.fileBadgeText}>{document.file_type.toUpperCase()}</Text>
                </View>
                <View style={styles.documentBody}>
                  <Text style={styles.documentName}>{document.filename}</Text>
                  <Text style={styles.documentMeta}>{document.chunk_count} 个片段，点击预览</Text>
                </View>
              </Pressable>
              <Pressable style={styles.deleteFileButton} onPress={() => deleteDocument(document.id)}>
                <Text style={styles.deleteFileText}>删</Text>
              </Pressable>
            </View>
          ))
        )}

        {preview && (
          <View style={styles.previewPanel}>
            <View style={styles.previewHeader}>
              <View style={styles.previewTitleWrap}>
                <Text style={styles.previewTitle}>文档预览</Text>
                <Text style={styles.previewFile}>{preview.filename}</Text>
              </View>
              <Pressable style={styles.smallButton} onPress={() => openOriginalFile(preview.id)}>
                <Text style={styles.smallButtonText}>原文件</Text>
              </Pressable>
            </View>
            {preview.chunks.slice(0, 8).map(chunk => {
              const clip = parseMediaLabel(chunk.location_label);
              return (
                <View key={chunk.chunk_id} style={styles.previewChunk}>
                  <Text style={styles.previewLocation}>{chunk.location_label}</Text>
                  <Text style={styles.previewText}>{chunk.text}</Text>
                  {clip && <MediaClipPlayer documentId={preview.id} clip={clip} />}
                </View>
              );
            })}
            {cloudWords.length > 0 && (
              <View style={styles.docCloudBox}>
                <Text style={styles.docCloudTitle}>本资料词云</Text>
                <View style={styles.cloudBox}>
                  {cloudWords.slice(0, 30).map((word, index) => (
                    <Text
                      key={`${word.text}-${index}`}
                      style={[
                        styles.cloudWord,
                        {
                          fontSize: Math.min(24, 12 + word.weight * 1.4),
                          backgroundColor: index % 3 === 0 ? '#e0f2fe' : index % 3 === 1 ? '#ecfdf5' : '#fef3c7',
                          color: index % 3 === 0 ? '#0369a1' : index % 3 === 1 ? '#047857' : '#92400e',
                        },
                      ]}
                    >
                      {word.text}
                    </Text>
                  ))}
                </View>
              </View>
            )}
          </View>
        )}

        {loading && <ActivityIndicator color="#2563eb" style={styles.loader} />}

        <Text style={styles.sectionTitle}>搜索结果</Text>

        {(summaryLoading || aiSummary) && (
          <View style={styles.aiPanel}>
            <Text style={styles.aiTitle}>AI 整理</Text>
            {summaryLoading ? (
              <View style={styles.aiLoadingRow}>
                <ActivityIndicator color="#2563eb" />
                {/* 明确说在读原图：视觉模型逐字抄代码要几十秒，不说清会以为卡住了 */}
                <Text style={styles.aiHint}>AI 正在阅读原文页面并整理，约需 30-60 秒…</Text>
              </View>
            ) : aiSummary?.usable ? (
              <>
                <AiAnswer text={aiSummary.answer} />
                {aiSummary.used_images ? (
                  <Text style={styles.aiHint}>
                    已读取 {aiSummary.used_images} 页原图，代码与公式按原图逐字整理
                  </Text>
                ) : null}
                {aiSummary.citations && aiSummary.citations.length > 0 && (
                  <View style={styles.aiCiteBox}>
                    {aiSummary.citations.map(cite => (
                      <Text key={cite.n} style={styles.aiCite}>
                        【{cite.n}】{cite.filename}
                        {cite.location_label ? ` · ${cite.location_label}` : ''}
                      </Text>
                    ))}
                  </View>
                )}
              </>
            ) : (
              <Text style={styles.aiHint}>{aiSummary?.message || '暂无 AI 整理。'}</Text>
            )}
          </View>
        )}

        {results.length === 0 && !loading ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>还没有结果</Text>
            <Text style={styles.emptyText}>上传资料后，试试搜索原词或近义词，比如“检索”和“查找”。</Text>
          </View>
        ) : (
          results.map(result => {
            const expanded = expandedResultId === result.chunk_id;
            const hasPage = !!result.page_number;
            return (
              <Pressable
                key={result.chunk_id}
                style={styles.resultCard}
                onPress={() => setExpandedResultId(expanded ? null : result.chunk_id)}
              >
                <View style={styles.resultHeader}>
                  <Text style={styles.resultFile}>{result.filename}</Text>
                  <Text style={styles.resultLocation}>{formatResultLocation(result)}</Text>
                </View>
                <Text style={styles.resultText}>{stripMarkers(result.snippet || result.text)}</Text>
                {expanded && <Text style={styles.fullText}>{result.text}</Text>}
                {expanded && (() => {
                  const clip = parseMediaLabel(result.location_label);
                  if (clip) {
                    return <MediaClipPlayer documentId={result.document_id} clip={clip} />;
                  }
                  return null;
                })()}
                {expanded && hasPage ? (
                  <View style={styles.pageImageBox}>
                    <Text style={styles.pageImageLabel}>
                      原文第 {result.page_number} 页（公式、代码按原样显示）
                    </Text>
                    <Image
                      source={{ uri: `${API_BASE}/documents/${result.document_id}/pages/${result.page_number}.png?zoom=2.2` }}
                      style={styles.pageImage}
                      resizeMode="contain"
                    />
                  </View>
                ) : null}
                <View style={styles.resultActions}>
                  <Text style={styles.expandHint}>{expanded ? '收起片段' : '点击查看完整片段'}</Text>
                  <Pressable
                    style={styles.locateButton}
                    onPress={() => locateInDocument(result.document_id, result.page_number)}
                  >
                    <Text style={styles.locateButtonText}>
                      {hasPage ? `定位到第 ${result.page_number} 页` : '打开原文'}
                    </Text>
                  </Pressable>
                </View>
              </Pressable>
            );
          })
        )}
      </ScrollView>
    );
  }

  function renderUpload() {
    return (
      <ScrollView contentContainerStyle={styles.screen}>
        <Pressable style={styles.backButton} onPress={() => setView('space')}>
          <Text style={styles.backButtonText}>返回</Text>
        </Pressable>
        <View style={styles.uploadPanel}>
          <Text style={styles.uploadIcon}>+</Text>
          <Text style={styles.uploadTitle}>导入资料</Text>
          <Text style={styles.uploadText}>
            支持 PDF、Word、TXT、MD、音频（MP3/WAV/M4A 等）和视频（MP4/MOV/MKV 等）。音视频会自动转写为文字后入库。当前会上传到
            {activeFolder ? `「${activeFolder.name}」` : '默认文件夹'}。
          </Text>
          <Pressable style={styles.primaryButtonWide} onPress={pickAndUploadFile}>
            <Text style={styles.primaryButtonText}>选择文件</Text>
          </Pressable>
          {uploadStatus ? (
            <Text style={{ marginTop: 12, color: '#2563eb', fontSize: 13, textAlign: 'center' }}>
              {uploadStatus}
            </Text>
          ) : null}
          {loading ? (
            <ActivityIndicator color="#2563eb" style={{ marginTop: 8 }} />
          ) : null}
        </View>
      </ScrollView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <View style={styles.appShell}>
        <View style={styles.statusPill}>
          <View style={[styles.statusDot, apiOnline ? styles.dotOnline : styles.dotOffline]} />
          <Text style={styles.statusText}>
            {apiOnline ? `已连接 · ${APP_VERSION}` : `离线 · ${APP_VERSION}`}
          </Text>
        </View>
        {view === 'home' && renderHome()}
        {view === 'space' && renderSpace()}
        {view === 'upload' && renderUpload()}
      </View>
    </SafeAreaView>
  );
}

function stripMarkers(text: string) {
  return text.replace(/\[/g, '').replace(/\]/g, '');
}

function formatResultLocation(result: SearchResult) {
  return [result.folder_name, result.location_label].filter(Boolean).join(' · ');
}

function inferMimeType(fileName: string) {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.pdf')) return 'application/pdf';
  if (lower.endsWith('.docx')) return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
  if (lower.endsWith('.md')) return 'text/markdown';
  // 音频
  if (lower.endsWith('.mp3')) return 'audio/mpeg';
  if (lower.endsWith('.wav')) return 'audio/wav';
  if (lower.endsWith('.m4a')) return 'audio/mp4';
  if (lower.endsWith('.aac')) return 'audio/aac';
  if (lower.endsWith('.flac')) return 'audio/flac';
  if (lower.endsWith('.ogg')) return 'audio/ogg';
  if (lower.endsWith('.opus')) return 'audio/opus';
  // 视频
  if (lower.endsWith('.mp4')) return 'video/mp4';
  if (lower.endsWith('.mov')) return 'video/quicktime';
  if (lower.endsWith('.mkv')) return 'video/x-matroska';
  if (lower.endsWith('.avi')) return 'video/x-msvideo';
  if (lower.endsWith('.webm')) return 'video/webm';
  if (lower.endsWith('.flv')) return 'video/x-flv';
  if (lower.endsWith('.wmv')) return 'video/x-ms-wmv';
  if (lower.endsWith('.mpg') || lower.endsWith('.mpeg')) return 'video/mpeg';
  if (lower.endsWith('.3gp')) return 'video/3gpp';
  return 'text/plain';
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f6f8fb',
  },
  appShell: {
    flex: 1,
    maxWidth: 460,
    width: '100%',
    alignSelf: 'center',
    backgroundColor: '#f6f8fb',
  },
  errorPanel: {
    flex: 1,
    margin: 20,
    padding: 22,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#fecaca',
  },
  errorTitle: {
    color: '#991b1b',
    fontSize: 20,
    fontWeight: '900',
    marginBottom: 10,
  },
  errorText: {
    color: '#475569',
    fontSize: 14,
    lineHeight: 22,
  },
  screen: {
    padding: 20,
    paddingTop: Platform.OS === 'android' ? 42 : 20,
    paddingBottom: 42,
  },
  statusPill: {
    position: 'absolute',
    zIndex: 2,
    top: Platform.OS === 'android' ? 12 : 8,
    right: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.9)',
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 999,
  },
  dotOnline: {
    backgroundColor: '#16a34a',
  },
  dotOffline: {
    backgroundColor: '#f59e0b',
  },
  statusText: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '600',
  },
  header: {
    marginTop: 24,
    marginBottom: 22,
  },
  kicker: {
    color: '#2563eb',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 8,
  },
  title: {
    color: '#111827',
    fontSize: 30,
    lineHeight: 37,
    fontWeight: '800',
    marginBottom: 10,
  },
  subtitle: {
    color: '#64748b',
    fontSize: 15,
    lineHeight: 23,
  },
  createBox: {
    flexDirection: 'row',
    gap: 10,
    padding: 8,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    shadowColor: '#0f172a',
    shadowOpacity: 0.08,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
    elevation: 3,
  },
  input: {
    flex: 1,
    minHeight: 48,
    paddingHorizontal: 14,
    color: '#111827',
    fontSize: 15,
  },
  primaryButton: {
    minWidth: 72,
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
    backgroundColor: '#2563eb',
  },
  primaryButtonWide: {
    width: '100%',
    minHeight: 54,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
    backgroundColor: '#2563eb',
  },
  disabledButton: {
    backgroundColor: '#94a3b8',
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '800',
  },
  sectionTitle: {
    color: '#111827',
    fontSize: 18,
    fontWeight: '800',
    marginTop: 26,
    marginBottom: 12,
  },
  spaceCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    padding: 18,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
  },
  spaceCopy: {
    flex: 1,
    paddingRight: 12,
  },
  spaceTitle: {
    color: '#111827',
    fontSize: 17,
    fontWeight: '800',
    marginBottom: 6,
  },
  spaceDescription: {
    color: '#64748b',
    fontSize: 13,
    lineHeight: 19,
  },
  spaceMeta: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 58,
    height: 58,
    borderRadius: 20,
    backgroundColor: '#eef6ff',
  },
  metaNumber: {
    color: '#2563eb',
    fontSize: 18,
    fontWeight: '800',
  },
  metaLabel: {
    color: '#64748b',
    fontSize: 11,
    fontWeight: '700',
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingVertical: 8,
    marginBottom: 8,
  },
  backButtonText: {
    color: '#2563eb',
    fontSize: 17,
    fontWeight: '700',
  },
  spaceHero: {
    padding: 22,
    borderRadius: 28,
    backgroundColor: '#ffffff',
    shadowColor: '#0f172a',
    shadowOpacity: 0.07,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 12 },
    elevation: 3,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 18,
  },
  statItem: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 18,
    backgroundColor: '#f8fafc',
    alignItems: 'center',
  },
  statValue: {
    color: '#111827',
    fontSize: 20,
    fontWeight: '800',
  },
  statLabel: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 4,
  },
  folderTools: {
    flexDirection: 'row',
    gap: 10,
    padding: 8,
    borderRadius: 22,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
  },
  folderInput: {
    flex: 1,
    minHeight: 44,
    paddingHorizontal: 12,
    color: '#111827',
    fontSize: 14,
  },
  addFolderButton: {
    minWidth: 66,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 16,
    backgroundColor: '#111827',
  },
  addFolderText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '800',
  },
  folderList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 10,
  },
  folderChip: {
    minHeight: 38,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#dbe7f3',
    backgroundColor: '#ffffff',
    overflow: 'hidden',
  },
  folderChipActive: {
    borderColor: '#2563eb',
    backgroundColor: '#eef6ff',
  },
  folderMain: {
    minHeight: 38,
    justifyContent: 'center',
    paddingLeft: 12,
    paddingRight: 8,
  },
  folderChipText: {
    color: '#475569',
    fontSize: 13,
    fontWeight: '800',
  },
  folderChipTextActive: {
    color: '#2563eb',
  },
  folderDeleteButton: {
    minWidth: 36,
    minHeight: 38,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff1f2',
  },
  folderDeleteText: {
    color: '#e11d48',
    fontSize: 12,
    fontWeight: '900',
  },
  searchBox: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 18,
    padding: 8,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
  },
  searchInput: {
    flex: 1,
    minHeight: 48,
    paddingHorizontal: 14,
    color: '#111827',
    fontSize: 15,
  },
  searchButton: {
    minWidth: 76,
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
    backgroundColor: '#111827',
  },
  searchButtonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '800',
  },
  synonymText: {
    marginTop: 8,
    color: '#64748b',
    fontSize: 12,
    lineHeight: 18,
  },
  actionRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 12,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 46,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
    backgroundColor: '#eef2ff',
    paddingHorizontal: 8,
  },
  secondaryButtonText: {
    color: '#2563eb',
    fontSize: 14,
    fontWeight: '800',
    textAlign: 'center',
  },
  cloudBox: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    padding: 14,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
  },
  cloudWord: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 14,
    overflow: 'hidden',
    fontWeight: '800',
  },
  docCloudBox: {
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eef2f7',
  },
  docCloudTitle: {
    color: '#2563eb',
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 8,
  },
  loader: {
    marginTop: 20,
  },
  emptyState: {
    padding: 22,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
  },
  emptyTitle: {
    color: '#111827',
    fontSize: 16,
    fontWeight: '800',
  },
  emptyText: {
    color: '#64748b',
    fontSize: 14,
    lineHeight: 21,
    marginTop: 6,
  },
  documentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 20,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
    marginBottom: 10,
  },
  documentOpenArea: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  fileBadge: {
    width: 56,
    height: 46,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#eef6ff',
  },
  fileBadgeText: {
    color: '#2563eb',
    fontSize: 11,
    fontWeight: '900',
  },
  documentBody: {
    flex: 1,
  },
  documentName: {
    color: '#111827',
    fontSize: 14,
    fontWeight: '800',
  },
  documentMeta: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 4,
  },
  deleteFileButton: {
    minWidth: 42,
    minHeight: 42,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 15,
    backgroundColor: '#fff1f2',
  },
  deleteFileText: {
    color: '#e11d48',
    fontSize: 12,
    fontWeight: '900',
  },
  previewPanel: {
    marginTop: 16,
    padding: 16,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#dbeafe',
  },
  previewHeader: {
    flexDirection: 'row',
    gap: 12,
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  previewTitleWrap: {
    flex: 1,
  },
  previewTitle: {
    color: '#111827',
    fontSize: 17,
    fontWeight: '900',
  },
  previewFile: {
    color: '#64748b',
    fontSize: 12,
    marginTop: 4,
  },
  smallButton: {
    minHeight: 36,
    paddingHorizontal: 14,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111827',
  },
  smallButtonText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '800',
  },
  previewChunk: {
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#eef2f7',
  },
  previewLocation: {
    color: '#2563eb',
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 6,
  },
  previewText: {
    color: '#1f2937',
    fontSize: 14,
    lineHeight: 22,
  },
  aiPanel: {
    marginTop: 4,
    marginBottom: 8,
    padding: 16,
    borderRadius: 20,
    backgroundColor: '#f8fbff',
    borderWidth: 1,
    borderColor: '#dbeafe',
  },
  aiTitle: {
    color: '#1d4ed8',
    fontSize: 15,
    fontWeight: '800',
    marginBottom: 10,
  },
  aiLoadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  aiText: {
    color: '#1f2937',
    fontSize: 14,
    lineHeight: 22,
  },
  aiBold: {
    fontWeight: '800',
    color: '#111827',
  },
  aiGap: {
    height: 8,
  },
  aiCiteBox: {
    marginTop: 10,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    gap: 2,
  },
  aiCite: {
    color: '#64748b',
    fontSize: 12,
    lineHeight: 18,
  },
  aiHint: {
    color: '#64748b',
    fontSize: 13,
    lineHeight: 20,
  },
  resultCard: {
    padding: 16,
    borderRadius: 22,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
    marginBottom: 12,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
    marginBottom: 10,
  },
  resultFile: {
    flex: 1,
    color: '#2563eb',
    fontSize: 13,
    fontWeight: '800',
  },
  resultLocation: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '700',
  },
  resultText: {
    color: '#1f2937',
    fontSize: 15,
    lineHeight: 23,
  },
  fullText: {
    color: '#475569',
    fontSize: 14,
    lineHeight: 22,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eef2f7',
  },
  expandHint: {
    color: '#2563eb',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 10,
  },
  resultActions: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  locateButton: {
    marginTop: 10,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    backgroundColor: '#eff6ff',
    borderWidth: 1,
    borderColor: '#bfdbfe',
  },
  locateButtonText: {
    color: '#1d4ed8',
    fontSize: 12,
    fontWeight: '800',
  },
  pageImageBox: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eef2f7',
  },
  pageImageLabel: {
    color: '#64748b',
    fontSize: 12,
    marginBottom: 8,
  },
  pageImage: {
    width: '100%',
    aspectRatio: 4 / 3,
    borderRadius: 12,
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e5edf6',
  },
  uploadPanel: {
    minHeight: 420,
    padding: 24,
    borderRadius: 30,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#edf2f7',
  },
  uploadIcon: {
    width: 78,
    height: 78,
    borderRadius: 28,
    textAlign: 'center',
    textAlignVertical: 'center',
    overflow: 'hidden',
    color: '#2563eb',
    backgroundColor: '#eef6ff',
    fontSize: 44,
    lineHeight: 74,
    fontWeight: '300',
    marginBottom: 20,
  },
  uploadTitle: {
    color: '#111827',
    fontSize: 25,
    fontWeight: '800',
    marginBottom: 10,
  },
  uploadText: {
    color: '#64748b',
    fontSize: 15,
    lineHeight: 23,
    textAlign: 'center',
    marginBottom: 24,
  },
});
