import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { StatusBar } from 'expo-status-bar';
import React, { Component, ReactNode, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const API_BASE = 'http://192.168.1.222:8000';
const APP_VERSION = 'preview-search-cloud-v6';

type Space = {
  id: number;
  name: string;
  description: string;
  document_count: number;
  chunk_count: number;
};

type DocumentItem = {
  id: number;
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
  filename: string;
  location_label: string;
  snippet: string;
  text: string;
};

type CloudWord = {
  text: string;
  weight: number;
};

type ViewMode = 'home' | 'space' | 'upload';

const emptySpace: Space = {
  id: 0,
  name: '知识空间',
  description: '把 PDF、Word 和笔记放进来，按知识点检索原文片段。',
  document_count: 0,
  chunk_count: 0,
};

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
    <AppErrorBoundary>
      <KnowledgeSpaceApp />
    </AppErrorBoundary>
  );
}

function KnowledgeSpaceApp() {
  const [view, setView] = useState<ViewMode>('home');
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [activeSpace, setActiveSpace] = useState<Space>(emptySpace);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [preview, setPreview] = useState<DocumentPreview | null>(null);
  const [cloudWords, setCloudWords] = useState<CloudWord[]>([]);
  const [spaceName, setSpaceName] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [expandedResultId, setExpandedResultId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);

  const activeStats = useMemo(
    () => [
      { label: '资料', value: activeSpace.document_count },
      { label: '片段', value: activeSpace.chunk_count },
      { label: '命中', value: results.length },
    ],
    [activeSpace, results.length],
  );

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
        await Promise.all([loadDocuments(nextActive.id), loadWordCloud(nextActive.id)]);
      }
    } catch {
      setApiOnline(false);
      setSpaces([]);
      setDocuments([]);
      setCloudWords([]);
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

  async function loadWordCloud(spaceId: number) {
    try {
      const response = await fetch(`${API_BASE}/spaces/${spaceId}/word-cloud`);
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

  async function openSpace(space: Space) {
    setActiveSpace(space);
    setPreview(null);
    setResults([]);
    setView('space');
    await Promise.all([loadDocuments(space.id), loadWordCloud(space.id)]);
  }

  async function openDocument(documentId: number) {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}`);
      if (!response.ok) throw new Error('Preview failed');
      setPreview(await response.json());
    } catch {
      Alert.alert('预览失败', '文档已经上传，但暂时无法读取预览文本。');
    } finally {
      setLoading(false);
    }
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

  async function searchSpace() {
    const text = query.trim();
    if (!text || !activeSpace.id) return;
    setLoading(true);
    setExpandedResultId(null);

    try {
      const response = await fetch(`${API_BASE}/spaces/${activeSpace.id}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, limit: 30 }),
      });
      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      setResults(data.results || []);
    } catch {
      Alert.alert('搜索失败', '没有连上电脑端服务，或文档还没有解析完成。');
    } finally {
      setLoading(false);
    }
  }

  async function pickAndUploadFile() {
    if (!apiOnline || !activeSpace.id) {
      Alert.alert('服务未连接', '请先保持电脑端服务运行，并进入一个知识空间。');
      return;
    }

    const picked = await DocumentPicker.getDocumentAsync({
      copyToCacheDirectory: true,
      multiple: false,
      type: [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown',
      ],
    });

    if (picked.canceled || picked.assets.length === 0) return;

    setLoading(true);
    try {
      const uploaded = await uploadWithFallback(picked.assets[0]);
      Alert.alert('上传完成', `已解析 ${uploaded.chunk_count} 个片段。`);
      await loadSpaces(activeSpace.id);
      setView('space');
    } catch (error) {
      Alert.alert('上传失败', error instanceof Error ? error.message : '文件没有上传成功。');
    } finally {
      setLoading(false);
    }
  }

  async function uploadWithFallback(asset: DocumentPicker.DocumentPickerAsset) {
    const readableUri = await prepareReadableUri(asset);
    try {
      const contentBase64 = await FileSystem.readAsStringAsync(readableUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const response = await fetch(`${API_BASE}/spaces/${activeSpace.id}/files/base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: asset.name,
          content_base64: contentBase64,
        }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Base64 upload failed');
      return body;
    } catch (base64Error) {
      const response = await FileSystem.uploadAsync(`${API_BASE}/spaces/${activeSpace.id}/files`, readableUri, {
        fieldName: 'file',
        httpMethod: 'POST',
        mimeType: asset.mimeType || inferMimeType(asset.name),
        uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      });
      const body = JSON.parse(response.body);
      if (response.status < 200 || response.status >= 300) {
        const message = body.detail || (base64Error instanceof Error ? base64Error.message : 'Native upload failed');
        throw new Error(message);
      }
      return body;
    }
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
          <Text style={styles.subtitle}>上传 PDF、Word 或笔记，直接检索知识点、查看原文片段和资料词云。</Text>
        </View>

        <View style={styles.createBox}>
          <TextInput
            style={styles.input}
            placeholder="新建空间，比如 高数"
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

        <View style={styles.actionRow}>
          <Pressable style={styles.secondaryButton} onPress={() => setView('upload')}>
            <Text style={styles.secondaryButtonText}>上传资料</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => loadWordCloud(activeSpace.id)}>
            <Text style={styles.secondaryButtonText}>刷新词云</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>资料词云</Text>
        {cloudWords.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>暂无词云</Text>
            <Text style={styles.emptyText}>上传资料并解析后，这里会显示高频概念。</Text>
          </View>
        ) : (
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
        )}

        <Text style={styles.sectionTitle}>已上传资料</Text>
        {documents.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>还没有资料</Text>
            <Text style={styles.emptyText}>上传成功后，文件会显示在这里。</Text>
          </View>
        ) : (
          documents.map(document => (
            <Pressable key={document.id} style={styles.documentRow} onPress={() => openDocument(document.id)}>
              <View style={styles.fileBadge}>
                <Text style={styles.fileBadgeText}>{document.file_type.toUpperCase()}</Text>
              </View>
              <View style={styles.documentBody}>
                <Text style={styles.documentName}>{document.filename}</Text>
                <Text style={styles.documentMeta}>{document.chunk_count} 个片段，点击预览</Text>
              </View>
            </Pressable>
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
            {preview.chunks.slice(0, 8).map(chunk => (
              <View key={chunk.chunk_id} style={styles.previewChunk}>
                <Text style={styles.previewLocation}>{chunk.location_label}</Text>
                <Text style={styles.previewText}>{chunk.text}</Text>
              </View>
            ))}
          </View>
        )}

        {loading && <ActivityIndicator color="#2563eb" style={styles.loader} />}

        <Text style={styles.sectionTitle}>搜索结果</Text>
        {results.length === 0 && !loading ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>还没有结果</Text>
            <Text style={styles.emptyText}>上传资料后，试试搜索文档里的原词，比如章节标题或公式名称。</Text>
          </View>
        ) : (
          results.map(result => {
            const expanded = expandedResultId === result.chunk_id;
            return (
              <Pressable
                key={result.chunk_id}
                style={styles.resultCard}
                onPress={() => setExpandedResultId(expanded ? null : result.chunk_id)}
              >
                <View style={styles.resultHeader}>
                  <Text style={styles.resultFile}>{result.filename}</Text>
                  <Text style={styles.resultLocation}>{result.location_label}</Text>
                </View>
                <Text style={styles.resultText}>{stripMarkers(result.snippet || result.text)}</Text>
                {expanded && <Text style={styles.fullText}>{result.text}</Text>}
                <Text style={styles.expandHint}>{expanded ? '收起片段' : '点击查看完整片段'}</Text>
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
          <Text style={styles.uploadText}>支持 PDF、Word、TXT、MD。上传后会自动解析成片段，并加入搜索和词云。</Text>
          <Pressable style={styles.primaryButtonWide} onPress={pickAndUploadFile}>
            <Text style={styles.primaryButtonText}>选择文件</Text>
          </Pressable>
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

function inferMimeType(fileName: string) {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.pdf')) return 'application/pdf';
  if (lower.endsWith('.docx')) return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
  if (lower.endsWith('.md')) return 'text/markdown';
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
    backgroundColor: 'rgba(255,255,255,0.88)',
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
  },
  secondaryButtonText: {
    color: '#2563eb',
    fontSize: 14,
    fontWeight: '800',
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
    gap: 12,
    padding: 14,
    borderRadius: 20,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#edf2f7',
    marginBottom: 10,
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
