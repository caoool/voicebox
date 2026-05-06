import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Mic2,
  Play,
  Trash2,
  Upload,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api/client';
import type { VoiceConversionResponse } from '@/lib/api/types';
import { ALL_LANGUAGES, ENGINE_LANGUAGES } from '@/lib/constants/languages';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { usePlayerStore } from '@/stores/playerStore';
import { useServerStore } from '@/stores/serverStore';
import { cn } from '@/lib/utils/cn';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';

// ── constants ─────────────────────────────────────────────────────────────

const ENGINES = [
  { value: 'qwen', label: 'Qwen TTS' },
  { value: 'qwen_custom_voice', label: 'Qwen CustomVoice' },
  { value: 'luxtts', label: 'LuxTTS' },
  { value: 'chatterbox', label: 'Chatterbox' },
  { value: 'chatterbox_turbo', label: 'Chatterbox Turbo' },
  { value: 'tada', label: 'TADA' },
  { value: 'kokoro', label: 'Kokoro' },
];

const MODEL_SIZES = ['0.6B', '1.7B', '1B', '3B'];

const STT_MODELS = ['base', 'small', 'medium', 'large', 'turbo'];

// ── helpers ───────────────────────────────────────────────────────────────

function formatDuration(seconds?: number | null): string {
  if (!seconds) return '--';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString();
}

// ── sub-components ────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: VoiceConversionResponse['status'] }) {
  if (status === 'generating')
    return (
      <Badge variant="secondary" className="gap-1">
        <Loader2 className="h-3 w-3 animate-spin" />
        Converting
      </Badge>
    );
  if (status === 'completed')
    return (
      <Badge className="gap-1 bg-green-600/20 text-green-400 border-green-600/30">
        <CheckCircle2 className="h-3 w-3" />
        Done
      </Badge>
    );
  return (
    <Badge variant="destructive" className="gap-1">
      <AlertCircle className="h-3 w-3" />
      Failed
    </Badge>
  );
}

// ── polling hook ──────────────────────────────────────────────────────────

function useConversionStatus(
  conversionId: string | null,
  onDone: (item: VoiceConversionResponse) => void,
) {
  const queryClient = useQueryClient();
  const serverUrl = useServerStore((s) => s.serverUrl);

  useEffect(() => {
    if (!conversionId) return;

    const eventSource = new EventSource(`${serverUrl}/voice-convert/${conversionId}/status`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as VoiceConversionResponse;
        if (data.status === 'completed' || data.status === 'failed') {
          eventSource.close();
          queryClient.invalidateQueries({ queryKey: ['voice-conversions'] });
          onDone(data);
        }
      } catch {
        // ignore parse errors
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      queryClient.invalidateQueries({ queryKey: ['voice-conversions'] });
    };

    return () => eventSource.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversionId, serverUrl]);
}

// ── main component ────────────────────────────────────────────────────────

export function VoiceConvertTab() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: profiles } = useProfiles();
  const setAudioUrl = usePlayerStore((s) => s.setAudioUrl);
  const serverUrl = useServerStore((s) => s.serverUrl);

  // Form state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [profileId, setProfileId] = useState<string>('');
  const [engine, setEngine] = useState<string>('qwen');
  const [modelSize, setModelSize] = useState<string>('1.7B');
  const [language, setLanguage] = useState<string>('en');
  const [sttModel, setSttModel] = useState<string>('turbo');
  const [activeConversionId, setActiveConversionId] = useState<string | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Language options for the chosen engine
  const supportedLanguages = ENGINE_LANGUAGES[engine] ?? (['en'] as const);

  // Reset language when engine changes if current lang not supported
  useEffect(() => {
    if (!supportedLanguages.includes(language as never)) {
      setLanguage('en');
    }
  }, [engine, supportedLanguages, language]);

  // Conversion history
  const { data: conversions, isLoading } = useQuery({
    queryKey: ['voice-conversions'],
    queryFn: () => apiClient.listVoiceConversions({ limit: 50 }),
    refetchInterval: activeConversionId ? 3000 : false,
  });

  // Start conversion mutation
  const startMutation = useMutation({
    mutationFn: () =>
      apiClient.startVoiceConvert({
        file: selectedFile!,
        profileId,
        engine,
        modelSize,
        language,
        sttModel,
      }),
    onSuccess: (data) => {
      setActiveConversionId(data.id);
      setLastError(null);
      queryClient.invalidateQueries({ queryKey: ['voice-conversions'] });
    },
    onError: (err: Error) => {
      setLastError(err.message);
    },
  });

  // Delete conversion mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteVoiceConversion(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['voice-conversions'] });
    },
  });

  // Poll SSE for active conversion
  useConversionStatus(activeConversionId, (item) => {
    setActiveConversionId(null);
    if (item.status === 'failed') {
      setLastError(item.error ?? 'Conversion failed');
    }
  });

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setSelectedFile(f);
    setLastError(null);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const f = e.dataTransfer.files[0] ?? null;
    if (f) {
      setSelectedFile(f);
      setLastError(null);
    }
  }

  function handlePlayback(item: VoiceConversionResponse) {
    if (!item.audio_path) return;
    const url = `${serverUrl}/audio/${encodeURIComponent(item.audio_path)}`;
    setAudioUrl(url);
  }

  const canSubmit =
    !!selectedFile && !!profileId && !startMutation.isPending && !activeConversionId;

  return (
    <div
      className="flex flex-col h-full gap-6 py-6"
      style={{ paddingBottom: BOTTOM_SAFE_AREA_PADDING }}
    >
      <div className="flex items-center gap-3">
        <Mic2 className="h-6 w-6 text-accent" />
        <div>
          <h1 className="text-xl font-semibold">{t('voiceConvert.title', 'Voice Convert')}</h1>
          <p className="text-sm text-muted-foreground">
            {t(
              'voiceConvert.subtitle',
              'Convert source audio to a target voice profile (STT → TTS pipeline).',
            )}
          </p>
        </div>
      </div>

      {/* ── conversion form ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Source audio */}
        <div className="space-y-2">
          <Label>{t('voiceConvert.sourceAudio', 'Source audio')}</Label>
          <div
            className={cn(
              'border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors',
              selectedFile
                ? 'border-accent/60 bg-accent/5'
                : 'border-border hover:border-accent/40 hover:bg-muted/30',
            )}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            <Upload className="h-8 w-8 text-muted-foreground" />
            {selectedFile ? (
              <div className="text-center">
                <p className="text-sm font-medium">{selectedFile.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(0)} KB
                </p>
              </div>
            ) : (
              <div className="text-center">
                <p className="text-sm font-medium">
                  {t('voiceConvert.dropAudio', 'Drop audio here or click to upload')}
                </p>
                <p className="text-xs text-muted-foreground">WAV, MP3, FLAC, OGG, M4A, WebM</p>
              </div>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.webm"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        {/* Options */}
        <div className="space-y-4">
          {/* Target profile */}
          <div className="space-y-1.5">
            <Label>{t('voiceConvert.targetProfile', 'Target voice profile')}</Label>
            <Select value={profileId} onValueChange={setProfileId}>
              <SelectTrigger>
                <SelectValue placeholder={t('voiceConvert.selectProfile', 'Select a profile…')} />
              </SelectTrigger>
              <SelectContent>
                {(profiles ?? []).map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Engine */}
          <div className="space-y-1.5">
            <Label>{t('voiceConvert.engine', 'TTS Engine')}</Label>
            <Select value={engine} onValueChange={setEngine}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ENGINES.map((e) => (
                  <SelectItem key={e.value} value={e.value}>
                    {e.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Model size (only for qwen-based engines) */}
          {(engine === 'qwen' || engine === 'qwen_custom_voice') && (
            <div className="space-y-1.5">
              <Label>{t('voiceConvert.modelSize', 'Model size')}</Label>
              <Select value={modelSize} onValueChange={setModelSize}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODEL_SIZES.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Language */}
          <div className="space-y-1.5">
            <Label>{t('voiceConvert.language', 'Language')}</Label>
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(supportedLanguages as readonly string[]).map((code) => (
                  <SelectItem key={code} value={code}>
                    {ALL_LANGUAGES[code as keyof typeof ALL_LANGUAGES] ?? code}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* STT model */}
          <div className="space-y-1.5">
            <Label>{t('voiceConvert.sttModel', 'Whisper model (transcription)')}</Label>
            <Select value={sttModel} onValueChange={setSttModel}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {STT_MODELS.map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Error */}
      {lastError && (
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded-lg px-4 py-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{lastError}</span>
        </div>
      )}

      {/* Submit */}
      <div className="flex justify-end">
        <Button
          onClick={() => startMutation.mutate()}
          disabled={!canSubmit}
          className="gap-2"
        >
          {startMutation.isPending || activeConversionId ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              {t('voiceConvert.converting', 'Converting…')}
            </>
          ) : (
            t('voiceConvert.convert', 'Convert')
          )}
        </Button>
      </div>

      {/* ── history ─────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0 flex flex-col gap-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          {t('voiceConvert.history', 'Recent conversions')}
        </h2>

        {isLoading ? (
          <div className="flex items-center justify-center py-12 text-muted-foreground gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{t('common.loading', 'Loading…')}</span>
          </div>
        ) : !conversions?.items.length ? (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground gap-2">
            <Mic2 className="h-8 w-8 opacity-30" />
            <p className="text-sm">{t('voiceConvert.empty', 'No conversions yet')}</p>
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-y-auto">
            <div className="space-y-2 pr-2">
              {conversions.items.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center gap-3 rounded-lg border border-border bg-card/50 px-4 py-3"
                >
                  {/* Status */}
                  <StatusBadge status={item.status} />

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {item.transcript
                        ? `"${item.transcript.slice(0, 60)}${item.transcript.length > 60 ? '…' : ''}"`
                        : t('voiceConvert.noTranscript', 'Transcribing…')}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatDate(item.created_at)} · {item.engine ?? '?'} ·{' '}
                      {formatDuration(item.duration)}
                    </p>
                    {item.error && (
                      <p className="text-xs text-destructive mt-0.5">{item.error}</p>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 shrink-0">
                    {item.status === 'completed' && item.audio_path && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        title={t('common.play', 'Play')}
                        onClick={() => handlePlayback(item)}
                      >
                        <Play className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      title={t('common.delete', 'Delete')}
                      onClick={() => deleteMutation.mutate(item.id)}
                      disabled={deleteMutation.isPending}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
