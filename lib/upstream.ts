// One call to one provider, in that provider's own wire.
//
// The upstream response is always read whole (JSON or SSE) and parsed into a
// canonical Answer before the client is answered, so a failure can still fail
// over. A non-2xx answer, an unreachable provider, a timeout and an unparseable
// answer are all failures: the caller tries the next candidate.

import { authHeaders, isRecord, joinBase, type Json, type ProviderEntry } from './brief.ts';
import {
  normalizeRequest,
  parseAnthropicResponse,
  parseGeminiResponse,
  parseOpenAIResponse,
  renderAnthropic,
  renderGemini,
  renderOpenAI,
  type Answer,
} from './translate.ts';

export type UpstreamOutcome =
  | { ok: true; answer: Answer }
  | { ok: false; status: number; code: string; message: string };

/** How long one provider call may take before it counts as a failure. */
const CALL_TIMEOUT_MS = 10000;

/** The error code of a refused call, read from the wire's own envelope. */
function errorCode(wire: ProviderEntry['wire'], status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value)) return `http_${status}`;
    if (wire === 'gemini' && isRecord(value.error) && typeof value.error.status === 'string') return value.error.status;
    if (isRecord(value.error) && typeof value.error.code === 'string' && value.error.code !== '') return value.error.code;
    if (isRecord(value.error) && typeof value.error.type === 'string') return value.error.type;
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/**
 * Call one provider with one request. `clientWire` is the wire the client used;
 * the request is rendered in the provider's wire, preserving the fields the
 * provider echoes (system, user text, images, tools, stops, cap, JSON mode).
 */
export async function callUpstream(
  envUrl: string,
  provider: ProviderEntry,
  modelId: string,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  timeoutMs: number = CALL_TIMEOUT_MS,
): Promise<UpstreamOutcome> {
  const norm = normalizeRequest(clientWire, body);
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = renderOpenAI(norm, modelId);
    if (norm.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = renderAnthropic(norm, modelId);
    if (norm.stream) headers.accept = 'text/event-stream';
  } else {
    const action = norm.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:${action}${norm.stream ? '?alt=sse' : ''}`;
    payload = renderGemini(norm);
    if (norm.stream) headers.accept = 'text/event-stream';
  }

  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', message: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) {
    return { ok: false, status: res.status, code: errorCode(provider.wire, res.status, text), message: text.slice(0, 300) };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIResponse(text, norm.stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicResponse(text, norm.stream)
        : parseGeminiResponse(text, norm.stream);
  if (answer === null) {
    return { ok: false, status: 502, code: 'bad_upstream', message: 'the provider answered no parseable answer' };
  }
  return { ok: true, answer };
}
