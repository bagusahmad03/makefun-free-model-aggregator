// The trial brief: the provider directory, the quality table and the workload
// manifest, each one JSON document inside the brief text, found by its schema.
//
// The brief is the only file a trial gives the entry besides the env itself. It
// carries no budgets and no failure schedules: those are drawn per trial and are
// never published, so routing learns them only from the providers' answers.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

/** A JSON object, the only body shape this server accepts or builds. */
export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** One provider as the directory lists it. `base` is its path under the env URL. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table: a model's public rating and capabilities. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One workload manifest row, in the order the client sends the requests. */
export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadEntry[];
  error: string | null;
}

/** Join a provider's base path to the env URL; an absolute base is used as-is. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return prefix;
  return `${prefix}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers carrying a provider's key, in the scheme the directory names. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not JSON on its own: keep scanning.
    }
  }
  return found;
}

function wireOf(value: unknown): Wire | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function authOf(value: unknown): AuthScheme {
  if (value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

function providerOf(value: unknown): ProviderEntry | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  const wire = wireOf(value.wire);
  if (id === '' || wire === null) return null;
  // The directory writes `path`; accept the absolute spellings too, path first.
  const base =
    textOf(value.path) !== ''
      ? textOf(value.path)
      : textOf(value.base_path) !== ''
        ? textOf(value.base_path)
        : textOf(value.base_url) !== ''
          ? textOf(value.base_url)
          : textOf(value.base);
  return {
    id,
    name: textOf(value.name) !== '' ? textOf(value.name) : id,
    wire,
    auth: authOf(value.auth),
    base,
    key_env: textOf(value.key_env),
    key: textOf(value.key),
  };
}

function ratingOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = textOf(value.model);
  const provider = textOf(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numberOrNull(value.rating_milli) ?? 0,
    context_length: numberOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function workloadOf(value: unknown): WorkloadEntry | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const entry: WorkloadEntry = {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
  const minContext = numberOrNull(requires.min_context);
  if (minContext !== null) entry.requires.min_context = minContext;
  return entry;
}

function emptyBrief(error: string): Brief {
  return { providers: [], ratings: [], best: new Map(), manifest: [], error };
}

/** Read the brief at `path` and split it into its three documents. */
export function readBrief(path: string): Brief {
  if (path === '') return emptyBrief('MAKEFUN_PROMPT_FILE is not set');
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return emptyBrief(`cannot read ${path}: ${(err as Error).message}`);
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directoryDoc === undefined || !Array.isArray(directoryDoc.providers)) {
    return emptyBrief(`the brief at ${path} holds no ${PROVIDERS_SCHEMA} document`);
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const parsed = providerOf(v);
    return parsed === null ? [] : [parsed];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings: RatingRow[] =
    ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const parsed = ratingOf(v);
          return parsed === null ? [] : [parsed];
        })
      : [];
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && textOf(v.id) !== '') best.set(textOf(v.id), numberOrNull(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest: WorkloadEntry[] =
    workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const parsed = workloadOf(v);
          return parsed === null ? [] : [parsed];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingRowFor(ratings: readonly RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((row) => row.provider === providerId && row.model === modelId) ?? null;
}
