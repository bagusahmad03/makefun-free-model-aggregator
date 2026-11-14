// The catalog probe and the one catalog document.
//
// Every provider's catalog route is probed in its own wire, one at a time and in
// the directory's order, so the probe sequence is the same on every run. A probe
// that fails leaves its provider out of the catalog and gets no traffic.
//
// The catalog document is built once and used twice: `--list` prints it and the
// register body carries it, the same object both times. Its base_url is the
// directory's own path (never joined to the env URL): the env's origin differs
// between a live pass and its replay, and the catalog is part of the state export
// a replay must reproduce byte for byte.

import {
  authHeaders,
  isRecord,
  joinBase,
  ratingRowFor,
  type Json,
  type ProviderEntry,
  type RatingRow,
  type Wire,
} from './brief.ts';

/** One model as a provider's catalog route listed it. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe result. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** What a model can do, the quality table first and the listing second. */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: what `--list` prints and what /register carries. */
export interface Catalog {
  providers: CatalogProvider[];
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says. A
 * paid model is never routed.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** A model's capabilities: the quality table's row first, the listing second. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row !== null ? row.tools : (model.tools ?? null),
    vision: row !== null ? row.vision : (model.vision ?? null),
    json: row !== null ? row.json : (model.json ?? null),
  };
}

/** Whether capabilities meet a request's requirements; unknown never meets one. */
export function meetsRequirements(
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number },
  caps: Capabilities,
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** A listing's price block: free when every price is zero, unknown when silent. */
function zeroPriced(value: Json): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const numbers = Object.values(pricing).flatMap((entry) => {
    if (typeof entry === 'number') return [entry];
    if (typeof entry === 'string' && entry.trim() !== '') return [Number(entry)];
    return [];
  });
  if (numbers.length === 0) return null;
  return numbers.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, value: unknown): ProbedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? textOf(value.name).replace(/^models\//, '') : textOf(value.id);
  if (id === '') return null;
  const display =
    textOf(value.display_name) !== ''
      ? textOf(value.display_name)
      : textOf(value.displayName) !== ''
        ? textOf(value.displayName)
        : id;
  return {
    id,
    display_name: display,
    free: boolOrNull(value.free) ?? zeroPriced(value),
    context_length: numberOrNull(value.context_length) ?? numberOrNull(value.context_window) ?? numberOrNull(value.inputTokenLimit),
    tools: boolOrNull(value.tools),
    vision: boolOrNull(value.vision),
    json: boolOrNull(value.json),
  };
}

/** The models one catalog answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((value) => {
    const parsed = modelIn(wire, value);
    return parsed === null ? [] : [parsed];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating failure. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs: number): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byModelId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The catalog the probes and the quality table make, in directory order. */
export function buildCatalog(probes: readonly Probe[], ratings: readonly RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
