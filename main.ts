// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief, probes every provider's catalog route, registers
// the one catalog with the env, and answers the client's 20 requests: each with
// the best free model that meets its requirements, failing over to the next
// candidate when a provider refuses, in the wire the client used. It exits 0
// when the env says the workload is done.
//
// Routing is greedy by the quality table's rating and learns refusals as they
// come (see lib/router.ts). Every provider answer is buffered and translated
// from the canonical form (see lib/translate.ts), so a failure can still fail
// over before the client is answered. Nothing here reads a clock or a random
// source: the same brief, requests and provider answers always give the same
// bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { isRecord, ratingRowFor, readBrief, type Brief, type Json, type Requirement } from './lib/brief.ts';
import { buildCatalog, modelFree, probeProvider, type Catalog, type Probe } from './lib/catalog.ts';
import { attemptOrder, buildCandidates, RouterState, type Candidate } from './lib/router.ts';
import { callUpstream } from './lib/upstream.ts';
import {
  anthropicError,
  anthropicJsonResponse,
  anthropicModelList,
  anthropicStreamText,
  openAIError,
  openAIJsonResponse,
  openAIModelList,
  openAIStreamText,
  type Answer,
  type ServedModel,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Register attempts before the trial is abandoned. */
const REGISTER_ATTEMPTS = 3;
/** Milliseconds between register attempts. */
const REGISTER_RETRY_MS = 500;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface Loaded {
  brief: Brief;
  probes: Probe[];
  catalog: Catalog;
  candidates: Candidate[];
  router: RouterState;
}

let loaded: Loaded | null = null;
/** Client API requests arrived so far: the manifest is in arrival order. */
let manifestIndex = 0;
/** One request at a time reaches the providers, in arrival order. */
let queue: Promise<void> = Promise.resolve();

/** Probe every provider in directory order and build the one catalog. */
async function load(): Promise<Loaded> {
  const brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  const probes: Probe[] = [];
  for (const provider of brief.providers) {
    const probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    if (!probe.ok && probe.error !== null) log(probe.error);
    probes.push(probe);
  }
  const catalog = buildCatalog(probes, brief.ratings);
  const candidates = buildCandidates(probes, brief.ratings);
  return { brief, probes, catalog, candidates, router: new RouterState() };
}

/** The free models across the providers that answered, one row per model id. */
function servedModels(state: Loaded): ServedModel[] {
  const byId = new Map<string, ServedModel>();
  for (const probe of state.probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRowFor(state.brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!byId.has(model.id)) byId.set(model.id, { id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** The requirements of the request now arriving, from the manifest by order. */
function requirementsFor(state: Loaded, body: Json): Requirement {
  const entry = state.brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse, state: Loaded): void {
  const list = servedModels(state);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface Served {
  candidate: Candidate;
  answer: Answer;
}

/**
 * Serve one request: try the candidates in routing order until one provider
 * answers, failing over on every refusal. A provider that refuses is skipped
 * for the rest of this request; its other models share the same budget.
 */
async function serve(state: Loaded, wire: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<Served | null> {
  // JSON mode rides the OpenAI wire's response_format and Gemini's
  // responseMimeType; the Anthropic wire cannot carry it, so its providers are
  // never tried for a request the client sent in JSON mode.
  const jsonMode = wire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const excludedWires = new Set(jsonMode ? ['anthropic' as const] : []);
  const refused = new Set<string>();
  const tried = new Set<string>();
  for (;;) {
    const order = attemptOrder(state.candidates, { requires, excludedWires }, state.router, refused);
    const next = order.find((candidate) => !tried.has(`${candidate.provider.id}\n${candidate.model.id}`));
    if (next === undefined) return null;
    tried.add(`${next.provider.id}\n${next.model.id}`);
    const outcome = await callUpstream(ENV_URL, next.provider, next.model.id, wire, body);
    if (outcome.ok) {
      state.router.recordSuccess(next.provider.id);
      log(`served by ${next.provider.id}/${next.model.id} after ${tried.size} attempt(s)`);
      return { candidate: next, answer: outcome.answer };
    }
    state.router.recordRefusal(next.provider.id);
    refused.add(next.provider.id);
    log(`${next.provider.id}/${next.model.id} refused HTTP ${outcome.status} (${outcome.code}); failing over`);
  }
}

/** Answer one client request on either wire, through the router. */
async function handleApi(req: IncomingMessage, res: ServerResponse, state: Loaded, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const requires = requirementsFor(state, body);
  const served = await serve(state, wire, body, requires);
  if (served === null) {
    sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
    return;
  }
  const model = served.candidate.model.id;
  if (body.stream === true) {
    sendStream(res, wire === 'anthropic' ? anthropicStreamText(served.answer, model) : openAIStreamText(served.answer, model));
    return;
  }
  sendJson(res, 200, wire === 'anthropic' ? anthropicJsonResponse(served.answer, model) : openAIJsonResponse(served.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse, state: Loaded): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res, state);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    // Serialize provider traffic: one request at a time, in arrival order.
    const task = queue.then(() => handleApi(req, res, state, wire));
    queue = task.then(
      () => undefined,
      () => undefined,
    );
    await task;
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog, retrying briefly. */
async function register(port: number, catalog: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      if (attempt < REGISTER_ATTEMPTS) await sleep(REGISTER_RETRY_MS);
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  loaded = await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts. Wait for stdout to flush
    // so the tail is never lost.
    const text = `${JSON.stringify(loaded.catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const state = loaded;
  const server = createServer((req, res) => {
    void handle(req, res, state).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${state.catalog.providers.length} answered providers and ${state.candidates.length} free models`);
  await register(port, state.catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
