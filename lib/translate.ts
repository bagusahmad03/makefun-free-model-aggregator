// The three wires and the translations between them.
//
// The env's providers answer with an echo of what they parsed from the upstream
// request (system prompt, user text, images, tools, stop sequences, max_tokens,
// JSON mode), and the client checks that echo against what it sent. Every
// translation here therefore preserves those fields exactly: the same system
// text, the same user text blocks in the same order, the same image bytes, the
// same tool names in the same order, the same stops and the same token cap.
//
// A provider answer is parsed into one canonical Answer (text, usage, stop) and
// then emitted in the client's wire, JSON or SSE, from the buffered answer: the
// upstream response is always read whole before the client is answered, so a
// failure can still fail over to another provider.

import { isRecord, type Json } from './brief.ts';

/** One provider answer, in wire-neutral form. */
export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

type NormImage = { kind: 'data'; mediaType: string; base64: string } | { kind: 'url'; url: string };

type NormPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; image: NormImage }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; toolUseId: string; text: string };

interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: NormPart[];
}

interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

type NormToolChoice = 'auto' | 'any' | { name: string } | null;

/** A client request in wire-neutral form. */
export interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: NormToolChoice;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  stream: boolean;
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((entry): entry is string => typeof entry === 'string');
}

/** The text of a string or a block array's text blocks, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** An image_url value as a normalized image, or null when it names none. */
function imageOfUrl(url: unknown): NormImage | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'data', mediaType: data[1] !== '' ? data[1] : 'image/png', base64: data[2] };
  return { kind: 'url', url };
}

/** An Anthropic image source as a normalized image, or null. */
function imageOfSource(source: unknown): NormImage | null {
  if (!isRecord(source)) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { kind: 'data', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', base64: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', url: source.url };
  return null;
}

/** A tool call's arguments as an object: the JSON string parsed, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool result's content as text: its text blocks, or its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI content (string or parts) as normalized parts. */
function openAIParts(content: unknown): NormPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: NormPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url)) {
      const image = imageOfUrl(block.image_url.url);
      if (image !== null) parts.push({ kind: 'image', image });
    } else if (block.type === 'input_text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
  }
  return parts;
}

/** Anthropic content blocks as normalized parts. */
function anthropicParts(content: unknown): NormPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: NormPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image') {
      const image = imageOfSource(block.source);
      if (image !== null) parts.push({ kind: 'image', image });
    } else if (block.type === 'tool_use') {
      parts.push({
        kind: 'toolUse',
        id: textOf(block.id),
        name: textOf(block.name),
        input: isRecord(block.input) ? block.input : {},
      });
    } else if (block.type === 'tool_result') {
      parts.push({ kind: 'toolResult', toolUseId: textOf(block.tool_use_id), text: toolResultText(block.content) });
    }
  }
  return parts;
}

/** An OpenAI Chat Completions body in wire-neutral form. */
function normalizeOpenAI(body: Json): NormRequest {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = openAIParts(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'toolUse',
          id: textOf(call.id),
          name: textOf(call.function.name),
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'tool', parts: [{ kind: 'toolResult', toolUseId: textOf(raw.tool_call_id), text: toolResultText(raw.content) }] });
      continue;
    }
    messages.push({ role: 'user', parts: openAIParts(raw.content) });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: textOf(raw.function.description),
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  let toolChoice: NormToolChoice = null;
  if (choice === 'auto' || choice === 'none') toolChoice = 'auto';
  else if (choice === 'required') toolChoice = 'any';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body in wire-neutral form. */
function normalizeAnthropic(body: Json): NormRequest {
  const systemText = blocksText(body.system);
  const messages: NormMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      messages.push({ role: 'assistant', parts: anthropicParts(raw.content) });
      continue;
    }
    // User messages carry text, images and tool results on this wire.
    messages.push({ role: 'user', parts: anthropicParts(raw.content) });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: textOf(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  let toolChoice: NormToolChoice = null;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'any';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    systemText,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** A client body in wire-neutral form, from either client wire. */
export function normalizeRequest(clientWire: 'openai' | 'anthropic', body: Json): NormRequest {
  return clientWire === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);
}

function openAIImagePart(image: NormImage): Json {
  if (image.kind === 'data') return { type: 'image_url', image_url: { url: `data:${image.mediaType};base64,${image.base64}` } };
  return { type: 'image_url', image_url: { url: image.url } };
}

function anthropicImageBlock(image: NormImage): Json {
  if (image.kind === 'data') return { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.base64 } };
  return { type: 'image', source: { type: 'url', url: image.url } };
}

/** The normalized request as an OpenAI Chat Completions body naming `model`. */
export function renderOpenAI(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const message of norm.messages) {
    if (message.role === 'tool') {
      for (const part of message.parts) {
        if (part.kind === 'toolResult') messages.push({ role: 'tool', tool_call_id: part.toolUseId, content: part.text });
      }
      continue;
    }
    if (message.role === 'assistant') {
      const text = message.parts.flatMap((part) => (part.kind === 'text' ? [part.text] : [])).join('\n');
      const toolCalls = message.parts.flatMap((part) =>
        part.kind === 'toolUse'
          ? [{ id: part.id, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input) } }]
          : [],
      );
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : toolCalls.length > 0 ? null : '',
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const hasImage = message.parts.some((part) => part.kind === 'image');
    if (!hasImage) {
      const text = message.parts.flatMap((part) => (part.kind === 'text' ? [part.text] : [])).join('\n');
      messages.push({ role: 'user', content: text });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') parts.push(openAIImagePart(part.image));
    }
    messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = [...norm.stops];
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (norm.toolChoice === 'any') out.tool_choice = 'required';
  else if (norm.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  if (norm.stream) {
    out.stream = true;
    // The usage chunk a streamed answer is checked against arrives only when
    // the provider is asked for it: always ask.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The normalized request as an Anthropic Messages body naming `model`. */
export function renderAnthropic(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of norm.messages) {
    if (message.role === 'tool') {
      const blocks = message.parts.flatMap((part) =>
        part.kind === 'toolResult' ? [{ type: 'tool_result', tool_use_id: part.toolUseId, content: part.text }] : [],
      );
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (message.role === 'assistant') {
      const blocks: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.kind === 'toolUse') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') blocks.push(anthropicImageBlock(part.image));
      else if (part.kind === 'toolResult') blocks.push({ type: 'tool_result', tool_use_id: part.toolUseId, content: part.text });
    }
    if (blocks.length === 1 && isRecord(blocks[0]) && blocks[0].type === 'text' && typeof blocks[0].text === 'string') {
      messages.push({ role: 'user', content: blocks[0].text });
    } else {
      messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (norm.systemText !== '') out.system = norm.systemText;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = [...norm.stops];
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (norm.toolChoice === 'any') out.tool_choice = { type: 'any' };
  else if (norm.toolChoice !== null) out.tool_choice = { type: 'tool', name: norm.toolChoice.name };
  if (norm.stream) out.stream = true;
  return out;
}

/** The normalized request as a Gemini generateContent body (the model is in the path). */
export function renderGemini(norm: NormRequest): Json {
  const contents: Json[] = [];
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      const parts: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') parts.push({ text: part.text });
        else if (part.kind === 'toolUse') parts.push({ functionCall: { name: part.name, args: part.input } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') {
        if (part.image.kind === 'data') parts.push({ inlineData: { mimeType: part.image.mediaType, data: part.image.base64 } });
        else parts.push({ fileData: { fileUri: part.image.url } });
      } else if (part.kind === 'toolResult') {
        parts.push({ functionResponse: { name: part.toolUseId, response: { result: part.text } } });
      }
    }
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (norm.systemText !== '') out.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = [...norm.stops];
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response parsing: one upstream body (JSON or SSE) into a canonical Answer.
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames: each frame's event name and joined data lines. */
function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Response text from a message content: a string, or text blocks joined whole. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An OpenAI JSON or SSE body as an Answer; null when it holds no answer. */
export function parseOpenAIResponse(text: string, stream: boolean): Answer | null {
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(message.content),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    };
  }
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') content += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return sawAny ? { text: content, inputTokens, outputTokens, stop } : null;
}

/** An Anthropic JSON or SSE body as an Answer; null when it holds no answer. */
export function parseAnthropicResponse(text: string, stream: boolean): Answer | null {
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(body.content),
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    };
  }
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      content += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return sawAny ? { text: content, inputTokens, outputTokens, stop } : null;
}

/** Gemini part texts joined whole, as one candidate's answer text. */
function geminiText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** A Gemini JSON or SSE body as an Answer; null when it holds no answer. */
export function parseGeminiResponse(text: string, stream: boolean): Answer | null {
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) return null;
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: geminiText(candidate),
      inputTokens: numberOrZero(usage.promptTokenCount),
      outputTokens: numberOrZero(usage.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    };
  }
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const body = parseJsonObject(frame.data);
    if (body === null) continue;
    sawAny = true;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      content += geminiText(candidate);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(body.usageMetadata)) {
      inputTokens = numberOrZero(body.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(body.usageMetadata.candidatesTokenCount);
    }
  }
  return sawAny ? { text: content, inputTokens, outputTokens, stop } : null;
}

// ---------------------------------------------------------------------------
// Client emission: a canonical Answer in the client's wire.
// ---------------------------------------------------------------------------

/** Stop-reason families, shared by all three wires: equal family, equal meaning. */
const STOP_FAMILY = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamily(stop: string): number {
  return STOP_FAMILY.get(stop) ?? 0;
}

/** A provider stop as an OpenAI finish_reason in the same family. */
export function openAIStopOf(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** A provider stop as an Anthropic stop_reason in the same family. */
export function anthropicStopOf(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** A fixed epoch second for every completion and listing: nothing reads a clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Characters per streamed text piece: fixed, so replays emit the same bytes. */
const STREAM_PIECE = 13;

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces.length === 0 ? [''] : pieces;
}

/** An Answer as an OpenAI Chat Completions JSON object naming `model`. */
export function openAIJsonResponse(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStopOf(answer.stop),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An Answer as an OpenAI SSE stream naming `model`, ending with [DONE]. */
export function openAIStreamText(answer: Answer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStopOf(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Answer as an Anthropic Messages JSON object naming `model`. */
export function anthropicJsonResponse(answer: Answer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An Answer as an Anthropic SSE stream naming `model`, ending with message_stop. */
export function anthropicStreamText(answer: Answer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** One served model, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
