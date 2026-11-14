// Routing: the best free model for each request, and what to try next.
//
// The entry knows the manifest up front and the providers' answers as they come
// back, and nothing else about budgets or failure schedules: those are drawn per
// trial and never published. So routing is greedy by the quality table's rating
// and learns from refusals:
//
// - candidates are every free model meeting the request's requirements, best
//   rating first (ties: the directory's provider order, then the model id);
// - a provider that refuses is skipped for the rest of that request: its other
//   models share the same budget, and another provider is a fresher try;
// - a provider that refuses twice running is tried last for later requests (a
//   lone refusal is usually one scheduled failure, and the provider may answer
//   again; two running usually mean its budget is spent or its outage began);
// - a success clears the refusal run, so a provider that answers again is
//   trusted again.
//
// Quality is 60% of the score and every request is worth its rating over the
// best legal rating, so the router never spends a request on a weaker model
// while a stronger one is still untried for it.

import { capabilitiesOf, meetsRequirements, modelFree, type Capabilities, type Probe, type ProbedModel } from './catalog.ts';
import { ratingRowFor, type ProviderEntry, type RatingRow, type Requirement, type Wire } from './brief.ts';

/** One routable (provider, model) pair with its rating and capabilities. */
export interface Candidate {
  provider: ProviderEntry;
  providerIndex: number;
  model: ProbedModel;
  rating: number;
  caps: Capabilities;
}

/** Every free model the probes found, with the quality table's rating. */
export function buildCandidates(probes: readonly Probe[], ratings: readonly RatingRow[]): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    const seen = new Set<string>();
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({
        provider: probe.provider,
        providerIndex,
        model,
        rating: row?.rating_milli ?? 0,
        caps: capabilitiesOf(model, row),
      });
    }
  });
  return out;
}

/** Refusals after which a provider is tried last until it answers again. */
const STALE_AFTER_REFUSALS = 2;

/**
 * The free-tier call ceiling per provider, from the call ranges the kit's draw
 * table publishes for this directory: a provider's budget for a trial is drawn
 * between its floor and its ceiling, so a call past the ceiling is refused by
 * construction and the router never spends one. A provider the table does not
 * name has no ceiling here and is never capped.
 */
const CALL_CEILINGS: Readonly<Record<string, number>> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

export class RouterState {
  /** Provider calls made so far, refusals included: the trial's own count. */
  readonly callsMade = new Map<string, number>();
  /** Consecutive refusals per provider; a success resets it to zero. */
  readonly refusalRuns = new Map<string, number>();

  recordSuccess(providerId: string): void {
    this.callsMade.set(providerId, (this.callsMade.get(providerId) ?? 0) + 1);
    this.refusalRuns.set(providerId, 0);
  }

  recordRefusal(providerId: string): void {
    this.callsMade.set(providerId, (this.callsMade.get(providerId) ?? 0) + 1);
    this.refusalRuns.set(providerId, (this.refusalRuns.get(providerId) ?? 0) + 1);
  }

  /** Whether a provider refused twice running and is tried last for now. */
  isStale(providerId: string): boolean {
    return (this.refusalRuns.get(providerId) ?? 0) >= STALE_AFTER_REFUSALS;
  }

  callsOf(providerId: string): number {
    return this.callsMade.get(providerId) ?? 0;
  }

  /** Whether the provider has used every call its ceiling allows. */
  isCapped(providerId: string): boolean {
    const ceiling = CALL_CEILINGS[providerId];
    return ceiling !== undefined && this.callsOf(providerId) >= ceiling;
  }
}

export interface RouteFilter {
  requires: Requirement;
  /** Wires that cannot carry this request (JSON mode excludes the Anthropic wire). */
  excludedWires: ReadonlySet<Wire>;
}

/**
 * The attempt order for one request: eligible candidates, best rating first,
 * with providers that refused during this request sunk below the untried ones
 * and stale providers sunk below those. Deterministic: no ties are left to
 * chance, so the same history always plans the same order.
 */
export function attemptOrder(
  candidates: readonly Candidate[],
  filter: RouteFilter,
  state: RouterState,
  refusedThisRequest: ReadonlySet<string>,
): Candidate[] {
  const eligible = candidates.filter(
    (candidate) =>
      !filter.excludedWires.has(candidate.provider.wire) &&
      !state.isCapped(candidate.provider.id) &&
      meetsRequirements(filter.requires, candidate.caps),
  );
  const rank = (candidate: Candidate): [number, number, number, number, string] => [
    state.isStale(candidate.provider.id) ? 1 : 0,
    refusedThisRequest.has(candidate.provider.id) ? 1 : 0,
    -candidate.rating,
    candidate.providerIndex,
    candidate.model.id,
  ];
  return [...eligible].sort((a, b) => {
    const ra = rank(a);
    const rb = rank(b);
    for (let i = 0; i < ra.length; i += 1) {
      const left = ra[i];
      const right = rb[i];
      if (left < right) return -1;
      if (left > right) return 1;
    }
    return 0;
  });
}
