# Free model aggregator

One server that serves the OpenAI and Anthropic APIs over the directory of
free-tier providers the brief names. Every request goes to the best free model
that meets its requirements (by the brief's quality table); a provider that
refuses hands the request to the next candidate. The server reports the
catalog it read from the providers' own catalog routes, in the one document
the register call and `--list` both print.

## Layout

- `main.ts` — the server: brief, probes, register, routes, polling.
- `lib/brief.ts` — brief parsing: providers, quality table, manifest.
- `lib/catalog.ts` — catalog probes and the catalog document.
- `lib/router.ts` — candidate ranking and refusal-aware failover.
- `lib/translate.ts` — wire-neutral request form, per-wire rendering and
  parsing, client responses.
- `lib/upstream.ts` — one provider call, in that provider's own wire.

## Run

The harness starts it with the env and the brief in the environment:

```sh
node main.ts
```

`node main.ts --list` prints the catalog document and exits. It needs nothing
beyond the Node standard library and makes no calls outside the simulated env.
