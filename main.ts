import "@std/dotenv/load";

const OPENROUTER_KEY = Deno.env.get("OPENROUTER_KEY") ?? "";
const TAVILY_KEY = Deno.env.get("TAVILY_KEY") ?? "";
const EXA_KEY = Deno.env.get("EXA_KEY") ?? "";

if (!OPENROUTER_KEY || !TAVILY_KEY) {
  console.error("Missing OPENROUTER_KEY or TAVILY_KEY in .env");
  Deno.exit(1);
}
if (!EXA_KEY) console.warn("EXA_KEY not set — Exa will fall back to Tavily.");

const HTML = await Deno.readTextFile("./public/index.html");

// ── Types ─────────────────────────────────────────────────────────────────────

interface SearchResult { title: string; url: string; content: string; }
interface Message { role: string; content: string; }
interface AgentEvent { type: string; data: unknown; }
interface SearchQuery { query: string; domain: string; }
interface Article { domain: string; title: string; url: string; content: string; }

interface PredParams {
  model: string;
  domains: string[];
  topics: string;
  mode: "scan" | "focus";
  question: string;
}

// ── Tavily ────────────────────────────────────────────────────────────────────

async function tavilyNewsSearch(query: string, maxResults = 5): Promise<SearchResult[]> {
  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      api_key: TAVILY_KEY, query, search_depth: "basic",
      max_results: maxResults, topic: "news", days: 1,
    }),
  });
  if (!res.ok) throw new Error(`Tavily ${res.status}`);
  const data = await res.json();
  return (data.results ?? []).map((r: SearchResult & { content?: string }) => ({
    title: r.title, url: r.url, content: (r.content ?? "").slice(0, 500),
  }));
}

// ── Exa ───────────────────────────────────────────────────────────────────────

async function exaSearch(query: string, maxResults = 4): Promise<SearchResult[]> {
  if (!EXA_KEY) return tavilyNewsSearch(query, maxResults);
  const res = await fetch("https://api.exa.ai/search", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": EXA_KEY },
    body: JSON.stringify({ query, numResults: maxResults, type: "neural", contents: { text: { maxCharacters: 500 } } }),
  });
  if (!res.ok) { console.warn(`Exa ${res.status} — fallback`); return tavilyNewsSearch(query, maxResults); }
  const data = await res.json();
  return (data.results ?? []).map((r: { title: string; url: string; text?: string }) => ({
    title: r.title ?? "", url: r.url ?? "", content: (r.text ?? "").slice(0, 500),
  }));
}

// ── OpenRouter ────────────────────────────────────────────────────────────────

async function chat(model: string, messages: Message[]): Promise<string> {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": `Bearer ${OPENROUTER_KEY}`,
      "x-title": "News Lens",
    },
    body: JSON.stringify({ model, max_tokens: 2500, temperature: 0.35, messages }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message ?? `OpenRouter ${res.status}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}

// ── Prompts ───────────────────────────────────────────────────────────────────

const TODAY = new Date().toISOString().slice(0, 10);

function buildPlannerPrompt(p: PredParams): string {
  return `You are a research planning agent for a news intelligence tool.
Today: ${TODAY}
${p.mode === "focus" ? "FOCUS — depth over breadth." : "SCAN — broad sweep across domains."}
Domains: ${p.domains.join(", ")}
${p.topics ? `Topics/regions: ${p.topics}` : ""}
${p.question ? `User question: "${p.question}"` : ""}

Output a JSON array of search queries. No prose, no markdown fences.
[{"query":"...","domain":"geopolitical|financial|tech|crypto|ai"},...]
Pick ${p.mode === "focus" ? "3-5" : "2-4"} highly targeted, specific queries. Name countries, companies, people.`;
}

function buildAnalystPrompt(p: PredParams): string {
  return `You are a sharp geopolitical, financial, technology, cryptocurrency and AI intelligence analyst.
${p.mode === "focus" ? `FOCUS analysis. ${p.question ? `Answer: "${p.question}"` : ""}` : "SCAN — surface the most significant cross-domain developments."}
Horizon: 24 hours. Today: ${TODAY}.

You MUST do exactly two rounds:
- Round 1 (Tavily): you will receive breaking news articles. You MUST request follow-up searches to find deeper context and contradicting evidence.
- Round 2 (Exa): after follow-up results, finalize your predictions.
Do NOT finalize on round 1 unless you have 0 rounds remaining.

To search more:
{"action":"search","queries":[{"query":"...","domain":"geopolitical|financial|tech|crypto|ai"}],"reason":"..."}

To finalize:
{
  "action": "finalize",
  "predictions": [
    {
      "domain": "geopolitical|financial|tech|crypto|ai",
      "title": "Short punchy headline",
      "confidence": "HIGH|MEDIUM|LOW|RISK",
      "body": "2-3 sentences grounded in the articles.",
      "signals": "Specific supporting evidence."
    }
  ],
  "sources": ["Article title — publication"],
  "summary": "One sentence meta-observation."
}

Respond ONLY with valid JSON. No markdown fences.
4-6 predictions. Name countries, companies, people. No vague platitudes.
${p.question ? `At least one prediction directly answers: "${p.question}"` : ""}`;
}

// ── Predictions loop ──────────────────────────────────────────────────────────

async function runPredictions(params: PredParams, emit: (e: AgentEvent) => void) {
  const MAX_ROUNDS = params.mode === "focus" ? 3 : 2;
  const messages: Message[] = [];
  const allArticles: Article[] = [];
  const searchLog: { round: number; query: string; domain: string }[] = [];

  emit({ type: "status", data: "Planning searches..." });

  const planRaw = await chat(params.model, [
    { role: "system", content: buildPlannerPrompt(params) },
    { role: "user", content: "Plan my searches now." },
  ]);

  let queries: SearchQuery[] = [];
  try {
    const m = planRaw.replace(/```json|```/g, "").match(/\[[\s\S]*\]/);
    queries = JSON.parse(m?.[0] ?? "[]");
  } catch { throw new Error("Planner returned invalid JSON."); }

  async function runSearches(qs: SearchQuery[], round: number) {
    if (allArticles.length >= 20) {
      emit({ type: "status", data: "Article cap reached, skipping search." });
      return;
    }
    emit({ type: "search_plan", data: { round, queries: qs } });
    for (const q of qs) {
      // Round 1: Tavily (recency), Round 2+: Exa (depth)
      const results = round === 1
        ? await tavilyNewsSearch(q.query, 4)
        : await exaSearch(q.query, 4);
      emit({ type: "status", data: `[${round === 1 ? "Tavily" : "Exa"}] "${q.query}"` });
      searchLog.push({ round, query: q.query, domain: q.domain });
      results.forEach(r => allArticles.push({ domain: q.domain, ...r }));
    }
    emit({ type: "search_done", data: { round, total: allArticles.length } });
  }

  await runSearches(queries, 1);
  messages.push({ role: "system", content: buildAnalystPrompt(params) });

  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const articlesText = allArticles
      .map(a => `[${a.domain.toUpperCase()}] ${a.title}\n${a.url}\n${a.content}`)
      .join("\n\n---\n\n");
    const remaining = MAX_ROUNDS - round;
    messages.push({
      role: "user",
      content: `Articles (${allArticles.length}):\n\n${articlesText}\n\n${remaining > 0 ? `${remaining} round(s) left. Search more or finalize?` : "No more rounds. Finalize now."
        }`,
    });
    emit({ type: "status", data: `Reasoning over ${allArticles.length} articles...` });

    const raw = await chat(params.model, messages);
    messages.push({ role: "assistant", content: raw });

    let parsed: Record<string, unknown>;
    try {
      const m = raw.replace(/```json|```/g, "").match(/\{[\s\S]*\}/);
      if (!m) throw new Error("no JSON");
      parsed = JSON.parse(m[0]);
    } catch {
      // truncated JSON — force finalize
      parsed = { action: "finalize" };
    }
    
    if ((parsed.action === "search" && remaining > 0) || (round === 1 && remaining > 0)) {
      const followUpQueries = parsed.action === "search"
        ? (parsed.queries as SearchQuery[]) ?? []
        : params.domains.map(d => ({ query: `${params.topics || d} contradictions risks alternative view`, domain: d }));
      if (parsed.reason) emit({ type: "status", data: `Following up: ${parsed.reason}` });
      await runSearches(followUpQueries, round + 1);
      continue;
    }
    emit({ type: "result", data: { ...parsed, searchLog } });
    return;
  }
  throw new Error("Agent exhausted all rounds without finalizing.");
}

// ── HTTP handler ──────────────────────────────────────────────────────────────

function sseStream(fn: (emit: (e: AgentEvent) => void) => Promise<void>): Response {
  const stream = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      const emit = (e: AgentEvent) =>
        controller.enqueue(enc.encode(`data: ${JSON.stringify(e)}\n\n`));
      try { await fn(emit); }
      catch (err) { emit({ type: "error", data: (err as Error).message }); }
      finally { controller.close(); }
    },
  });
  return new Response(stream, {
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache", "connection": "keep-alive" },
  });
}

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);

  if (url.pathname === "/" || url.pathname === "/index.html")
    return new Response(HTML, { headers: { "content-type": "text/html; charset=utf-8" } });

  if (url.pathname === "/api/predictions" && req.method === "POST") {
    const params: PredParams = await req.json();
    return sseStream(emit => runPredictions(params, emit));
  }

  return new Response("Not found", { status: 404 });
}

const port = parseInt(Deno.env.get("PORT") ?? "8000");
console.log(`News Lens listening on http://localhost:${port}`);
Deno.serve({ port }, handler);
