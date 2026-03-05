/**
 * BrainBoost AI — app.ts  (TypeScript source)
 * Compile with:  tsc --noEmit   (type-check only)
 *
 * The runtime file is app.js (pre-compiled) which loads Transformers.js
 * dynamically from CDN. No build step required for GitHub Pages deployment.
 *
 * Fast  → Xenova/flan-t5-base           task: text2text-generation
 * Smart → Xenova/TinyLlama-1.1B-Chat-v1.0  task: text-generation
 *
 * Created by Vrishab Varun
 */

// ── Types ──────────────────────────────────────────────────────
type Mode = "fast" | "smart";

interface ModelConfig {
  readonly id:           string;
  readonly task:         string;
  readonly label:        string;
  readonly icon:         string;
  readonly size:         string;
  readonly maxNewTokens: number;
}

interface GenerationOutput {
  generated_text: string;
}

interface ProgressEvent {
  status:    "initiate" | "download" | "progress" | "done" | "ready";
  file?:     string;
  progress?: number;
  loaded?:   number;
  total?:    number;
}

// Transformers.js dynamic import result (loose typing for CDN version)
interface TransformersLib {
  pipeline: (
    task: string,
    model: string,
    options?: Record<string, unknown>
  ) => Promise<(input: string, opts?: Record<string, unknown>) => Promise<GenerationOutput | GenerationOutput[]>>;
  env: {
    allowLocalModels: boolean;
    useBrowserCache:  boolean;
  };
}

// ── Config ─────────────────────────────────────────────────────
const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

const MODELS: Record<Mode, ModelConfig> = {
  fast: {
    id:           "Xenova/flan-t5-base",
    task:         "text2text-generation",
    label:        "Flan-T5 Base",
    icon:         "⚡",
    size:         "~250 MB",
    maxNewTokens: 200,
  },
  smart: {
    id:           "Xenova/TinyLlama-1.1B-Chat-v1.0",
    task:         "text-generation",
    label:        "TinyLlama Chat",
    icon:         "🧠",
    size:         "~600 MB",
    maxNewTokens: 320,
  },
};

// ── Prompt helpers ──────────────────────────────────────────────
function buildPrompt(question: string, mode: Mode): string {
  if (mode === "fast") {
    return `Answer clearly and helpfully: ${question}`;
  }
  return (
    `<|system|>\nYou are BrainBoost AI, a smart and helpful assistant. ` +
    `Answer questions clearly and concisely.</s>\n` +
    `<|user|>\n${question}</s>\n` +
    `<|assistant|>\n`
  );
}

function extractAnswer(output: GenerationOutput | GenerationOutput[], mode: Mode): string {
  if (mode === "fast") {
    return (Array.isArray(output)
      ? output[0]?.generated_text
      : (output as GenerationOutput)?.generated_text ?? "") || "";
  }
  const full = Array.isArray(output)
    ? output[0]?.generated_text ?? ""
    : (output as GenerationOutput)?.generated_text ?? "";

  const marker = "<|assistant|>\n";
  const idx    = full.lastIndexOf(marker);
  let answer   = idx !== -1 ? full.slice(idx + marker.length) : full;
  answer       = answer.split("</s>")[0].split("<|")[0].trim();
  return answer;
}

// ── State ───────────────────────────────────────────────────────
let currentMode:     Mode    = "fast";
let isGenerating:    boolean = false;
let hasWebGPU:       boolean = false;
const pipelines:     Partial<Record<Mode, ReturnType<TransformersLib["pipeline"]> extends Promise<infer T> ? T : never>> = {};
let transformersLib: TransformersLib | null = null;

// ── DOM ─────────────────────────────────────────────────────────
const $  = (id: string): HTMLElement => document.getElementById(id)!;

const ui = {
  q:         $("q")         as HTMLTextAreaElement,
  askBtn:    $("ask-btn")   as HTMLButtonElement,
  askLabel:  $("ask-label") as HTMLSpanElement,
  qLen:      $("qlen")      as HTMLSpanElement,
  clearQ:    $("clear-q")   as HTMLButtonElement,
  modelBar:  $("model-bar") as HTMLDivElement,
  barText:   $("bar-text")  as HTMLSpanElement,
  barFill:   $("bar-fill")  as HTMLDivElement,
  barPct:    $("bar-pct")   as HTMLSpanElement,
  respWrap:  $("resp-wrap") as HTMLDivElement,
  respBody:  $("resp-body") as HTMLDivElement,
  respDots:  $("resp-dots") as HTMLDivElement,
  respMeta:  $("resp-meta") as HTMLDivElement,
  respIcon:  $("resp-icon") as HTMLSpanElement,
  copyBtn:   $("copy-btn")  as HTMLButtonElement,
  btnFast:   $("btn-fast")  as HTMLButtonElement,
  btnSmart:  $("btn-smart") as HTMLButtonElement,
  gpuPill:   $("gpu-pill")  as HTMLSpanElement,
};

// ── Boot ────────────────────────────────────────────────────────
async function boot(): Promise<void> {
  await detectWebGPU();
  initCanvas();
  initEvents();
  setTimeout(() => loadModel("fast").catch(console.error), 800);
}

// ── WebGPU ─────────────────────────────────────────────────────
async function detectWebGPU(): Promise<void> {
  try {
    if (!("gpu" in navigator)) return;
    const adapter = await (navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } })
      .gpu.requestAdapter();
    if (adapter) {
      hasWebGPU = true;
      ui.gpuPill.hidden = false;
    }
  } catch { /* unavailable */ }
}

// ── Library loading ─────────────────────────────────────────────
async function loadLibrary(): Promise<TransformersLib> {
  if (transformersLib) return transformersLib;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformersLib = await import(TRANSFORMERS_URL) as unknown as TransformersLib;
  transformersLib.env.allowLocalModels = false;
  transformersLib.env.useBrowserCache  = true;
  return transformersLib;
}

// ── Progress ────────────────────────────────────────────────────
function onProgress(ev: ProgressEvent): void {
  showBar(true);
  const { status, progress, loaded, total, file } = ev;
  if (status === "initiate") {
    ui.barText.textContent = `⬇ Fetching ${file ?? "model"}…`;
    setFill(0);
    ui.barPct.textContent  = "";
  } else if (status === "download" || status === "progress") {
    const pct = progress != null ? progress : (total != null && total > 0 ? ((loaded ?? 0) / total) * 100 : 0);
    setFill(pct);
    if (loaded != null && total != null && total > 0) {
      const mb = (b: number): string => (b / 1e6).toFixed(1);
      ui.barPct.textContent = `${mb(loaded)} / ${mb(total)} MB`;
    } else {
      ui.barPct.textContent = pct > 0 ? `${pct.toFixed(0)}%` : "";
    }
  } else if (status === "done") {
    setFill(100);
    ui.barText.textContent = "✓ Model ready!";
    ui.barPct.textContent  = "";
    setTimeout(() => showBar(false), 1200);
  }
}

function setFill(pct: number): void {
  ui.barFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

// ── Model loading ───────────────────────────────────────────────
async function loadModel(mode: Mode): Promise<unknown> {
  if (pipelines[mode]) return pipelines[mode];
  const cfg  = MODELS[mode];
  const lib  = await loadLibrary();
  showBar(true);
  ui.barText.textContent = `🧠 Loading ${cfg.label}…`;
  setFill(4);
  ui.barPct.textContent  = cfg.size;
  const device = (mode === "smart" && hasWebGPU) ? "webgpu" : "wasm";
  const pipe   = await lib.pipeline(cfg.task, cfg.id, {
    progress_callback: onProgress,
    device,
  });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (pipelines as any)[mode] = pipe;
  showBar(false);
  return pipe;
}

// ── Generate ────────────────────────────────────────────────────
async function generate(question: string): Promise<void> {
  if (isGenerating || !question.trim()) return;
  isGenerating = true;
  setAskLoading(true);

  ui.respWrap.style.display = "block";
  ui.respBody.textContent   = "";
  ui.respMeta.textContent   = "";
  ui.respDots.style.display = "flex";
  ui.respIcon.textContent   = MODELS[currentMode].icon;

  const t0 = performance.now();
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pipe   = (pipelines as any)[currentMode] ?? await loadModel(currentMode);
    const cfg    = MODELS[currentMode];
    const prompt = buildPrompt(question, currentMode);
    const output = await pipe(prompt, {
      max_new_tokens:     cfg.maxNewTokens,
      temperature:        currentMode === "fast" ? 0.7 : 0.65,
      repetition_penalty: 1.25,
      do_sample:          true,
      top_k:              50,
      top_p:              0.92,
    });
    const ms     = performance.now() - t0;
    const answer = extractAnswer(output as GenerationOutput | GenerationOutput[], currentMode);
    const text   = answer || "I couldn't produce a good answer. Try rephrasing.";
    ui.respDots.style.display = "none";
    typewrite(ui.respBody, text);
    const words = text.split(/\s+/).filter(Boolean).length;
    ui.respMeta.textContent =
      `${cfg.label} · ${words} words · ${(ms / 1000).toFixed(2)}s · ~${((words / ms) * 1000).toFixed(1)} tok/s`;
  } catch (err: unknown) {
    ui.respDots.style.display = "none";
    const msg = err instanceof Error ? err.message : String(err);
    ui.respBody.innerHTML = `<span style="color:#f59e0b">⚠ Error: ${esc(msg)}</span>`;
    console.error("[BrainBoost]", err);
  } finally {
    isGenerating = false;
    setAskLoading(false);
  }
}

// ── Typewriter ──────────────────────────────────────────────────
function typewrite(el: HTMLElement, text: string): void {
  el.textContent = "";
  const STEP = 5; let i = 0;
  function tick() {
    if (i >= text.length) return;
    el.textContent += text.slice(i, i + STEP);
    i += STEP;
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Mode switch ─────────────────────────────────────────────────
function switchMode(mode: Mode): void {
  if (mode === currentMode) return;
  currentMode = mode;
  ui.btnFast.classList.toggle("tab--active",  mode === "fast");
  ui.btnSmart.classList.toggle("tab--active", mode === "smart");
  ui.btnFast.setAttribute("aria-selected",  String(mode === "fast"));
  ui.btnSmart.setAttribute("aria-selected", String(mode === "smart"));
  ui.respWrap.style.display = "none";
  showBar(false);
  loadModel(mode).catch(console.error);
}

// ── UI helpers ──────────────────────────────────────────────────
function showBar(v: boolean): void { ui.modelBar.classList.toggle("model-bar--hidden", !v); }
function setAskLoading(l: boolean): void {
  ui.askBtn.disabled = l; ui.q.disabled = l;
  ui.askLabel.textContent = l ? "Thinking…" : "Ask";
}
function esc(s: string): string {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function toast(msg: string): void {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2200);
}

// ── Neural canvas ───────────────────────────────────────────────
function initCanvas(): void {
  const canvas = document.getElementById("neural-canvas") as HTMLCanvasElement;
  const ctx    = canvas?.getContext("2d");
  if (!ctx) return;
  interface Node { x:number; y:number; vx:number; vy:number; r:number; }
  const COUNT = window.innerWidth < 600 ? 24 : 44;
  const MDIST = 130;
  const nodes: Node[] = [];
  const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
  const spawn  = () => {
    nodes.length = 0;
    for (let i = 0; i < COUNT; i++) nodes.push({
      x: Math.random()*canvas.width, y: Math.random()*canvas.height,
      vx: (Math.random()-0.5)*0.38,  vy: (Math.random()-0.5)*0.38,
      r: Math.random()*2+0.8,
    });
  };
  const draw = () => {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x<0||n.x>canvas.width)  n.vx*=-1;
      if (n.y<0||n.y>canvas.height) n.vy*=-1;
    }
    for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {
      const dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
      const d=Math.sqrt(dx*dx+dy*dy);
      if (d<MDIST) {
        ctx.beginPath(); ctx.moveTo(nodes[i].x,nodes[i].y); ctx.lineTo(nodes[j].x,nodes[j].y);
        ctx.strokeStyle=`rgba(34,211,238,${(1-d/MDIST)*0.3})`; ctx.lineWidth=0.7; ctx.stroke();
      }
    }
    for (const n of nodes) { ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,Math.PI*2); ctx.fillStyle="rgba(168,85,247,0.65)"; ctx.fill(); }
    requestAnimationFrame(draw);
  };
  resize(); spawn(); draw();
  window.addEventListener("resize",()=>{ resize(); spawn(); });
}

// ── Events ──────────────────────────────────────────────────────
function initEvents(): void {
  ui.btnFast.addEventListener("click",  () => switchMode("fast"));
  ui.btnSmart.addEventListener("click", () => switchMode("smart"));
  ui.askBtn.addEventListener("click",   () => { const q=ui.q.value.trim(); if(q&&!isGenerating) generate(q); });
  ui.q.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key==="Enter"&&!e.shiftKey) { e.preventDefault(); const q=ui.q.value.trim(); if(q&&!isGenerating) generate(q); }
  });
  ui.q.addEventListener("input", () => {
    const len=ui.q.value.length;
    ui.qLen.textContent=`${len} / 800`;
    ui.clearQ.style.display=len>0?"inline-flex":"none";
    ui.q.style.height="auto";
    ui.q.style.height=Math.min(ui.q.scrollHeight,240)+"px";
  });
  ui.clearQ.addEventListener("click", () => {
    ui.q.value=""; ui.qLen.textContent="0 / 800";
    ui.clearQ.style.display="none"; ui.q.style.height="auto"; ui.q.focus();
  });
  ui.copyBtn.addEventListener("click", async () => {
    const text=ui.respBody.textContent??"";
    if(!text) return;
    try { await navigator.clipboard.writeText(text); toast("✓ Copied!"); }
    catch { toast("Select text manually to copy"); }
  });
}

boot();
