/**
 * BrainBoost AI — app.js
 * ─────────────────────────────────────────────────────────
 * Fully client-side AI using HuggingFace Transformers.js (v2).
 *
 * Fast  → Xenova/flan-t5-base          task: text2text-generation (~250 MB)
 * Smart → Xenova/TinyLlama-1.1B-Chat-v1.0  task: text-generation  (~600 MB)
 *
 * Both models are quantized ONNX, cached in the browser after first download.
 * Created by Vrishab Varun
 */

// ── CDN import (Transformers.js v2 — most compatible) ──────────
const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// ── Model Registry ──────────────────────────────────────────────
const MODELS = {
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

// ── Prompt templates ────────────────────────────────────────────
function buildPrompt(question, mode) {
  if (mode === "fast") {
    // Flan-T5 is instruction-tuned — pass directly
    return `Answer clearly and helpfully: ${question}`;
  }
  // TinyLlama uses ChatML / <|system|> format
  return (
    `<|system|>\nYou are BrainBoost AI, a smart and helpful assistant. ` +
    `Answer questions clearly and concisely.</s>\n` +
    `<|user|>\n${question}</s>\n` +
    `<|assistant|>\n`
  );
}

function extractAnswer(output, mode) {
  if (mode === "fast") {
    // text2text: output is already just the answer
    return (Array.isArray(output)
      ? output[0]?.generated_text
      : output?.generated_text ?? "") || "";
  }
  // text-generation: strip the prompt prefix
  const full = Array.isArray(output)
    ? output[0]?.generated_text ?? ""
    : output?.generated_text ?? "";

  // Remove everything up to and including <|assistant|>
  const marker = "<|assistant|>\n";
  const idx = full.lastIndexOf(marker);
  let answer = idx !== -1 ? full.slice(idx + marker.length) : full;

  // Strip trailing special tokens
  answer = answer.split("</s>")[0].split("<|")[0].trim();
  return answer;
}

// ── State ───────────────────────────────────────────────────────
let currentMode    = "fast";
let isGenerating   = false;
let hasWebGPU      = false;
const pipelines    = {};         // { fast: pipe | null, smart: pipe | null }
let transformersLib = null;      // loaded once from CDN

// ── DOM ─────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const ui = {
  q:         $("q"),
  askBtn:    $("ask-btn"),
  askLabel:  $("ask-label"),
  qLen:      $("qlen"),
  clearQ:    $("clear-q"),
  modelBar:  $("model-bar"),
  barText:   $("bar-text"),
  barFill:   $("bar-fill"),
  barPct:    $("bar-pct"),
  respWrap:  $("resp-wrap"),
  respBody:  $("resp-body"),
  respDots:  $("resp-dots"),
  respMeta:  $("resp-meta"),
  respIcon:  $("resp-icon"),
  copyBtn:   $("copy-btn"),
  btnFast:   $("btn-fast"),
  btnSmart:  $("btn-smart"),
  gpuPill:   $("gpu-pill"),
};

// ── WebGPU detection ────────────────────────────────────────────
async function detectWebGPU() {
  try {
    if (!("gpu" in navigator)) return;
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      hasWebGPU = true;
      ui.gpuPill.hidden = false;
    }
  } catch { /* not available */ }
}

// ── Load Transformers.js once ───────────────────────────────────
async function loadLibrary() {
  if (transformersLib) return transformersLib;
  transformersLib = await import(TRANSFORMERS_URL);

  // Configure cache + no local models
  transformersLib.env.allowLocalModels = false;
  transformersLib.env.useBrowserCache  = true;

  return transformersLib;
}

// ── Progress callback ───────────────────────────────────────────
function onProgress(ev) {
  showBar(true);
  const { status, progress, loaded, total, file } = ev;

  if (status === "initiate") {
    ui.barText.textContent = `⬇ Fetching ${file ?? "model"}…`;
    setFill(0);
    ui.barPct.textContent  = "";

  } else if (status === "download" || status === "progress") {
    const pct = progress != null
      ? progress
      : total > 0 ? (loaded / total) * 100 : 0;
    setFill(pct);
    if (loaded != null && total != null && total > 0) {
      const mb  = (b) => (b / 1e6).toFixed(1);
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

function setFill(pct) {
  ui.barFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

// ── Load model (lazy) ───────────────────────────────────────────
async function loadModel(mode) {
  if (pipelines[mode]) return pipelines[mode];

  const cfg  = MODELS[mode];
  const lib  = await loadLibrary();

  showBar(true);
  ui.barText.textContent = `🧠 Loading ${cfg.label}…`;
  setFill(4);
  ui.barPct.textContent  = cfg.size;

  // Use WebGPU for smart if available, else wasm
  const device = (mode === "smart" && hasWebGPU) ? "webgpu" : "wasm";

  const pipe = await lib.pipeline(cfg.task, cfg.id, {
    progress_callback: onProgress,
    device,
  });

  pipelines[mode] = pipe;
  showBar(false);
  return pipe;
}

// ── Generate ────────────────────────────────────────────────────
async function generate(question) {
  if (isGenerating || !question.trim()) return;

  isGenerating = true;
  setAskLoading(true);

  // Show response shell immediately with dots
  ui.respWrap.style.display  = "block";
  ui.respBody.textContent    = "";
  ui.respMeta.textContent    = "";
  ui.respDots.style.display  = "flex";
  ui.respIcon.textContent    = MODELS[currentMode].icon;

  const t0 = performance.now();

  try {
    const pipe   = await loadModel(currentMode);
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
    const answer = extractAnswer(output, currentMode);
    const text   = answer || "I couldn't produce a good answer. Try rephrasing your question.";

    ui.respDots.style.display = "none";
    typewrite(ui.respBody, text);

    const words      = text.split(/\s+/).filter(Boolean).length;
    const tokPerSec  = ((words / ms) * 1000).toFixed(1);
    ui.respMeta.textContent =
      `${cfg.label} · ${words} words · ${(ms / 1000).toFixed(2)}s · ~${tokPerSec} tok/s`;

  } catch (err) {
    ui.respDots.style.display = "none";
    const msg = err instanceof Error ? err.message : String(err);
    ui.respBody.innerHTML =
      `<span style="color:#f59e0b">⚠ Error: ${esc(msg)}</span>`;
    console.error("[BrainBoost]", err);
  } finally {
    isGenerating = false;
    setAskLoading(false);
  }
}

// ── Typewriter effect ───────────────────────────────────────────
function typewrite(el, text) {
  el.textContent = "";
  const STEP = 5;
  let i = 0;
  function tick() {
    if (i >= text.length) return;
    el.textContent += text.slice(i, i + STEP);
    i += STEP;
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Mode switching ──────────────────────────────────────────────
function switchMode(mode) {
  if (mode === currentMode) return;
  currentMode = mode;

  ui.btnFast.classList.toggle("tab--active",  mode === "fast");
  ui.btnSmart.classList.toggle("tab--active", mode === "smart");
  ui.btnFast.setAttribute("aria-selected",  String(mode === "fast"));
  ui.btnSmart.setAttribute("aria-selected", String(mode === "smart"));

  // Reset response area
  ui.respWrap.style.display = "none";
  showBar(false);

  // Preload model silently
  loadModel(mode).catch(console.error);
}

// ── UI helpers ──────────────────────────────────────────────────
function showBar(v) {
  ui.modelBar.classList.toggle("model-bar--hidden", !v);
}

function setAskLoading(loading) {
  ui.askBtn.disabled  = loading;
  ui.q.disabled       = loading;
  ui.askLabel.textContent = loading ? "Thinking…" : "Ask";
}

function esc(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function toast(msg) {
  const t = document.createElement("div");
  t.className   = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2200);
}

// ── Neural canvas ───────────────────────────────────────────────
function initCanvas() {
  const canvas = document.getElementById("neural-canvas");
  const ctx    = canvas?.getContext("2d");
  if (!ctx) return;

  const COUNT    = window.innerWidth < 600 ? 24 : 44;
  const MAX_DIST = 130;
  const nodes    = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function spawn() {
    nodes.length = 0;
    for (let i = 0; i < COUNT; i++) {
      nodes.push({
        x:  Math.random() * canvas.width,
        y:  Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.38,
        vy: (Math.random() - 0.5) * 0.38,
        r:  Math.random() * 2 + 0.8,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
      if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx   = nodes[i].x - nodes[j].x;
        const dy   = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(34,211,238,${(1 - dist / MAX_DIST) * 0.3})`;
          ctx.lineWidth   = 0.7;
          ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(168,85,247,0.65)";
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  resize(); spawn(); draw();
  window.addEventListener("resize", () => { resize(); spawn(); });
}

// ── Event listeners ─────────────────────────────────────────────
function initEvents() {
  ui.btnFast.addEventListener("click",  () => switchMode("fast"));
  ui.btnSmart.addEventListener("click", () => switchMode("smart"));

  ui.askBtn.addEventListener("click", () => {
    const q = ui.q.value.trim();
    if (q && !isGenerating) generate(q);
  });

  ui.q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const q = ui.q.value.trim();
      if (q && !isGenerating) generate(q);
    }
  });

  ui.q.addEventListener("input", () => {
    const len = ui.q.value.length;
    ui.qLen.textContent    = `${len} / 800`;
    ui.clearQ.style.display = len > 0 ? "inline-flex" : "none";
    // Auto-resize
    ui.q.style.height = "auto";
    ui.q.style.height = Math.min(ui.q.scrollHeight, 240) + "px";
  });

  ui.clearQ.addEventListener("click", () => {
    ui.q.value             = "";
    ui.qLen.textContent    = "0 / 800";
    ui.clearQ.style.display = "none";
    ui.q.style.height      = "auto";
    ui.q.focus();
  });

  ui.copyBtn.addEventListener("click", async () => {
    const text = ui.respBody.textContent ?? "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      toast("✓ Copied to clipboard");
    } catch {
      toast("Select text manually to copy");
    }
  });
}

// ── Boot ────────────────────────────────────────────────────────
async function boot() {
  await detectWebGPU();
  initCanvas();
  initEvents();

  // Start preloading fast model silently in background
  setTimeout(() => loadModel("fast").catch(console.error), 800);
}

boot();
