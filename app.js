/**
 * BrainBoost AI — app.js
 * ─────────────────────────────────────────────────────────────
 * 100% client-side AI via HuggingFace Transformers.js v2 (Xenova)
 *
 * ⚡ FAST  → Xenova/LaMini-Flan-T5-783M
 *            task: text2text-generation
 *            ~500 MB · LaMini instruction-tuned on top of Flan-T5
 *            prompt: "### Instruction:\n...\n\n### Response:"
 *
 * 🧠 SMART → Xenova/TinyLlama-1.1B-Chat-v1.0
 *            task: text-generation (causal LM)
 *            ~600 MB · Chat-tuned, Zephyr/ChatML template
 *            prompt: <|system|> ... <|user|> ... <|assistant|>
 *
 * Created by Vrishab Varun
 */

// ── CDN ─────────────────────────────────────────────────────────
const CDN = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// ── Model registry ───────────────────────────────────────────────
const MODELS = {
  fast: {
    id:           "Xenova/LaMini-Flan-T5-783M",
    task:         "text2text-generation",
    label:        "LaMini-Flan-T5",
    icon:         "⚡",
    size:         "~500 MB",
    maxNewTokens: 256,
    // LaMini instruction format
    prompt: (q) =>
      `Below is an instruction that describes a task.\n` +
      `Write a response that appropriately completes the request.\n\n` +
      `### Instruction:\n${q}\n\n### Response:`,
    // seq2seq returns only the generated part — no stripping needed
    extract: (out) =>
      (Array.isArray(out) ? out[0]?.generated_text : out?.generated_text) ?? "",
    // Greedy — do_sample MUST be false for seq2seq in Transformers.js v2
    genOpts: (maxNewTokens) => ({
      max_new_tokens: maxNewTokens,
      do_sample:      false,
    }),
  },

  smart: {
    id:           "Xenova/TinyLlama-1.1B-Chat-v1.0",
    task:         "text-generation",
    label:        "TinyLlama Chat",
    icon:         "🧠",
    size:         "~600 MB",
    maxNewTokens: 400,
    // Zephyr / TinyLlama chat template
    prompt: (q) =>
      `<|system|>\nYou are BrainBoost AI — a smart, friendly assistant. ` +
      `Give clear, accurate, well-structured answers. ` +
      `Be concise but complete.</s>\n` +
      `<|user|>\n${q.trim()}</s>\n` +
      `<|assistant|>\n`,
    // Causal LM returns full text — strip the prompt, stop at </s> or next turn
    extract: (out, prompt) => {
      const full = (Array.isArray(out) ? out[0]?.generated_text : out?.generated_text) ?? "";
      // Remove prompt prefix
      let reply = full.startsWith(prompt) ? full.slice(prompt.length) : full;
      // Stop at end token or next user turn
      reply = reply.split("</s>")[0].split("<|user|>")[0].split("<|system|>")[0];
      return reply.trim();
    },
    genOpts: (maxNewTokens) => ({
      max_new_tokens:     maxNewTokens,
      do_sample:          true,
      temperature:        0.7,
      top_k:              40,
      top_p:              0.90,
      repetition_penalty: 1.15,
    }),
  },
};

// ── State ────────────────────────────────────────────────────────
let mode         = "fast";
let generating   = false;
let webGPU       = false;
const pipes      = {};          // cached pipelines
let lib          = null;        // Transformers.js loaded once

// ── DOM ──────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const ui = {
  q:        $("q"),
  askBtn:   $("ask-btn"),
  askLbl:   $("ask-label"),
  qLen:     $("qlen"),
  clearQ:   $("clear-q"),
  bar:      $("model-bar"),
  barTxt:   $("bar-text"),
  barFill:  $("bar-fill"),
  barPct:   $("bar-pct"),
  rWrap:    $("resp-wrap"),
  rBody:    $("resp-body"),
  rDots:    $("resp-dots"),
  rMeta:    $("resp-meta"),
  rIcon:    $("resp-icon"),
  copyBtn:  $("copy-btn"),
  btnFast:  $("btn-fast"),
  btnSmart: $("btn-smart"),
  gpuPill:  $("gpu-pill"),
};

// ── WebGPU ───────────────────────────────────────────────────────
async function checkGPU() {
  try {
    if (!("gpu" in navigator)) return;
    const a = await navigator.gpu.requestAdapter();
    if (a) { webGPU = true; ui.gpuPill.hidden = false; }
  } catch { /* no gpu */ }
}

// ── Load Transformers.js once ────────────────────────────────────
async function getLib() {
  if (lib) return lib;
  lib = await import(CDN);
  lib.env.allowLocalModels = false;
  lib.env.useBrowserCache  = true;
  return lib;
}

// ── Progress ─────────────────────────────────────────────────────
function onProg(e) {
  showBar(true);
  const { status, progress, loaded, total, file } = e;

  if (status === "initiate") {
    ui.barTxt.textContent = `⬇ Downloading ${file ?? "model files"}…`;
    fill(0); ui.barPct.textContent = "";

  } else if (status === "progress" || status === "download") {
    const pct = progress ?? (total > 0 ? (loaded / total) * 100 : 0);
    fill(pct);
    ui.barPct.textContent = (loaded && total)
      ? `${(loaded/1e6).toFixed(0)} / ${(total/1e6).toFixed(0)} MB`
      : pct > 0 ? `${pct.toFixed(0)}%` : "";

  } else if (status === "done") {
    fill(100);
    ui.barTxt.textContent = "✓ Model ready!";
    ui.barPct.textContent = "";
    setTimeout(() => showBar(false), 1100);
  }
}

function fill(pct) {
  ui.barFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

// ── Load pipeline ────────────────────────────────────────────────
async function loadPipe(m) {
  if (pipes[m]) return pipes[m];

  const cfg = MODELS[m];
  const t   = await getLib();

  showBar(true);
  ui.barTxt.textContent = `🧠 Loading ${cfg.label}…`;
  fill(5); ui.barPct.textContent = cfg.size;

  const device = (m === "smart" && webGPU) ? "webgpu" : "wasm";

  const pipe = await t.pipeline(cfg.task, cfg.id, {
    progress_callback: onProg,
    device,
  });

  pipes[m] = pipe;
  showBar(false);
  return pipe;
}

// ── Generate ─────────────────────────────────────────────────────
async function generate(question) {
  if (generating || !question.trim()) return;

  generating = true;
  setBusy(true);

  // Show response container with loading dots
  ui.rWrap.style.display = "block";
  ui.rBody.textContent   = "";
  ui.rMeta.textContent   = "";
  ui.rDots.style.display = "flex";
  ui.rIcon.textContent   = MODELS[mode].icon;

  const t0 = performance.now();

  try {
    const pipe   = await loadPipe(mode);
    const cfg    = MODELS[mode];
    const prompt = cfg.prompt(question);
    const opts   = cfg.genOpts(cfg.maxNewTokens);

    const raw    = await pipe(prompt, opts);
    const ms     = performance.now() - t0;
    const answer = cfg.extract(raw, prompt);
    const text   = clean(answer) || fallback(question);

    ui.rDots.style.display = "none";
    typewrite(ui.rBody, text);

    const wc  = text.split(/\s+/).filter(Boolean).length;
    const tps = ((wc / ms) * 1000).toFixed(1);
    ui.rMeta.textContent =
      `${cfg.label} · ${wc} words · ${(ms/1000).toFixed(1)}s · ~${tps} tok/s`;

  } catch (err) {
    ui.rDots.style.display = "none";
    const msg = err instanceof Error ? err.message : String(err);
    ui.rBody.innerHTML =
      `<span style="color:#f59e0b">⚠ ${esc(msg)}</span>`;
    console.error("[BrainBoost]", err);
  } finally {
    generating = false;
    setBusy(false);
  }
}

// ── Output cleaning ───────────────────────────────────────────────
function clean(text) {
  if (!text) return "";
  let t = text.trim();

  // Strip leftover special tokens
  t = t.replace(/<\|[^|]*\|>/g, "").replace(/<\/s>/g, "");

  // Strip "### Response:" prefix if model echoed it
  t = t.replace(/^#+\s*Response\s*:/i, "").trim();
  t = t.replace(/^(A:|Answer:|Assistant:)\s*/i, "").trim();

  // Collapse 3+ blank lines into 2
  t = t.replace(/\n{3,}/g, "\n\n");

  return t.trim();
}

function fallback(question) {
  const q = question.toLowerCase();
  if (q.includes("hello") || q.includes("hi"))
    return "Hello! I'm BrainBoost AI — ask me anything!";
  return "I wasn't able to generate a good response. Try asking a more specific question.";
}

// ── Typewriter ────────────────────────────────────────────────────
function typewrite(el, text) {
  el.textContent = "";
  const CHUNK = 4;
  let i = 0;
  function tick() {
    if (i >= text.length) return;
    el.textContent += text.slice(i, i + CHUNK);
    i += CHUNK;
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Mode switch ───────────────────────────────────────────────────
function switchMode(m) {
  if (m === mode) return;
  mode = m;
  ui.btnFast.classList.toggle("tab--active",  m === "fast");
  ui.btnSmart.classList.toggle("tab--active", m === "smart");
  ui.btnFast.setAttribute("aria-selected",  String(m === "fast"));
  ui.btnSmart.setAttribute("aria-selected", String(m === "smart"));
  ui.rWrap.style.display = "none";
  showBar(false);
  // Silently preload
  loadPipe(m).catch(console.error);
}

// ── Helpers ───────────────────────────────────────────────────────
function showBar(v) { ui.bar.classList.toggle("model-bar--hidden", !v); }

function setBusy(v) {
  ui.askBtn.disabled = v;
  ui.q.disabled      = v;
  ui.askLbl.textContent = v ? "Thinking…" : "Ask";
}

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function toast(msg) {
  const t = document.createElement("div");
  t.className   = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2000);
}

// ── Neural canvas ─────────────────────────────────────────────────
function initCanvas() {
  const cv  = document.getElementById("neural-canvas");
  const ctx = cv?.getContext("2d");
  if (!ctx) return;

  const N    = window.innerWidth < 600 ? 22 : 42;
  const DIST = 130;
  const pts  = [];

  const resize = () => { cv.width = innerWidth; cv.height = innerHeight; };

  const spawn = () => {
    pts.length = 0;
    for (let i = 0; i < N; i++) pts.push({
      x: Math.random() * cv.width,
      y: Math.random() * cv.height,
      vx: (Math.random() - 0.5) * 0.36,
      vy: (Math.random() - 0.5) * 0.36,
      r:  Math.random() * 1.8 + 0.7,
    });
  };

  const draw = () => {
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > cv.width)  p.vx *= -1;
      if (p.y < 0 || p.y > cv.height) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        const d  = Math.hypot(dx, dy);
        if (d < DIST) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(34,211,238,${(1 - d / DIST) * 0.28})`;
          ctx.lineWidth   = 0.6;
          ctx.stroke();
        }
      }
    }
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(168,85,247,0.6)";
      ctx.fill();
    }
    requestAnimationFrame(draw);
  };

  resize(); spawn(); draw();
  addEventListener("resize", () => { resize(); spawn(); });
}

// ── Events ────────────────────────────────────────────────────────
function initEvents() {
  ui.btnFast.addEventListener("click",  () => switchMode("fast"));
  ui.btnSmart.addEventListener("click", () => switchMode("smart"));

  ui.askBtn.addEventListener("click", () => {
    const q = ui.q.value.trim();
    if (q && !generating) generate(q);
  });

  ui.q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const q = ui.q.value.trim();
      if (q && !generating) generate(q);
    }
  });

  ui.q.addEventListener("input", () => {
    const n = ui.q.value.length;
    ui.qLen.textContent     = `${n} / 800`;
    ui.clearQ.style.display = n > 0 ? "inline-flex" : "none";
    ui.q.style.height       = "auto";
    ui.q.style.height       = Math.min(ui.q.scrollHeight, 240) + "px";
  });

  ui.clearQ.addEventListener("click", () => {
    ui.q.value              = "";
    ui.qLen.textContent     = "0 / 800";
    ui.clearQ.style.display = "none";
    ui.q.style.height       = "auto";
    ui.q.focus();
  });

  ui.copyBtn.addEventListener("click", async () => {
    const t = ui.rBody.textContent ?? "";
    if (!t) return;
    try { await navigator.clipboard.writeText(t); toast("✓ Copied!"); }
    catch { toast("Select text to copy manually"); }
  });
}

// ── Boot ──────────────────────────────────────────────────────────
async function boot() {
  await checkGPU();
  initCanvas();
  initEvents();
  // Silently warm up fast model after 1s
  setTimeout(() => loadPipe("fast").catch(console.error), 1000);
}

boot();
