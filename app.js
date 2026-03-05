/**
 * BrainBoost AI — app.js
 * Private on-device chatbot · Created by Vrishab Varun
 *
 * Fast  → Xenova/flan-t5-base              (~250 MB)  text2text-generation
 * Smart → Xenova/TinyLlama-1.1B-Chat-v1.0  (~600 MB)  text-generation
 *
 * Models are quantized ONNX and cached in the browser after first download.
 * No data ever leaves your device.
 */

const TRANSFORMERS_CDN =
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

const MODELS = {
  fast: {
    id:   "Xenova/flan-t5-base",
    task: "text2text-generation",
    name: "Flan-T5",
    icon: "⚡",
    size: "~250 MB",
    maxTokens: 200,
  },
  smart: {
    id:   "Xenova/TinyLlama-1.1B-Chat-v1.0",
    task: "text-generation",
    name: "TinyLlama",
    icon: "🧠",
    size: "~600 MB",
    maxTokens: 300,
  },
};

const MAX_TURNS = 4; // conversation memory (turns kept in context)

// ── State ────────────────────────────────────────────────────────
let mode        = "fast";
let busy        = false;
let hasGPU      = false;
let lib         = null;   // Transformers.js module
const pipes     = {};     // loaded pipelines
const history   = [];     // [{role, content}, ...]

// ── DOM — IDs must exactly match index.html ───────────────────────
const el = {
  chat:     document.getElementById("chat"),
  welcome:  document.getElementById("welcome"),
  msg:      document.getElementById("msg"),         // textarea
  sendBtn:  document.getElementById("send-btn"),
  dlBar:    document.getElementById("dl-bar"),
  dlText:   document.getElementById("dl-text"),
  dlFill:   document.getElementById("dl-fill"),
  dlPct:    document.getElementById("dl-pct"),
  btnFast:  document.getElementById("btn-fast"),
  btnSmart: document.getElementById("btn-smart"),
  gpuBadge: document.getElementById("gpu-badge"),
  clearBtn: document.getElementById("clear-btn"),
  modeHint: document.getElementById("mode-hint"),
};

// ── WebGPU ───────────────────────────────────────────────────────
async function detectGPU() {
  try {
    if (!("gpu" in navigator)) return;
    const a = await navigator.gpu.requestAdapter();
    if (a) { hasGPU = true; el.gpuBadge.hidden = false; }
  } catch { /* unavailable */ }
}

// ── Transformers.js (loaded once from CDN) ───────────────────────
async function getLib() {
  if (lib) return lib;
  lib = await import(TRANSFORMERS_CDN);
  lib.env.allowLocalModels = false;
  lib.env.useBrowserCache  = true;
  return lib;
}

// ── Download progress ─────────────────────────────────────────────
function onProgress(ev) {
  showDL(true);
  const { status, progress, loaded, total, file } = ev;
  if (status === "initiate") {
    el.dlText.textContent = `⬇ Fetching ${file ?? "model"}…`;
    fillDL(0); el.dlPct.textContent = "";
  } else if (status === "download" || status === "progress") {
    const pct = progress != null ? progress : (total > 0 ? (loaded / total) * 100 : 0);
    fillDL(pct);
    el.dlPct.textContent = (loaded && total && total > 0)
      ? `${(loaded/1e6).toFixed(0)} / ${(total/1e6).toFixed(0)} MB`
      : (pct > 0 ? `${pct.toFixed(0)}%` : "");
  } else if (status === "done") {
    fillDL(100);
    el.dlText.textContent = "✓ Ready!";
    el.dlPct.textContent  = "";
    setTimeout(() => showDL(false), 1000);
  }
}
function fillDL(pct) { el.dlFill.style.width = `${Math.min(100, Math.max(0, pct))}%`; }
function showDL(v)   { el.dlBar.classList.toggle("dl-bar--hidden", !v); }

// ── Model loading (lazy + cached) ────────────────────────────────
async function loadPipe(m) {
  if (pipes[m]) return pipes[m];
  const cfg = MODELS[m];
  const api = await getLib();
  showDL(true);
  el.dlText.textContent = `🧠 Loading ${cfg.name}…`;
  fillDL(4); el.dlPct.textContent = cfg.size;
  const device = (m === "smart" && hasGPU) ? "webgpu" : "wasm";
  const pipe   = await api.pipeline(cfg.task, cfg.id, {
    progress_callback: onProgress, device,
  });
  pipes[m] = pipe;
  showDL(false);
  return pipe;
}

// ── Prompt building (includes memory) ────────────────────────────
function buildPrompt(question, m) {
  const hist = history.slice(-(MAX_TURNS * 2));

  if (m === "smart") {
    let p = `<|system|>\nYou are BrainBoost AI, a friendly and helpful private assistant. Give clear, concise, conversational answers.</s>\n`;
    for (const msg of hist) {
      if (msg.role === "user")
        p += `<|user|>\n${msg.content}</s>\n<|assistant|>\n`;
      else
        p += `${msg.content}</s>\n`;
    }
    p += `<|user|>\n${question}</s>\n<|assistant|>\n`;
    return p;
  }

  // Flan-T5 plain-text with optional context
  let ctx = "";
  if (hist.length > 0) {
    ctx = "Previous conversation:\n";
    for (const msg of hist)
      ctx += `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}\n`;
    ctx += "\n";
  }
  return `${ctx}Answer clearly and helpfully: ${question}`;
}

function extractAnswer(out, m) {
  if (m === "fast") {
    return (Array.isArray(out) ? out[0]?.generated_text : out?.generated_text ?? "").trim();
  }
  const full = (Array.isArray(out) ? out[0]?.generated_text : out?.generated_text ?? "").trim();
  const mark = "<|assistant|>\n";
  const i    = full.lastIndexOf(mark);
  let ans    = i !== -1 ? full.slice(i + mark.length) : full;
  ans        = ans.split("</s>")[0].split("<|")[0].trim();
  return ans;
}

// ── Chat UI ───────────────────────────────────────────────────────
function hideWelcome() { el.welcome.style.display = "none"; }

function addBubble(role, text) {
  hideWelcome();
  const group = document.createElement("div");
  group.className = `msg-group msg-group--${role}`;

  const sender = document.createElement("div");
  sender.className = "msg-sender";
  sender.textContent = role === "user"
    ? "You"
    : `${MODELS[mode].icon} BrainBoost · ${MODELS[mode].name}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble bubble--${role}`;

  if (text === "__typing__") {
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  } else {
    bubble.textContent = text;
  }

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = now();

  group.append(sender, bubble, meta);
  el.chat.appendChild(group);
  scrollDown();
  return { bubble, meta };
}

function typewrite(bubble, text, onDone) {
  bubble.innerHTML = "";
  let i = 0;
  function tick() {
    if (i >= text.length) { onDone?.(); return; }
    bubble.textContent += text.slice(i, i + 5);
    i += 5;
    scrollDown();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function scrollDown() { el.chat.scrollTop = el.chat.scrollHeight; }
function now() { return new Date().toLocaleTimeString([], { hour:"2-digit", minute:"2-digit" }); }

// ── Generate ──────────────────────────────────────────────────────
async function generate(question) {
  if (busy || !question.trim()) return;
  busy = true;
  lockInput(true);

  addBubble("user", question);
  history.push({ role: "user", content: question });

  const { bubble: aiBubble, meta: aiMeta } = addBubble("ai", "__typing__");
  const t0 = performance.now();

  try {
    const pipe  = await loadPipe(mode);
    const cfg   = MODELS[mode];
    const prompt = buildPrompt(question, mode);

    // Flan-T5 MUST use greedy (do_sample:false) — sampling crashes on short inputs
    const opts = mode === "fast"
      ? { max_new_tokens: cfg.maxTokens, do_sample: false }
      : {
          max_new_tokens: cfg.maxTokens,
          do_sample: true, temperature: 0.65,
          top_k: 40, top_p: 0.9, repetition_penalty: 1.3,
        };

    const out    = await pipe(prompt, opts);
    const ms     = performance.now() - t0;
    const answer = extractAnswer(out, mode);
    const text   = answer || "Hmm, I couldn't come up with a good answer — try rephrasing?";

    typewrite(aiBubble, text, () => {
      const words = text.split(/\s+/).filter(Boolean).length;
      aiMeta.textContent =
        `${now()} · ${cfg.name} · ${(ms/1000).toFixed(1)}s · ~${((words/ms)*1000).toFixed(0)} tok/s`;
    });

    history.push({ role: "assistant", content: text });
    // Trim old turns
    while (history.length > (MAX_TURNS + 1) * 2) history.splice(0, 2);

  } catch (err) {
    aiBubble.innerHTML = `<span style="color:var(--amber)">⚠ ${esc(err?.message ?? err)}</span>`;
    console.error("[BrainBoost]", err);
  } finally {
    busy = false;
    lockInput(false);
    el.msg.focus();
  }
}

// ── Mode switch ───────────────────────────────────────────────────
function switchMode(m) {
  if (m === mode) return;
  mode = m;
  el.btnFast.classList.toggle("mode-btn--on",  m === "fast");
  el.btnSmart.classList.toggle("mode-btn--on", m === "smart");
  el.btnFast.setAttribute("aria-pressed",  String(m === "fast"));
  el.btnSmart.setAttribute("aria-pressed", String(m === "smart"));
  el.modeHint.textContent = m === "fast"
    ? "⚡ Fast · Flan-T5"
    : "🧠 Smart · TinyLlama";

  // Show a small in-chat notice (only if there are messages)
  if (el.welcome.style.display === "none") {
    const n = document.createElement("div");
    n.className = "mode-notice";
    n.textContent = m === "fast"
      ? "⚡ Switched to Fast mode (Flan-T5)"
      : "🧠 Switched to Smart mode (TinyLlama)";
    el.chat.appendChild(n);
    scrollDown();
  }

  loadPipe(m).catch(console.error);
}

// ── Clear chat ─────────────────────────────────────────────────────
function clearChat() {
  [...el.chat.children].forEach(c => { if (c.id !== "welcome") c.remove(); });
  history.length = 0;
  el.welcome.style.display = "";
  toast("Chat cleared");
}

// ── Helpers ───────────────────────────────────────────────────────
function lockInput(v) {
  el.msg.disabled    = v;
  el.sendBtn.disabled = v;
}
function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2000);
}

// ── Neural canvas ─────────────────────────────────────────────────
function initCanvas() {
  const cv  = document.getElementById("neural-canvas");
  const ctx = cv?.getContext("2d");
  if (!ctx) return;
  const N   = window.innerWidth < 600 ? 20 : 38;
  const D   = 120;
  let nodes = [];
  const resize = () => { cv.width = window.innerWidth; cv.height = window.innerHeight; };
  const spawn  = () => {
    nodes = Array.from({ length: N }, () => ({
      x:  Math.random() * cv.width,  y:  Math.random() * cv.height,
      vx: (Math.random()-.5)*.35,    vy: (Math.random()-.5)*.35,
      r:  Math.random()*1.8+.7,
    }));
  };
  const draw = () => {
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > cv.width)  n.vx *= -1;
      if (n.y < 0 || n.y > cv.height) n.vy *= -1;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i+1; j < nodes.length; j++) {
        const dx = nodes[i].x-nodes[j].x, dy = nodes[i].y-nodes[j].y;
        const d  = Math.sqrt(dx*dx+dy*dy);
        if (d < D) {
          ctx.beginPath(); ctx.moveTo(nodes[i].x,nodes[i].y); ctx.lineTo(nodes[j].x,nodes[j].y);
          ctx.strokeStyle = `rgba(34,211,238,${(1-d/D)*.22})`; ctx.lineWidth = .6; ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fillStyle = "rgba(168,85,247,.5)"; ctx.fill();
    }
    requestAnimationFrame(draw);
  };
  resize(); spawn(); draw();
  window.addEventListener("resize", () => { resize(); spawn(); });
}

// ── Event listeners ───────────────────────────────────────────────
function initEvents() {
  // Mode buttons
  el.btnFast.addEventListener("click",  () => switchMode("fast"));
  el.btnSmart.addEventListener("click", () => switchMode("smart"));

  // Clear
  el.clearBtn.addEventListener("click", clearChat);

  // Send on button click
  el.sendBtn.addEventListener("click", doSend);

  // Send on Enter (Shift+Enter = newline)
  el.msg.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSend();
    }
  });

  // Auto-grow textarea + enable/disable send button
  el.msg.addEventListener("input", () => {
    const hasText = el.msg.value.trim().length > 0;
    el.sendBtn.disabled = !hasText || busy;
    el.msg.style.height = "auto";
    el.msg.style.height = Math.min(el.msg.scrollHeight, 150) + "px";
  });

  // Suggestion chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.q;
      if (q && !busy) generate(q);
    });
  });
}

function doSend() {
  const q = el.msg.value.trim();
  if (!q || busy) return;
  el.msg.value = "";
  el.msg.style.height = "auto";
  el.sendBtn.disabled = true;
  generate(q);
}

// ── Boot ──────────────────────────────────────────────────────────
async function boot() {
  await detectGPU();
  initCanvas();
  initEvents();
  // Start loading fast model silently after 1s
  setTimeout(() => loadPipe("fast").catch(console.error), 1000);
}

boot();
