/**
 * BrainBoost AI — app.js
 * Private on-device chatbot with conversation memory.
 * Fast  → Xenova/flan-t5-base              text2text-generation (~250 MB)
 * Smart → Xenova/TinyLlama-1.1B-Chat-v1.0  text-generation      (~600 MB)
 * Created by Vrishab Varun
 */

const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

const MODELS = {
  fast: {
    id: "Xenova/flan-t5-base",
    task: "text2text-generation",
    label: "Flan-T5",
    icon: "⚡",
    size: "~250 MB",
    maxNewTokens: 200,
  },
  smart: {
    id: "Xenova/TinyLlama-1.1B-Chat-v1.0",
    task: "text-generation",
    label: "TinyLlama",
    icon: "🧠",
    size: "~600 MB",
    maxNewTokens: 300,
  },
};

const MODE_HINTS = {
  fast:  "⚡ Fast mode · Flan-T5",
  smart: "🧠 Smart mode · TinyLlama Chat",
};

// Max conversation turns to keep in context (each turn = 1 user + 1 AI message)
const MAX_HISTORY_TURNS = 4;

// ── State ───────────────────────────────────────────────────────
let currentMode     = "fast";
let isGenerating    = false;
let hasWebGPU       = false;
const pipelines     = {};
let transformersLib = null;

// conversationHistory: [{role:"user"|"assistant", content:string}, ...]
const conversationHistory = [];

// ── DOM ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const ui = {
  chatWindow: $("chat-window"),
  emptyState: $("empty-state"),
  q:          $("q"),
  sendBtn:    $("send-btn"),
  modelBar:   $("model-bar"),
  barText:    $("bar-text"),
  barFill:    $("bar-fill"),
  barPct:     $("bar-pct"),
  btnFast:    $("btn-fast"),
  btnSmart:   $("btn-smart"),
  gpuPill:    $("gpu-pill"),
  clearChat:  $("clear-chat"),
  modeHint:   $("mode-hint"),
};

// ── WebGPU ──────────────────────────────────────────────────────
async function detectWebGPU() {
  try {
    if (!("gpu" in navigator)) return;
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) { hasWebGPU = true; ui.gpuPill.hidden = false; }
  } catch { /* not available */ }
}

// ── Transformers.js library (loaded once) ───────────────────────
async function loadLibrary() {
  if (transformersLib) return transformersLib;
  transformersLib = await import(TRANSFORMERS_URL);
  transformersLib.env.allowLocalModels = false;
  transformersLib.env.useBrowserCache  = true;
  return transformersLib;
}

// ── Download progress ───────────────────────────────────────────
function onProgress(ev) {
  showBar(true);
  const { status, progress, loaded, total, file } = ev;
  if (status === "initiate") {
    ui.barText.textContent = `⬇ Fetching ${file ?? "model"}…`;
    setFill(0); ui.barPct.textContent = "";
  } else if (status === "download" || status === "progress") {
    const pct = progress != null ? progress : (total > 0 ? (loaded / total) * 100 : 0);
    setFill(pct);
    ui.barPct.textContent = (loaded && total)
      ? `${(loaded/1e6).toFixed(0)}/${(total/1e6).toFixed(0)} MB`
      : pct > 0 ? `${pct.toFixed(0)}%` : "";
  } else if (status === "done") {
    setFill(100);
    ui.barText.textContent = "✓ Model ready!";
    ui.barPct.textContent = "";
    setTimeout(() => showBar(false), 1000);
  }
}

function setFill(pct) {
  ui.barFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

// ── Load model (lazy, cached) ───────────────────────────────────
async function loadModel(mode) {
  if (pipelines[mode]) return pipelines[mode];
  const cfg  = MODELS[mode];
  const lib  = await loadLibrary();
  showBar(true);
  ui.barText.textContent = `🧠 Loading ${cfg.label}…`;
  setFill(4);
  ui.barPct.textContent = cfg.size;
  const device = (mode === "smart" && hasWebGPU) ? "webgpu" : "wasm";
  const pipe   = await lib.pipeline(cfg.task, cfg.id, {
    progress_callback: onProgress, device,
  });
  pipelines[mode] = pipe;
  showBar(false);
  return pipe;
}

// ── Prompt builder — includes conversation memory ───────────────
function buildPrompt(question, mode) {
  // Keep last MAX_HISTORY_TURNS turns (2 messages per turn)
  const history = conversationHistory.slice(-(MAX_HISTORY_TURNS * 2));

  if (mode === "smart") {
    // TinyLlama ChatML format with full history
    let prompt =
      `<|system|>\nYou are BrainBoost AI, a friendly and helpful private assistant. ` +
      `Give clear, concise, conversational answers. Remember what was said earlier.</s>\n`;
    for (const msg of history) {
      if (msg.role === "user") {
        prompt += `<|user|>\n${msg.content}</s>\n<|assistant|>\n`;
      } else {
        prompt += `${msg.content}</s>\n`;
      }
    }
    prompt += `<|user|>\n${question}</s>\n<|assistant|>\n`;
    return prompt;
  }

  // Fast mode (Flan-T5 seq2seq) — prepend plain-text context
  let context = "";
  if (history.length > 0) {
    context = "Previous conversation:\n";
    for (const msg of history) {
      context += msg.role === "user"
        ? `User: ${msg.content}\n`
        : `Assistant: ${msg.content}\n`;
    }
    context += "\n";
  }
  return `${context}Answer clearly and helpfully: ${question}`;
}

function extractAnswer(output, mode) {
  if (mode === "fast") {
    return (Array.isArray(output)
      ? output[0]?.generated_text
      : output?.generated_text ?? "").trim();
  }
  // TinyLlama — strip everything before the last <|assistant|>
  const full = (Array.isArray(output)
    ? output[0]?.generated_text
    : output?.generated_text ?? "").trim();
  const marker = "<|assistant|>\n";
  const idx    = full.lastIndexOf(marker);
  let answer   = idx !== -1 ? full.slice(idx + marker.length) : full;
  answer       = answer.split("</s>")[0].split("<|")[0].trim();
  return answer;
}

// ── Chat UI ─────────────────────────────────────────────────────

function setEmptyVisible(v) {
  ui.emptyState.style.display = v ? "flex" : "none";
}

function addBubble(role, initialText) {
  setEmptyVisible(false);

  const group    = document.createElement("div");
  group.className = `msg-group msg-group--${role}`;
  group.dataset.mode = currentMode;

  const sender    = document.createElement("div");
  sender.className = "msg-sender";
  sender.textContent = role === "user"
    ? "You"
    : `${MODELS[currentMode].icon} BrainBoost · ${MODELS[currentMode].label}`;

  const bubble    = document.createElement("div");
  bubble.className = `bubble bubble--${role}`;

  if (initialText === "TYPING") {
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  } else {
    bubble.textContent = initialText;
  }

  const meta    = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = timestamp();

  group.appendChild(sender);
  group.appendChild(bubble);
  group.appendChild(meta);
  ui.chatWindow.appendChild(group);
  scrollToBottom();

  return { group, bubble, meta };
}

function typewriteBubble(bubble, text, onDone) {
  bubble.innerHTML = ""; // clear typing dots
  const STEP = 4;
  let i = 0;
  function tick() {
    if (i >= text.length) { onDone?.(); return; }
    bubble.textContent += text.slice(i, i + STEP);
    i += STEP;
    scrollToBottom();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function scrollToBottom() {
  ui.chatWindow.scrollTop = ui.chatWindow.scrollHeight;
}

function timestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Main generate ───────────────────────────────────────────────
async function generate(question) {
  if (isGenerating || !question.trim()) return;

  isGenerating = true;
  setInputDisabled(true);

  // Show user bubble
  addBubble("user", question);

  // Save to history
  conversationHistory.push({ role: "user", content: question });

  // Show AI typing indicator
  const { bubble: aiBubble, meta: aiMeta } = addBubble("ai", "TYPING");

  const t0 = performance.now();

  try {
    const pipe   = await loadModel(currentMode);
    const cfg    = MODELS[currentMode];
    const prompt = buildPrompt(question, currentMode);

    // Flan-T5 must use greedy (do_sample:false) to avoid "offset out of bounds" crash
    const genOpts = currentMode === "fast"
      ? { max_new_tokens: cfg.maxNewTokens, do_sample: false }
      : {
          max_new_tokens:     cfg.maxNewTokens,
          do_sample:          true,
          temperature:        0.65,
          top_k:              40,
          top_p:              0.90,
          repetition_penalty: 1.3,
        };

    const output = await pipe(prompt, genOpts);
    const ms     = performance.now() - t0;
    const answer = extractAnswer(output, currentMode);
    const text   = answer || "Hmm, I couldn't come up with a good answer. Could you rephrase?";

    // Typewrite the answer
    typewriteBubble(aiBubble, text, () => {
      const words = text.split(/\s+/).filter(Boolean).length;
      aiMeta.textContent =
        `${timestamp()} · ${cfg.label} · ${(ms/1000).toFixed(1)}s · ~${((words/ms)*1000).toFixed(0)} tok/s`;
    });

    // Save AI response to history
    conversationHistory.push({ role: "assistant", content: text });

    // Trim history to avoid context overflow
    while (conversationHistory.length > MAX_HISTORY_TURNS * 2 + 2) {
      conversationHistory.splice(0, 2);
    }

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    aiBubble.innerHTML = `<span style="color:#f59e0b">⚠ ${esc(msg)}</span>`;
    console.error("[BrainBoost]", err);
  } finally {
    isGenerating = false;
    setInputDisabled(false);
    ui.q.focus();
  }
}

// ── Mode switch ─────────────────────────────────────────────────
function switchMode(mode) {
  if (mode === currentMode) return;
  currentMode = mode;
  ui.btnFast.classList.toggle("mode-btn--active",  mode === "fast");
  ui.btnSmart.classList.toggle("mode-btn--active", mode === "smart");
  ui.btnFast.setAttribute("aria-pressed",  String(mode === "fast"));
  ui.btnSmart.setAttribute("aria-pressed", String(mode === "smart"));
  ui.modeHint.textContent = MODE_HINTS[mode];

  // Small mode-change notice in chat (only if chat is active)
  if (ui.emptyState.style.display === "none") {
    const notice = document.createElement("div");
    notice.style.cssText =
      "text-align:center;font-size:.65rem;color:var(--text3);" +
      "font-family:'JetBrains Mono',monospace;padding:4px 0 8px;letter-spacing:.04em;";
    notice.textContent = mode === "fast"
      ? "⚡ Switched to Fast mode (Flan-T5)"
      : "🧠 Switched to Smart mode (TinyLlama)";
    ui.chatWindow.appendChild(notice);
    scrollToBottom();
  }

  loadModel(mode).catch(console.error);
}

// ── Clear chat ──────────────────────────────────────────────────
function clearChat() {
  Array.from(ui.chatWindow.children).forEach((child) => {
    if (child.id !== "empty-state") child.remove();
  });
  conversationHistory.length = 0;
  setEmptyVisible(true);
  toast("Chat cleared");
}

// ── Helpers ─────────────────────────────────────────────────────
function setInputDisabled(v) {
  ui.q.disabled = v;
  ui.sendBtn.disabled = v;
}
function showBar(v) {
  ui.modelBar.classList.toggle("model-bar--hidden", !v);
}
function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2000);
}

// ── Neural canvas ───────────────────────────────────────────────
function initCanvas() {
  const canvas = document.getElementById("neural-canvas");
  const ctx    = canvas?.getContext("2d");
  if (!ctx) return;
  const COUNT = window.innerWidth < 600 ? 20 : 38;
  const MDIST = 120;
  const nodes = [];
  const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
  const spawn  = () => {
    nodes.length = 0;
    for (let i = 0; i < COUNT; i++) nodes.push({
      x:  Math.random() * canvas.width,
      y:  Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r:  Math.random() * 1.8 + 0.7,
    });
  };
  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
      if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < MDIST) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(34,211,238,${(1 - d/MDIST) * 0.25})`;
          ctx.lineWidth   = 0.6;
          ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(168,85,247,0.55)";
      ctx.fill();
    }
    requestAnimationFrame(draw);
  };
  resize(); spawn(); draw();
  window.addEventListener("resize", () => { resize(); spawn(); });
}

// ── Events ──────────────────────────────────────────────────────
function initEvents() {
  ui.btnFast.addEventListener("click",  () => switchMode("fast"));
  ui.btnSmart.addEventListener("click", () => switchMode("smart"));

  // Send
  const doSend = () => {
    const q = ui.q.value.trim();
    if (q && !isGenerating) {
      ui.q.value = "";
      ui.q.style.height = "auto";
      ui.sendBtn.disabled = true;
      generate(q);
    }
  };

  ui.sendBtn.addEventListener("click", doSend);

  ui.q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(); }
  });

  // Auto-grow + enable send
  ui.q.addEventListener("input", () => {
    ui.sendBtn.disabled = ui.q.value.trim().length === 0 || isGenerating;
    ui.q.style.height = "auto";
    ui.q.style.height = Math.min(ui.q.scrollHeight, 160) + "px";
  });

  // Suggestion chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.q;
      if (q && !isGenerating) generate(q);
    });
  });

  ui.clearChat.addEventListener("click", clearChat);
}

// ── Boot ────────────────────────────────────────────────────────
async function boot() {
  await detectWebGPU();
  initCanvas();
  initEvents();
  setEmptyVisible(true);
  // Preload fast model in background after 1s
  setTimeout(() => loadModel("fast").catch(console.error), 1000);
}

boot();
