// BrainBoost AI - app.js
// Compiled from app.ts — runs entirely client-side with Transformers.js
// Fast Mode: Xenova/distilgpt2 | Smart Mode: Xenova/Phi-3-mini-4k-instruct

// ==================== CONFIG ====================

const MODEL_CONFIGS = {
  fast: {
    id: "Xenova/distilgpt2",
    displayName: "distilgpt2",
    description: "Fast mode · 82M params · ~40MB download",
    maxNewTokens: 150,
    temperature: 0.7,
  },
  smart: {
    id: "Xenova/Phi-3-mini-4k-instruct",
    displayName: "Phi-3-mini",
    description: "Smart mode · 3.8B params · ~2.3GB download",
    maxNewTokens: 512,
    temperature: 0.6,
  },
};

// ==================== STATE ====================

const state = {
  currentMode: "fast",
  isGenerating: false,
  modelLoaded: { fast: false, smart: false },
  pipeline: { fast: null, smart: null },
};

// ==================== DOM REFS ====================

const getEl = (id) => document.getElementById(id);

const ui = {
  userInput: getEl("user-input"),
  askBtn: getEl("ask-btn"),
  charCount: getEl("char-count"),
  loadingState: getEl("loading-state"),
  loadingMessage: getEl("loading-message"),
  loadingNote: getEl("loading-note"),
  responseSection: getEl("response-section"),
  responseBody: getEl("response-body"),
  responseMeta: getEl("response-meta"),
  copyBtn: getEl("copy-btn"),
  modelNameDisplay: getEl("model-name-display"),
  modelStatusText: getEl("model-status-text"),
  btnFast: getEl("btn-fast"),
  btnSmart: getEl("btn-smart"),
};

// ==================== HELPERS ====================

const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

function setLoading(msg, note) {
  show(ui.loadingState);
  ui.loadingMessage.textContent = msg;
  ui.loadingNote.textContent = note ?? "";
}

function updateModelStrip(mode, status) {
  const cfg = MODEL_CONFIGS[mode];
  const svgDot = `<svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg>`;
  ui.modelNameDisplay.innerHTML = `${svgDot} ${cfg.displayName}`;
  ui.modelStatusText.textContent = status;
}

function formatDuration(ms) {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

async function detectWebGPU() {
  try {
    if (!("gpu" in navigator)) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

// ==================== MODEL LOADING ====================

async function loadModel(mode) {
  if (state.modelLoaded[mode] && state.pipeline[mode]) return;

  const cfg = MODEL_CONFIGS[mode];
  const hasWebGPU = await detectWebGPU();
  const device = hasWebGPU ? "webgpu" : "wasm";

  setLoading(
    `Loading AI model…`,
    `Downloading ${cfg.displayName} · ${cfg.description}. This may take a moment on first load. Models are cached after download.`
  );
  updateModelStrip(mode, "Downloading…");

  try {
    // Dynamically import Transformers.js from CDN
    const transformers = await import(
      "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js"
    );

    const { pipeline, env } = transformers;

    // Use browser cache, disable local models
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    setLoading(
      `Initializing ${cfg.displayName}…`,
      `Using ${device.toUpperCase()} backend · Models cached after first download`
    );

    const pipelineInstance = await pipeline("text-generation", cfg.id, {
      progress_callback: (p) => {
        if (p.status === "downloading") {
          const pct = p.total > 0 ? Math.round((p.loaded / p.total) * 100) : 0;
          const loadedMB = (p.loaded / 1024 / 1024).toFixed(1);
          const totalMB = (p.total / 1024 / 1024).toFixed(1);
          setLoading(
            `Downloading… ${pct}%`,
            `${cfg.displayName} · ${loadedMB}MB / ${totalMB}MB`
          );
        } else if (p.status === "loading") {
          setLoading(`Loading model weights…`, cfg.description);
        } else if (p.status === "ready") {
          setLoading(`Model ready!`);
        }
      },
    });

    state.pipeline[mode] = pipelineInstance;
    state.modelLoaded[mode] = true;
    hide(ui.loadingState);
    updateModelStrip(mode, `Ready · ${device.toUpperCase()}`);
  } catch (err) {
    hide(ui.loadingState);
    updateModelStrip(mode, "Load failed");
    showError(
      `Failed to load model: ${err?.message ?? "Unknown error"}. Check your internet connection and try again.`
    );
    throw err;
  }
}

// ==================== INFERENCE ====================

async function runInference(prompt, mode) {
  const pipe = state.pipeline[mode];
  const cfg = MODEL_CONFIGS[mode];

  if (!pipe) throw new Error("Model not loaded");

  // Format prompt by mode
  let formattedPrompt;
  if (mode === "smart") {
    formattedPrompt = `<|user|>\n${prompt}<|end|>\n<|assistant|>\n`;
  } else {
    formattedPrompt = `Q: ${prompt}\nA:`;
  }

  const output = await pipe(formattedPrompt, {
    max_new_tokens: cfg.maxNewTokens,
    temperature: cfg.temperature,
    do_sample: true,
    top_k: 50,
    top_p: 0.9,
    repetition_penalty: 1.2,
    return_full_text: false,
  });

  // Extract generated text
  let generated;
  if (Array.isArray(output) && output[0]?.generated_text) {
    generated = output[0].generated_text;
  } else if (typeof output === "string") {
    generated = output;
  } else {
    generated = JSON.stringify(output);
  }

  return cleanOutput(generated, mode);
}

function cleanOutput(text, mode) {
  let cleaned = text.trim();

  // Remove special tokens
  cleaned = cleaned.replace(/<\|.*?\|>/g, "").trim();
  cleaned = cleaned.replace(/^(A:|Assistant:)\s*/i, "").trim();

  // Deduplicate sentences
  const sentences = cleaned.split(/(?<=[.!?])\s+/);
  const seen = new Set();
  const unique = sentences.filter((s) => {
    const normalized = s.toLowerCase().trim();
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
  cleaned = unique.join(" ").trim();

  return (
    cleaned ||
    "I wasn't able to generate a helpful response. Try rephrasing your question."
  );
}

// ==================== UI UPDATES ====================

function showError(msg) {
  // Remove existing errors
  document.querySelector(".error-msg")?.remove();

  const errEl = document.createElement("div");
  errEl.className = "error-msg";
  errEl.textContent = msg;

  const main = document.querySelector(".main-card");
  main?.insertAdjacentElement("afterend", errEl);

  setTimeout(() => errEl.remove(), 10000);
}

function typewriterEffect(el, text, onDone) {
  el.textContent = "";
  el.classList.add("typing-cursor");

  const CHUNK = 4;
  let idx = 0;

  function step() {
    if (idx >= text.length) {
      el.classList.remove("typing-cursor");
      onDone?.();
      return;
    }
    el.textContent += text.slice(idx, idx + CHUNK);
    idx += CHUNK;

    // Auto-scroll response area
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });

    requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

// ==================== MAIN HANDLER ====================

async function handleAsk() {
  const question = ui.userInput.value.trim();
  if (!question || state.isGenerating) return;

  state.isGenerating = true;
  ui.askBtn.disabled = true;
  ui.userInput.disabled = true;

  // Clear previous state
  document.querySelector(".error-msg")?.remove();
  hide(ui.responseSection);

  const startTime = performance.now();

  try {
    // Load model lazily on first use
    if (!state.modelLoaded[state.currentMode]) {
      await loadModel(state.currentMode);
    }

    // Generate response
    setLoading("Thinking…", "Generating your response");
    updateModelStrip(state.currentMode, "Generating…");

    const answer = await runInference(question, state.currentMode);

    const elapsed = performance.now() - startTime;
    hide(ui.loadingState);

    // Display response
    show(ui.responseSection);
    const cfg = MODEL_CONFIGS[state.currentMode];
    ui.responseMeta.textContent = `${cfg.displayName} · ${state.currentMode === "fast" ? "Fast" : "Smart"} mode · ${formatDuration(elapsed)}`;

    typewriterEffect(ui.responseBody, answer, () => {
      updateModelStrip(state.currentMode, "Ready");
    });
  } catch (err) {
    hide(ui.loadingState);
    updateModelStrip(state.currentMode, "Error");
    if (!err?.message?.includes("Failed to load")) {
      showError(`Generation error: ${err?.message ?? "Unknown error"}`);
    }
  } finally {
    state.isGenerating = false;
    ui.askBtn.disabled = false;
    ui.userInput.disabled = false;
    ui.userInput.focus();
  }
}

// ==================== MODE SWITCHING ====================

function switchMode(mode) {
  if (mode === state.currentMode) return;

  state.currentMode = mode;

  ui.btnFast.classList.toggle("active", mode === "fast");
  ui.btnSmart.classList.toggle("active", mode === "smart");

  const loaded = state.modelLoaded[mode];
  updateModelStrip(mode, loaded ? "Ready (cached)" : "Ready to load");

  hide(ui.responseSection);
  hide(ui.loadingState);
  document.querySelector(".error-msg")?.remove();
}

// ==================== EVENTS ====================

function initEventListeners() {
  ui.askBtn.addEventListener("click", () => handleAsk());

  ui.userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  });

  ui.userInput.addEventListener("input", () => {
    ui.charCount.textContent = String(ui.userInput.value.length);
  });

  ui.btnFast.addEventListener("click", () => switchMode("fast"));
  ui.btnSmart.addEventListener("click", () => switchMode("smart"));

  ui.copyBtn.addEventListener("click", async () => {
    const text = ui.responseBody.textContent ?? "";
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      const orig = ui.copyBtn.innerHTML;
      ui.copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7L5 10L12 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Copied!`;
      setTimeout(() => {
        ui.copyBtn.innerHTML = orig;
      }, 2000);
    } catch {
      // Fallback: select text
      const range = document.createRange();
      range.selectNode(ui.responseBody);
      window.getSelection()?.removeAllRanges();
      window.getSelection()?.addRange(range);
    }
  });
}

// ==================== INIT ====================

function init() {
  initEventListeners();
  updateModelStrip("fast", "Ready to load");

  // Detect WebGPU and show in badge
  detectWebGPU().then((hasGPU) => {
    if (hasGPU) {
      const badge = document.querySelector(".header-badge");
      if (badge) {
        badge.innerHTML = `<span class="badge-dot"></span><span>Local · Private · WebGPU ✓</span>`;
      }
      console.log("✅ WebGPU available — GPU acceleration enabled");
    } else {
      console.log("ℹ️ WebGPU not available — using WASM backend");
    }
  });

  console.log("🧠 BrainBoost AI initialized");
  console.log("⚡ Fast mode:", MODEL_CONFIGS.fast.id);
  console.log("🧠 Smart mode:", MODEL_CONFIGS.smart.id);
}

init();
