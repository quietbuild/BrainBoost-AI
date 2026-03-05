// BrainBoost AI - app.ts
// Runs entirely client-side with Transformers.js
// Supports Fast (distilgpt2) and Smart (Phi-3-mini-4k-instruct) modes

// ==================== TYPES ====================

type AIMode = "fast" | "smart";

interface ModelConfig {
  id: string;
  displayName: string;
  description: string;
  maxNewTokens: number;
  temperature: number;
}

interface AppState {
  currentMode: AIMode;
  isGenerating: boolean;
  modelLoaded: { fast: boolean; smart: boolean };
  pipeline: { fast: any | null; smart: any | null };
}

// ==================== CONFIG ====================

const MODEL_CONFIGS: Record<AIMode, ModelConfig> = {
  fast: {
    id: "Xenova/distilgpt2",
    displayName: "distilgpt2",
    description: "Fast mode · 82M params · Loads ~40MB",
    maxNewTokens: 150,
    temperature: 0.7,
  },
  smart: {
    id: "Xenova/Phi-3-mini-4k-instruct",
    displayName: "Phi-3-mini",
    description: "Smart mode · 3.8B params · Loads ~2.3GB",
    maxNewTokens: 512,
    temperature: 0.6,
  },
};

// ==================== STATE ====================

const state: AppState = {
  currentMode: "fast",
  isGenerating: false,
  modelLoaded: { fast: false, smart: false },
  pipeline: { fast: null, smart: null },
};

// ==================== DOM REFS ====================

const getEl = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const ui = {
  userInput: getEl<HTMLTextAreaElement>("user-input"),
  askBtn: getEl<HTMLButtonElement>("ask-btn"),
  charCount: getEl<HTMLSpanElement>("char-count"),
  loadingState: getEl<HTMLDivElement>("loading-state"),
  loadingMessage: getEl<HTMLSpanElement>("loading-message"),
  loadingNote: getEl<HTMLParagraphElement>("loading-note"),
  responseSection: getEl<HTMLDivElement>("response-section"),
  responseBody: getEl<HTMLDivElement>("response-body"),
  responseMeta: getEl<HTMLDivElement>("response-meta"),
  copyBtn: getEl<HTMLButtonElement>("copy-btn"),
  modelNameDisplay: getEl<HTMLSpanElement>("model-name-display"),
  modelStatusText: getEl<HTMLSpanElement>("model-status-text"),
  btnFast: getEl<HTMLButtonElement>("btn-fast"),
  btnSmart: getEl<HTMLButtonElement>("btn-smart"),
};

// ==================== HELPERS ====================

function show(el: HTMLElement): void {
  el.classList.remove("hidden");
}

function hide(el: HTMLElement): void {
  el.classList.add("hidden");
}

function setLoading(msg: string, note?: string): void {
  show(ui.loadingState);
  ui.loadingMessage.textContent = msg;
  ui.loadingNote.textContent = note ?? "";
}

function updateModelStrip(mode: AIMode, status: string): void {
  const cfg = MODEL_CONFIGS[mode];
  // Rebuild model chip content with SVG dot preserved
  const svgDot = `<svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg>`;
  ui.modelNameDisplay.innerHTML = `${svgDot} ${cfg.displayName}`;
  ui.modelStatusText.textContent = status;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

async function detectWebGPU(): Promise<boolean> {
  try {
    if (!("gpu" in navigator)) return false;
    const adapter = await (navigator as any).gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

// ==================== MODEL LOADING ====================

async function loadModel(mode: AIMode): Promise<void> {
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
    const { pipeline, env } = await import(
      "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js" as any
    );

    // Use WASM backend with browser cache
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    setLoading(
      `Initializing ${cfg.displayName}…`,
      `Using ${device.toUpperCase()} backend · Models cached after first download`
    );

    const pipelineInstance = await pipeline("text-generation", cfg.id, {
      progress_callback: (p: any) => {
        if (p.status === "downloading") {
          const pct =
            p.total > 0 ? Math.round((p.loaded / p.total) * 100) : 0;
          setLoading(
            `Downloading… ${pct}%`,
            `${cfg.displayName} · ${(p.loaded / 1024 / 1024).toFixed(1)}MB / ${(p.total / 1024 / 1024).toFixed(1)}MB`
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
  } catch (err: any) {
    hide(ui.loadingState);
    updateModelStrip(mode, "Load failed");
    showError(
      `Failed to load model: ${err?.message ?? "Unknown error"}. Check your connection and try again.`
    );
    throw err;
  }
}

// ==================== INFERENCE ====================

async function runInference(prompt: string, mode: AIMode): Promise<string> {
  const pipe = state.pipeline[mode];
  const cfg = MODEL_CONFIGS[mode];

  if (!pipe) throw new Error("Model not loaded");

  // Format prompt based on mode
  let formattedPrompt: string;
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
  const generated: string =
    Array.isArray(output) && output[0]?.generated_text
      ? output[0].generated_text
      : typeof output === "string"
      ? output
      : JSON.stringify(output);

  // Clean up output
  return cleanOutput(generated, mode);
}

function cleanOutput(text: string, mode: AIMode): string {
  let cleaned = text.trim();

  // Remove common artifacts
  cleaned = cleaned.replace(/<\|.*?\|>/g, "").trim();
  cleaned = cleaned.replace(/^(A:|Assistant:)\s*/i, "").trim();

  // Remove repetition (basic heuristic)
  const sentences = cleaned.split(/(?<=[.!?])\s+/);
  const seen = new Set<string>();
  const unique = sentences.filter((s) => {
    const normalized = s.toLowerCase().trim();
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
  cleaned = unique.join(" ").trim();

  return cleaned || "I wasn't able to generate a helpful response. Try rephrasing your question.";
}

// ==================== UI UPDATES ====================

function showError(msg: string): void {
  const errEl = document.createElement("div");
  errEl.className = "error-msg";
  errEl.textContent = msg;

  // Remove any existing errors
  const existing = document.querySelector(".error-msg");
  existing?.remove();

  ui.responseSection.parentElement?.insertBefore(errEl, ui.responseSection);
  setTimeout(() => errEl.remove(), 8000);
}

function typewriterEffect(el: HTMLElement, text: string, onDone?: () => void): void {
  el.textContent = "";
  el.classList.add("typing-cursor");

  const CHUNK = 3; // characters per frame
  let idx = 0;

  function step(): void {
    if (idx >= text.length) {
      el.classList.remove("typing-cursor");
      onDone?.();
      return;
    }
    el.textContent += text.slice(idx, idx + CHUNK);
    idx += CHUNK;
    requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

// ==================== MAIN HANDLER ====================

async function handleAsk(): Promise<void> {
  const question = ui.userInput.value.trim();
  if (!question || state.isGenerating) return;

  state.isGenerating = true;
  ui.askBtn.disabled = true;
  ui.userInput.disabled = true;

  // Remove old errors
  document.querySelector(".error-msg")?.remove();
  hide(ui.responseSection);

  const startTime = performance.now();

  try {
    // Load model if needed
    if (!state.modelLoaded[state.currentMode]) {
      await loadModel(state.currentMode);
    }

    // Run inference
    setLoading("Thinking…", "Generating your response");
    updateModelStrip(state.currentMode, "Generating…");

    const answer = await runInference(question, state.currentMode);

    const elapsed = performance.now() - startTime;
    hide(ui.loadingState);

    // Show response
    show(ui.responseSection);
    const cfg = MODEL_CONFIGS[state.currentMode];
    ui.responseMeta.textContent = `${cfg.displayName} · ${state.currentMode === "fast" ? "Fast" : "Smart"} mode · ${formatDuration(elapsed)}`;

    typewriterEffect(ui.responseBody, answer, () => {
      updateModelStrip(state.currentMode, "Ready");
    });
  } catch (err: any) {
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

function switchMode(mode: AIMode): void {
  if (mode === state.currentMode) return;

  state.currentMode = mode;

  // Update buttons
  ui.btnFast.classList.toggle("active", mode === "fast");
  ui.btnSmart.classList.toggle("active", mode === "smart");

  // Update strip
  const loaded = state.modelLoaded[mode];
  updateModelStrip(mode, loaded ? "Ready (cached)" : "Ready to load");

  // Clear response
  hide(ui.responseSection);
  hide(ui.loadingState);
  document.querySelector(".error-msg")?.remove();
}

// ==================== EVENT LISTENERS ====================

function initEventListeners(): void {
  // Ask button
  ui.askBtn.addEventListener("click", () => {
    void handleAsk();
  });

  // Enter key (Shift+Enter for newline)
  ui.userInput.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleAsk();
    }
  });

  // Char counter
  ui.userInput.addEventListener("input", () => {
    ui.charCount.textContent = String(ui.userInput.value.length);
  });

  // Mode buttons
  ui.btnFast.addEventListener("click", () => switchMode("fast"));
  ui.btnSmart.addEventListener("click", () => switchMode("smart"));

  // Copy button
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
      // Fallback
      const range = document.createRange();
      range.selectNode(ui.responseBody);
      window.getSelection()?.removeAllRanges();
      window.getSelection()?.addRange(range);
    }
  });
}

// ==================== INIT ====================

function init(): void {
  initEventListeners();

  // Set initial strip state
  updateModelStrip("fast", "Ready to load");

  // Detect WebGPU and update badge
  detectWebGPU().then((hasGPU) => {
    if (hasGPU) {
      const badge = document.querySelector(".header-badge");
      if (badge) {
        badge.innerHTML = `<span class="badge-dot"></span><span>Local · Private · WebGPU</span>`;
      }
    }
  });

  console.log("🧠 BrainBoost AI initialized");
  console.log("Fast mode:", MODEL_CONFIGS.fast.id);
  console.log("Smart mode:", MODEL_CONFIGS.smart.id);
}

// Boot
init();
