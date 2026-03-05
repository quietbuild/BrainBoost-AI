/**
 * BrainBoost AI — app.ts  (TypeScript source)
 * Runtime file is app.js — no build step needed for GitHub Pages.
 *
 * Fast  → Xenova/LaMini-Flan-T5-783M        text2text-generation · ~400 MB
 * Smart → Xenova/TinyLlama-1.1B-Chat-v1.0   text-generation      · ~600 MB
 *
 * Created by Vrishab Varun
 */

type Role = "user" | "ai";
type Mode = "fast" | "smart";

interface ModelConfig {
  id:           string;
  task:         string;
  label:        string;
  icon:         string;
  size:         string;
  maxNewTokens: number;
  doSample:     boolean;
}

interface Turn { role: Role; text: string; }

const TRANSFORMERS_CDN =
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

const MODELS: Record<Mode, ModelConfig> = {
  fast: {
    id: "Xenova/LaMini-Flan-T5-783M", task: "text2text-generation",
    label: "LaMini-Flan-T5", icon: "⚡", size: "~400 MB",
    maxNewTokens: 220, doSample: false,
  },
  smart: {
    id: "Xenova/TinyLlama-1.1B-Chat-v1.0", task: "text-generation",
    label: "TinyLlama Chat", icon: "🧠", size: "~600 MB",
    maxNewTokens: 380, doSample: true,
  },
};

const PERSONA =
  "You are BrainBoost AI, a friendly, smart, and helpful AI assistant. " +
  "You give clear, accurate, and concise answers. You are warm and conversational. " +
  "You remember the conversation history and refer back to it when relevant.";

const MAX_TURNS = 6;

let currentMode: Mode  = "fast";
let busy               = false;
let hasWebGPU          = false;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let lib: any           = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const pipes: Record<string, any> = {};
const history: Turn[]  = [];

const $ = (id: string) => document.getElementById(id)!;

async function detectWebGPU(): Promise<void> {
  try {
    if (!("gpu" in navigator)) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const a = await (navigator as any).gpu.requestAdapter();
    if (a) { hasWebGPU = true; $("gpu-badge").hidden = false; }
  } catch {}
}

async function getLib(): Promise<unknown> {
  if (lib) return lib;
  lib = await import(TRANSFORMERS_CDN);
  lib.env.allowLocalModels = false;
  lib.env.useBrowserCache  = true;
  return lib;
}

function buildPrompt(userMsg: string, m: Mode): string {
  const recent = history.slice(-MAX_TURNS * 2);
  if (m === "fast") {
    const ctx = recent.map(t =>
      t.role === "user" ? `Human: ${t.text}` : `Assistant: ${t.text}`
    ).join("\n");
    return `${PERSONA}\n\n${ctx ? ctx + "\n" : ""}Human: ${userMsg}\nAssistant:`;
  }
  let prompt = `<|system|>\n${PERSONA}</s>\n`;
  for (const t of recent) {
    prompt += t.role === "user"
      ? `<|user|>\n${t.text}</s>\n<|assistant|>\n`
      : `${t.text}</s>\n`;
  }
  prompt += `<|user|>\n${userMsg}</s>\n<|assistant|>\n`;
  return prompt;
}

// See app.js for full runtime implementation.
// This file is for TypeScript type-checking only (tsc --noEmit).
export {};
