import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

let generator;
let currentModel;

const models = {
  mini: "Xenova/all-MiniLM-L6-v2",
  tiny: "Xenova/TinyLlama-1.1B-Chat-v1.0",
  phi: "Xenova/phi-2"
};

function detectModel() {
  const ram = navigator.deviceMemory || 4;
  const cores = navigator.hardwareConcurrency || 4;

  if (ram >= 12 && cores >= 8) return "phi";
  if (ram >= 6) return "tiny";
  return "mini";
}

async function loadModel(type) {

  const label = document.getElementById("modelLabel");
  const progress = document.getElementById("progress");

  label.innerText = "Loading AI model...";
  progress.innerText = "Downloading model...";

  currentModel = models[type];

  generator = await pipeline("text-generation", currentModel);

  label.innerText = "Using AI: " + currentModel;
  progress.innerText = "AI ready.";
}

async function init() {

  let selection = document.getElementById("modelSelect").value;

  if (selection === "auto") {
    selection = detectModel();
  }

  await loadModel(selection);
}

async function askAI() {

  if (!generator) {
    alert("AI still loading.");
    return;
  }

  const chat = document.getElementById("chat");
  const input = document.getElementById("prompt");

  const text = input.value;

  if (!text) return;

  chat.innerHTML += `<div class="message user">You: ${text}</div>`;

  input.value = "";

  const result = await generator(text, { max_new_tokens: 80 });

  chat.innerHTML += `<div class="message ai">AI: ${result[0].generated_text}</div>`;

  chat.scrollTop = chat.scrollHeight;
}

document.getElementById("askBtn").addEventListener("click", askAI);

document.getElementById("prompt").addEventListener("keydown", function(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    askAI();
  }
});

document.getElementById("modelSelect").addEventListener("change", init);

init();
