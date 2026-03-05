import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

let generator;
let currentModel;

const models = {
mini: "Xenova/all-MiniLM-L6-v2",
phi: "Xenova/phi-2"
};

function detectModel(){

const ram = navigator.deviceMemory || 4;
const cores = navigator.hardwareConcurrency || 4;

if(ram >= 8 && cores >= 8){
return "phi";
}

return "mini";
}

async function loadModel(type){

document.getElementById("modelLabel").innerText =
"Loading AI...";

currentModel = models[type];

generator = await pipeline("text-generation", currentModel);

document.getElementById("modelLabel").innerText =
"Using AI: " + currentModel;

}

async function init(){

let selection = document.getElementById("modelSelect").value;

if(selection === "auto"){
selection = detectModel();
}

await loadModel(selection);

}

document.getElementById("modelSelect").addEventListener("change", init);

window.askAI = async function(){

const text = document.getElementById("prompt").value;

const result = await generator(text, {max_new_tokens:50});

document.getElementById("response").innerText =
result[0].generated_text;

};

init();
