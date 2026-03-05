import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

let generator;
let currentModel;

const models = {
mini:"Xenova/all-MiniLM-L6-v2",
tiny:"Xenova/TinyLlama-1.1B-Chat-v1.0",
phi:"Xenova/phi-2"
};

function detectModel(){

const ram = navigator.deviceMemory || 4;
const cores = navigator.hardwareConcurrency || 4;

if(ram >= 12 && cores >= 8) return "phi";
if(ram >= 6) return "tiny";

return "mini";
}

async function loadModel(type){

const progress = document.getElementById("progress");
const label = document.getElementById("modelLabel");

label.innerText="Loading AI model...";
progress.innerText="Downloading model files...";

currentModel=models[type];

generator = await pipeline("text-generation", currentModel,{
progress_callback:(x)=>{
progress.innerText=`Loading: ${Math.round(x.progress*100)}%`;
}
});

label.innerText="Using AI: "+currentModel;
progress.innerText="AI ready.";
}

async function init(){

let selection=document.getElementById("modelSelect").value;

if(selection==="auto"){
selection=detectModel();
}

await loadModel(selection);
}

document.getElementById("modelSelect").addEventListener("change",init);

document.getElementById("askBtn").onclick=askAI;

async function askAI(){

const chat=document.getElementById("chat");
const prompt=document.getElementById("prompt").value;

chat.innerHTML+=`<div class="message user">You: ${prompt}</div>`;

const result=await generator(prompt,{max_new_tokens:80});

chat.innerHTML+=`<div class="message ai">AI: ${result[0].generated_text}</div>`;

chat.scrollTop=chat.scrollHeight;
}

init();
