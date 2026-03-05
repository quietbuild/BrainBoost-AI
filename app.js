let mode = "fast"

const models = {
fast:"Xenova/SmolLM2-135M-Instruct",
smart:"Xenova/Phi-3-mini-4k-instruct"
}

let generator

async function loadModel(){
document.getElementById("activeModel").innerText="Loading AI..."

generator = await window.transformers.pipeline(
"text-generation",
models[mode]
)

document.getElementById("activeModel").innerText=
"Active AI: "+(mode==="fast"?"Fast ⚡":"Smart 🧠")
}

function setMode(m){
mode=m
loadModel()
}

function addMessage(text,cls){
const msg=document.createElement("div")
msg.className="message "+cls
msg.innerText=text
document.getElementById("chat").appendChild(msg)
msg.scrollIntoView()
}

async function sendMessage(){

const input=document.getElementById("input")
const text=input.value
if(!text)return

addMessage(text,"user")
input.value=""

addMessage("Thinking...","ai")

const result = await generator(text,{
max_new_tokens:80
})

document.querySelectorAll(".ai").pop().innerText =
result[0].generated_text
}

document.getElementById("send").onclick=sendMessage

document.getElementById("input")
.addEventListener("keypress",e=>{
if(e.key==="Enter") sendMessage()
})

loadModel()
