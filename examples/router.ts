import { Router } from "../src/index.js";

// One router for every language: English text goes to the English checkpoint, everything else to the
// multilingual one. Each checkpoint is downloaded and loaded the first time a message needs it.
// To use local exports instead: new Router({ models: { english: { modelDir: "./onnx" }, multilingual: { modelDir: "./onnx/multilingual" } } })
const router = new Router({
  onProgress: ({ file, received, total }) => {
    if (total) process.stderr.write(`\r${file}: ${((received / total) * 100).toFixed(0)}%   `);
  },
});

const questions = {
  department: {
    type: "choice",
    instructions: "Which team should handle this message?",
    criteria: { billing: "charges, payments, refunds, invoices", shipping: "deliveries, packages, tracking", technical: "app bugs, crashes, errors" },
  },
  angry: { type: "noul", instructions: "Is the customer angry?" },
} as const;

const messages = [
  "I was charged twice on my credit card this month, please give me my money back.",
  "Il pacco doveva arrivare la settimana scorsa e il tracciamento non si aggiorna da giorni.",
  "La aplicación se cierra cada vez que abro la configuración.",
  "Mein Paket ist seit zwei Wochen nicht angekommen. Wo ist es?",
  "这个月我的信用卡被扣了两次款，请退钱给我。",
];

for (const message of messages) {
  const r = await router.systemOne(message, questions);
  console.log(`\n${message}`);
  console.log(`  model: ${r.routing.model} (${r.routing.reason})`);
  console.log(`  department: ${r.answers.department.choice}`, r.answers.department.probabilities);
  console.log(`  angry: ${r.answers.angry.noul}`);
}
await router.close();
