const readline = require("readline");
const { selectMoveFromSnapshot } = require("./patch-export.js");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.on("line", async (line) => {
  try {
    const { color, state, rollouts, trace, greedyProb } = JSON.parse(line);
    const move = await selectMoveFromSnapshot(state, color, rollouts, undefined, trace, greedyProb);
    process.stdout.write(JSON.stringify(move) + "\n");
  } catch (err) {
    process.stdout.write(JSON.stringify({ error: String(err) }) + "\n");
  }
});
