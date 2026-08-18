const readline = require("readline");
const path = require("path");

const externalBridge = path.resolve(
  __dirname,
  "..",
  "..",
  "external",
  "js-mcts",
  "patch-export.js"
);
const { selectMoveFromSnapshot } = require(externalBridge);

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = value + Math.imul(value ^ (value >>> 7), 61 | value) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.on("line", async (line) => {
  const originalRandom = Math.random;
  try {
    const { color, state, rollouts, trace, greedyProb, seed } = JSON.parse(line);
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new Error("seeded JS bridge requires a uint32 seed");
    }
    Math.random = mulberry32(seed);
    const move = await selectMoveFromSnapshot(
      state,
      color,
      rollouts,
      undefined,
      trace,
      greedyProb
    );
    process.stdout.write(JSON.stringify(move) + "\n");
  } catch (err) {
    process.stdout.write(JSON.stringify({ error: String(err) }) + "\n");
  } finally {
    Math.random = originalRandom;
  }
});
