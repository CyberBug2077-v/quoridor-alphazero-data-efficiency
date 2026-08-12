const fs = require("fs");
const path = require("path");
const util = require("util");
const vm = require("vm");

const BOARD_SIZE = 9;

function makeSandboxConsole() {
  const write = (label, args) => {
    const message = util.format(...args);
    process.stderr.write(`[${label}] ${message}\n`);
  };
  return {
    log: (...args) => write("LOG", args),
    info: (...args) => write("INFO", args),
    warn: (...args) => write("WARN", args),
    error: (...args) => write("ERROR", args),
    debug: (...args) => write("DEBUG", args),
  };
}

const sandbox = {
  console: makeSandboxConsole(),
  Math,
  Date,
  JSON,
  Array,
  Uint8Array,
  Uint16Array,
  Uint32Array,
  Int8Array,
  Int16Array,
  Int32Array,
  Float32Array,
  Float64Array,
  Map,
  Set,
  WeakMap,
  WeakSet,
  Promise,
};

const context = vm.createContext(sandbox);
context.global = context;
context.window = context;
context.self = context;

function loadScript(relativePath) {
  const absolute = path.resolve(__dirname, relativePath);
  const source = fs.readFileSync(absolute, "utf8");
  const script = new vm.Script(source, { filename: relativePath });
  script.runInContext(context);
}

loadScript("src/js/game.js");
loadScript("src/js/ai.js");

const Game = vm.runInContext("Game", context);
const AI = vm.runInContext("AI", context);

const DEFAULT_UCT_CONST = 0.5;

function selectMoveFromSnapshot(snapshot, color, rollouts = 20000, uctConst = DEFAULT_UCT_CONST, trace = false, greedyProb = 0.7) {
  if (!snapshot || typeof snapshot !== "object") {
    throw new Error("snapshot must be an object");
  }
  const jsSnapshot = toJsSnapshot(snapshot);
  const game = makeStateFromSnapshot(jsSnapshot);
  const ai = new AI(rollouts, uctConst, false, false, greedyProb);
  const result = ai.chooseNextMove(game, trace);
  let chosenMove = result;
  let traceInfo = null;
  if (trace && result && result.move) {
    chosenMove = result.move;
    traceInfo = result.trace || null;
  }
  const jsMove = normaliseMove(chosenMove, color);
  const finalMove = fromJsMove(jsMove);
  if (!trace) {
    return finalMove;
  }
  const children = Array.isArray(traceInfo?.children) ? traceInfo.children : [];
  const mappedChildren = children.map((child) => {
    const childMove = fromJsMove(normaliseMove(child.move, color));
    return { move: childMove, numSims: child.numSims };
  });
  return {
    move: finalMove,
    trace: {
      totalSims: traceInfo?.totalSims || 0,
      children: mappedChildren,
      rollout: traceInfo?.rollout || null,
    },
  };
}

module.exports = { selectMoveFromSnapshot };

function makeStateFromSnapshot(snapshot) {
  const game = new Game(false);
  game.board.pawns[0].isHumanPlayer = false;
  game.board.pawns[1].isHumanPlayer = false;
  game.winner = null;

  setPawn(game.board.pawns[0], snapshot.pawn0);
  setPawn(game.board.pawns[1], snapshot.pawn1);

  // Reset wall structures
  reset2D(game.board.walls.horizontal, false);
  reset2D(game.board.walls.vertical, false);
  reset2D(game.openWays.upDown, true);
  reset2D(game.openWays.leftRight, true);
  reset2D(game.validNextWalls.horizontal, true);
  reset2D(game.validNextWalls.vertical, true);
  reset2D(game._probableNextWalls.horizontal, false);
  reset2D(game._probableNextWalls.vertical, false);
  game._probableValidNextWalls = null;
  game._probableValidNextWallsUpdated = false;
  game._validNextPositionsUpdated = false;

  const fenceCounts = snapshot.fenceCounts || {};
  const pawn0Fences = fenceCounts.pawn0 ?? game.board.pawns[0].numberOfLeftWalls;
  const pawn1Fences = fenceCounts.pawn1 ?? game.board.pawns[1].numberOfLeftWalls;

  const horizontalWalls = Array.isArray(snapshot.walls?.h) ? snapshot.walls.h : [];
  horizontalWalls.forEach(({ row, col }) => forcePlace(game, "h", row, col));

  const verticalWalls = Array.isArray(snapshot.walls?.v) ? snapshot.walls.v : [];
  verticalWalls.forEach(({ row, col }) => forcePlace(game, "v", row, col));

  game.board.pawns[0].numberOfLeftWalls = pawn0Fences;
  game.board.pawns[1].numberOfLeftWalls = pawn1Fences;

  game.turn = snapshot.turn ?? 0;
  game._validNextPositionsUpdated = false;
  game._probableValidNextWallsUpdated = false;

  return game;
}

function forcePlace(game, orientation, row, col) {
  const r = Number(row);
  const c = Number(col);
  if (!Number.isFinite(r) || !Number.isFinite(c)) {
    return;
  }

  const prevTurn = game.turn;
  const counts = [
    game.board.pawns[0].numberOfLeftWalls,
    game.board.pawns[1].numberOfLeftWalls,
  ];

  if (orientation === "h") {
    game.placeHorizontalWall(r, c, false);
  } else if (orientation === "v") {
    game.placeVerticalWall(r, c, false);
  } else {
    throw new Error(`Unknown wall orientation "${orientation}"`);
  }

  game.turn = prevTurn;
  game.board.pawns[0].numberOfLeftWalls = counts[0];
  game.board.pawns[1].numberOfLeftWalls = counts[1];
}

function setPawn(pawn, position) {
  if (!position) {
    return;
  }
  pawn.position.row = Number(position.row);
  pawn.position.col = Number(position.col);
}

function reset2D(arr, value) {
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr[i].length; j++) {
      arr[i][j] = value;
    }
  }
}

function normaliseMove(move, color) {
  if (!Array.isArray(move) || move.length !== 3) {
    throw new Error(`Unexpected move format from JS bot: ${JSON.stringify(move)}`);
  }
  if (move[0]) {
    return {
      type: "pawn",
      player: color,
      row: move[0][0],
      col: move[0][1],
    };
  }
  if (move[1]) {
    return {
      type: "fence",
      player: color,
      orientation: "h",
      row: move[1][0],
      col: move[1][1],
    };
  }
  if (move[2]) {
    return {
      type: "fence",
      player: color,
      orientation: "v",
      row: move[2][0],
      col: move[2][1],
    };
  }
  throw new Error(`JS bot returned an empty move: ${JSON.stringify(move)}`);
}

function toJsSnapshot(snapshot) {
  const mirrorPawn = (coord) =>
    coord == null
      ? null
      : {
          row: mirrorSquareRow(Number(coord.row)),
          col: Number(coord.col),
        };

  const normaliseWalls = (walls, orientation) => {
    if (!Array.isArray(walls)) {
      return [];
    }
    return walls
      .map((coord) => {
        if (!coord) {
          return null;
        }
        const row = Number(coord.row);
        const col = Number(coord.col);
        if (!Number.isFinite(row) || !Number.isFinite(col)) {
          return null;
        }
        return {
          row: mirrorWallRow(row, orientation),
          col,
        };
      })
      .filter(Boolean);
  };

  const fencesLeft = snapshot.fences_left || {};

  return {
    turn: mirrorTurn(snapshot.turn ?? 0),
    pawn0: mirrorPawn(snapshot.pawns?.black),
    pawn1: mirrorPawn(snapshot.pawns?.white),
    fenceCounts: {
      pawn0: fencesLeft.black,
      pawn1: fencesLeft.white,
    },
    walls: {
      h: normaliseWalls(snapshot.walls?.h, "h"),
      v: normaliseWalls(snapshot.walls?.v, "v"),
    },
  };
}

function fromJsMove(move) {
  if (!move || typeof move !== "object") {
    return move;
  }
  if (move.type === "pawn") {
    return {
      ...move,
      row: mirrorSquareRow(Number(move.row)),
      col: Number(move.col),
    };
  }
  if (move.type === "fence") {
    return {
      ...move,
      row: mirrorWallRow(Number(move.row), move.orientation),
      col: Number(move.col),
    };
  }
  return move;
}

function mirrorSquareRow(row) {
  return BOARD_SIZE - 1 - row;
}

function mirrorTurn(turn) {
  const t = Number(turn) || 0;
  return t ^ 1;
}

function mirrorWallRow(row, orientation) {
  const r = Number(row);
  if (!Number.isFinite(r)) {
    return r;
  }
  // Wall rows run from 0 to BOARD_SIZE - 2 (8 entries on a 9x9 board)
  if (orientation === "h" || orientation === "v") {
    return BOARD_SIZE - 2 - r;
  }
  return r;
}
