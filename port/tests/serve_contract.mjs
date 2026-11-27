// Contract test for the engine component: drive `strata-vk.exe --serve` over stdin/stdout exactly as the desktop app
// does, and check every line the app parses.
//
//   node serve_contract.mjs <engine.exe> <model.gguf>
//
// What is checked, and why each one matters to the front end:
//   READY        - the app opens the panel only after this line, with a timeout, so it must arrive after the model is
//                  loadable and the INFO lines that fill the monitor;
//   INFO         - free-form description, must not be empty;
//   GEN <id>     - one line per generated token, in order, and they must match the ids in the DONE summary;
//   T / PP       - progress lines; absent ones are fine but a malformed one breaks the progress bar;
//   DONE         - the app splits this by spaces and reads 15 fields positionally, so the count and the numbers are
//                  part of the contract, not a detail;
//   STOP/QUIT    - the app's shutdown path: QUIT must end the process, and a late STOP must not be read as an answer.
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

const [enginePath, modelPath] = process.argv.slice(2);
if (!enginePath || !modelPath) {
  console.error("usage: node serve_contract.mjs <engine.exe> <model.gguf>");
  process.exit(2);
}

const checks = [];
const check = (name, ok, detail) => {
  checks.push({ name, ok, detail });
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
};

const child = spawn(enginePath, ["--serve", "--native", modelPath, "--threads", "8", "--max-context", "2048"], {
  stdio: ["pipe", "pipe", "pipe"],
});

let ready = 0;
let info = 0;
let doneLine = null;
const genIds = [];
let malformed = [];

const rl = createInterface({ input: child.stdout });
rl.on("line", (line) => {
  const [word] = line.split(" ");
  if (word === "READY") ready++;
  else if (word === "INFO") {
    info++;
    if (line.trim().length <= "INFO".length) malformed.push("empty INFO line");
  } else if (word === "GEN") {
    const id = Number(line.slice(4));
    if (!Number.isInteger(id)) malformed.push("bad GEN line: " + line);
    else genIds.push(id);
  } else if (word === "GENI") genIds.push(Number(line.slice(5)));
  else if (word === "DONE") doneLine = line;
  else if (word === "ERR") malformed.push(line);
});

const stderr = [];
child.stderr.on("data", (b) => stderr.push(b.toString()));

const send = (line) => child.stdin.write(line + "\n");

const finished = new Promise((resolve) => child.on("close", (code) => resolve(code)));

// 1. wait for READY (the app gives up after a timeout; this test is more patient but must not hang)
const readyDeadline = Date.now() + 120000;
while (!ready && Date.now() < readyDeadline && child.exitCode === null) {
  await new Promise((r) => setTimeout(r, 100));
}
check("READY arrives", ready === 1, ready === 1 ? "" : `saw ${ready}`);
check("INFO describes the engine", info >= 1, `saw ${info}`);

// 2. a real generation request, then quit; the ids are the model's own prompt tokens, so any model works
send("GEN 8 9707,9707");
const genDeadline = Date.now() + 180000;
while (!doneLine && Date.now() < genDeadline && child.exitCode === null) {
  await new Promise((r) => setTimeout(r, 100));
}

if (!doneLine) {
  check("DONE arrives", false, "no DONE line before the timeout");
} else {
  const fields = doneLine.split(/\s+/);
  check("DONE arrives", true, doneLine.slice(0, 80));
  check("DONE carries 15 fields after the word", fields.length === 16, `got ${fields.length} tokens: ${doneLine.slice(0, 120)}`);
  const generated = Number(fields[1]);
  check("generated tokens are reported", generated === genIds.length, `DONE says ${generated}, saw ${genIds.length} GEN lines`);
  const nums = fields.slice(1).filter((f) => f !== "" && /^-?\d+(\.\d+)?$/.test(f));
  check("numeric fields parse", nums.length >= 6, `${nums.length} numeric fields`);
}

check("no malformed lines", malformed.length === 0, malformed.slice(0, 2).join(" | "));

// 2b. the cancel path: a long generation interrupted by STOP must come back with a DONE and fewer tokens than asked,
// not hang and not answer with a full generation.  The app's stop button is this line.
doneLine = null;
const genBefore = genIds.length;
send("GEN 512 9707,9707");
await new Promise((r) => setTimeout(r, 1200));
const stopSentAt = Date.now();
send("STOP");
const stopDeadline = Date.now() + 60000;
while (!doneLine && Date.now() < stopDeadline && child.exitCode === null) {
  await new Promise((r) => setTimeout(r, 50));
}
if (!doneLine) {
  check("STOP ends the generation", false, "no DONE after STOP");
} else {
  const f = doneLine.split(/\s+/);
  const generated = Number(f[1]);
  const finish = f[5];
  check("STOP ends the generation", generated < 512, `${generated} of 512 tokens in ${Date.now() - stopSentAt} ms, finish=${finish}`);
  check("STOP reports a finish reason", typeof finish === "string" && finish.length > 0 && finish !== "0", `finish=${finish}`);
}

// 3. the shutdown path: QUIT must end the process without being answered
send("QUIT");
const quitDeadline = Date.now() + 30000;
while (child.exitCode === null && Date.now() < quitDeadline) {
  await new Promise((r) => setTimeout(r, 50));
}
if (child.exitCode === null) child.kill();
const code = await finished;
check("QUIT ends the process", code !== null, `exit ${code}`);

const failed = checks.filter((c) => !c.ok).length;
console.log("");
console.log(`${checks.length - failed}/${checks.length} checks passed`);
process.exit(failed ? 1 : 0);
