// Copies `next build`'s static export into the Python package, where the
// backend serves it from. Run through `npm run build:app`, never on its own.
//
// A script rather than a shell one-liner because `cp -r` and `xcopy` are not
// the same command, and this has to work wherever you happen to be.

import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, "..", "out");
const target = resolve(here, "..", "..", "src", "pos", "static");

if (!existsSync(out)) {
  console.error(`No build found at ${out}.\nRun \`next build\` first -- or use \`npm run build:app\`, which does both.`);
  process.exit(1);
}

// Replace rather than merge: a page you renamed or deleted would otherwise
// linger here forever and still be served.
rmSync(target, { recursive: true, force: true });
cpSync(out, target, { recursive: true });

console.log(`Copied the built UI to ${target}`);
console.log("Commit it -- that folder is what the server actually serves.");
