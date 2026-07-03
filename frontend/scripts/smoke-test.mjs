import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");

function read(path) {
  return readFileSync(resolve(root, path), "utf8");
}

function assertIncludes(content, needle, label) {
  if (!content.includes(needle)) {
    throw new Error(`${label} missing '${needle}'`);
  }
}

const app = read("src/App.jsx");
const api = read("src/services/imatchApi.js");
const login = read("src/components/pages/LoginPage.jsx");
const nav = read("src/components/layout/HeaderNavigationBar.jsx");

assertIncludes(app, "LoginPage", "App route wiring");
assertIncludes(app, "FaceSearchExperience", "App route wiring");
assertIncludes(api, "image_base64", "iMatch API JSON/base64 integration");
assertIncludes(api, "Content-Type", "iMatch API JSON integration");
assertIncludes(login, "NexGen", "Login page");
assertIncludes(nav, "Login", "Navigation");

console.log("frontend smoke tests passed");
