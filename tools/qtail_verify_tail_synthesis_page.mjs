#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright-core";

const args = process.argv.slice(2);
const value = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const pageUrl = value("--page-url", "http://127.0.0.1:54655/qtail-tail-synthesis-model");
const modelRoot = path.resolve(value("--model-root", "/Volumes/ORICO/qtail_tail_synthesis_model"));
const outDir = path.join(modelRoot, "page_qa");
fs.mkdirSync(outDir, { recursive: true });

const training = JSON.parse(fs.readFileSync(path.join(modelRoot, "training_report.json"), "utf8"));
const synthesis = JSON.parse(fs.readFileSync(path.join(modelRoot, "example_output", "synthesis_report.json"), "utf8"));
const selftest = JSON.parse(fs.readFileSync(path.join(modelRoot, "package_selftest.json"), "utf8"));
const errors = [];
const views = [];
const executablePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

if (training.status !== "complete") errors.push("training_report_not_complete");
if (synthesis.status !== "complete") errors.push("synthesis_report_not_complete");
if (selftest.status !== "passed" || selftest.controls_passed !== selftest.controls_total) {
  errors.push("package_selftest_not_passed");
}
if (training.model_sha256 !== synthesis.model_sha256 || training.model_sha256 !== selftest.model_sha256) {
  errors.push("model_hash_binding_mismatch");
}

const browser = await chromium.launch({ headless: true, executablePath });
for (const spec of [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "mobile", width: 390, height: 844 },
]) {
  const context = await browser.newContext({ viewport: { width: spec.width, height: spec.height } });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  const response = await page.goto(pageUrl, { waitUntil: "networkidle", timeout: 30_000 });
  await page.waitForFunction(() => document.querySelectorAll("#allocation-body tr").length === 8, null, { timeout: 15_000 });
  const snapshot = await page.evaluate(() => {
    const body = document.body;
    const candidateRows = document.querySelectorAll("#candidate-list .candidate").length;
    const allocationRows = document.querySelectorAll("#allocation-body tr").length;
    const visibleArtifacts = [...document.querySelectorAll(".artifact")].filter((node) => {
      const style = getComputedStyle(node);
      return style.display !== "none" && style.visibility !== "hidden";
    }).length;
    const clipped = [...document.querySelectorAll(".metric,.stage,.artifact,.fact")]
      .filter((node) => [...node.children].some((child) => child.scrollWidth > child.clientWidth + 2))
      .map((node) => node.className);
    return {
      title: document.title,
      bodyScrollWidth: body.scrollWidth,
      viewportWidth: innerWidth,
      candidateRows,
      allocationRows,
      visibleArtifacts,
      clipped,
      tailGain: document.querySelector("#tail-gain")?.textContent?.trim(),
      sourceTail: document.querySelector("#source-tail-label")?.textContent?.trim(),
      syntheticTail: document.querySelector("#synthetic-tail-label")?.textContent?.trim(),
      productionText: document.body.innerText.includes("Production"),
    };
  });
  if (!response || response.status() !== 200) errors.push(`${spec.name}_http_status`);
  if (snapshot.title !== "Q-Tail 长尾合成模型 | 生产部署") errors.push(`${spec.name}_title`);
  if (snapshot.candidateRows !== 4) errors.push(`${spec.name}_candidate_rows`);
  if (snapshot.allocationRows !== 8) errors.push(`${spec.name}_allocation_rows`);
  if (snapshot.visibleArtifacts < 6) errors.push(`${spec.name}_artifact_links`);
  if (snapshot.bodyScrollWidth > snapshot.viewportWidth + 1) errors.push(`${spec.name}_horizontal_overflow`);
  if (snapshot.clipped.length) errors.push(`${spec.name}_clipped_cards`);
  if (!snapshot.productionText) errors.push(`${spec.name}_production_status`);
  if (snapshot.tailGain !== "+15.52 pp") errors.push(`${spec.name}_tail_gain`);
  if (consoleErrors.length) errors.push(`${spec.name}_console_errors`);
  const screenshot = path.join(outDir, `${spec.name}.png`);
  await page.screenshot({ path: screenshot, fullPage: true });
  views.push({ ...spec, ...snapshot, consoleErrors, screenshot });
  await context.close();
}

const assetPaths = [
  "results/qtail_tail_synthesis_model/training_report.json",
  "results/qtail_tail_synthesis_model/production_model.pt",
  "results/qtail_tail_synthesis_model/example_output/qtail_synthetic_allocation.csv",
  "results/qtail_tail_synthesis_model/example_output/qtail_synthetic_data.csv",
  "results/qtail_tail_synthesis_model/package/README.md",
  "results/qtail_tail_synthesis_model/package_selftest.json",
];
const assets = [];
const pageOrigin = new URL(pageUrl).origin;
for (const relative of assetPaths) {
  const url = new URL(relative, `${pageOrigin}/`).href;
  const response = await fetch(url, { method: "HEAD" });
  assets.push({ relative, status: response.status, contentLength: response.headers.get("content-length") });
  if (response.status !== 200) errors.push(`asset_http_${relative}`);
}
await browser.close();

const report = {
  format_version: "qtail_tail_synthesis_page_qa_v1",
  generated_at: new Date().toISOString(),
  status: errors.length ? "failed" : "passed",
  page_url: pageUrl,
  model_sha256: training.model_sha256,
  training_rows: training.training_rows,
  candidate_count: training.candidate_count,
  synthesis_budget: synthesis.synthetic_budget,
  tail_share_gain_pp: synthesis.tail_share_gain_pp,
  views,
  assets,
  errors,
};
fs.writeFileSync(path.join(modelRoot, "page_qa.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
process.exit(errors.length ? 1 : 0);
