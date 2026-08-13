#!/usr/bin/env node

import {createHash} from "node:crypto";
import {
  copyFile,
  mkdir,
  open,
  readFile,
  rename,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import {existsSync} from "node:fs";
import {spawnSync} from "node:child_process";
import {dirname, join, resolve} from "node:path";
import process from "node:process";
import {pathToFileURL} from "node:url";
import {chromium} from "playwright-core";

const DEFAULT_REPO_ROOT = "/Users/avalok/work/Q-TAIL-MVP";
const DEFAULT_JOB_ROOT = "/Volumes/ORICO/qtail_full_training";
const DEFAULT_PAGE_URL = "http://127.0.0.1:54655/qtail-droid-full-training";
const DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

function parseArgs(argv) {
  const args = {
    repoRoot: DEFAULT_REPO_ROOT,
    jobRoot: DEFAULT_JOB_ROOT,
    pageUrl: DEFAULT_PAGE_URL,
    chrome: DEFAULT_CHROME,
    smoke: false,
    postCommitReadOnly: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--smoke") {
      args.smoke = true;
    } else if (value === "--post-commit-read-only") {
      args.postCommitReadOnly = true;
    } else if (value === "--repo-root") {
      args.repoRoot = resolve(argv[++index]);
    } else if (value === "--job-root") {
      args.jobRoot = resolve(argv[++index]);
    } else if (value === "--page-url") {
      args.pageUrl = argv[++index];
    } else if (value === "--chrome") {
      args.chrome = resolve(argv[++index]);
    } else {
      throw new Error(`unknown argument: ${value}`);
    }
  }
  assert(
    !(args.smoke && args.postCommitReadOnly),
    "--smoke and --post-commit-read-only are mutually exclusive",
  );
  return args;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function atomicWriteJson(path, payload) {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  await rename(temporary, path);
}

async function atomicCopy(path, source) {
  const temporary = `${path}.tmp`;
  await copyFile(source, temporary);
  await rename(temporary, path);
}

async function acquireExclusiveRunLock(path) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const handle = await open(path, "wx", 0o600);
      await handle.writeFile(`${JSON.stringify({
        pid: process.pid,
        created_at: new Date().toISOString(),
      })}\n`, "utf8");
      return async () => {
        await handle.close().catch(() => {});
        await unlink(path).catch(() => {});
      };
    } catch (error) {
      if (!(error instanceof Error) || error.code !== "EEXIST") {
        throw error;
      }
      let ownerPid = null;
      try {
        ownerPid = Number((await readJson(path)).pid);
      } catch {
        ownerPid = null;
      }
      let ownerAlive = false;
      if (Number.isInteger(ownerPid) && ownerPid > 0) {
        try {
          process.kill(ownerPid, 0);
          ownerAlive = true;
        } catch (ownerError) {
          ownerAlive = ownerError?.code === "EPERM";
        }
      }
      if (ownerAlive) {
        throw new Error(
          `postcommit QA is already owned by live PID ${ownerPid}`,
        );
      }
      await unlink(path).catch(() => {});
    }
  }
  throw new Error("could not acquire exclusive postcommit QA run lock");
}

async function sha256(path) {
  const digest = createHash("sha256");
  digest.update(await readFile(path));
  return digest.digest("hex");
}

async function artifactEntry(path) {
  const metadata = await stat(path);
  return {
    path,
    bytes: metadata.size,
    sha256: await sha256(path),
  };
}

async function snapshotProcessLogs({jobRoot, repoRoot, resultRoot}) {
  const snapshotRoot = join(resultRoot, "process_logs_final");
  await mkdir(snapshotRoot, {recursive: true});
  const sources = [
    {
      name: "droid_full_pipeline.log",
      source: join(jobRoot, "logs", "droid_full_pipeline.log"),
      required: true,
      role: "download, checksum, environment, training, and finalization",
    },
    {
      name: "droid_feature_prewarm.log",
      source: join(jobRoot, "logs", "droid_feature_prewarm.log"),
      required: true,
      role: "full-record TFRecord preparse and feature-cache construction",
    },
    {
      name: "pipeline_watchdog.log",
      source: join(jobRoot, "logs", "pipeline_watchdog.log"),
      required: true,
      role: "pipeline liveness and restart supervision",
    },
    {
      name: "progress_loop.log",
      source: join(jobRoot, "logs", "progress_loop.log"),
      required: true,
      role: "minute-by-minute status refresh",
    },
    {
      name: "progress_refresh.log",
      source: join(jobRoot, "logs", "progress_refresh.log"),
      required: true,
      role: "stage-transition status refresh",
    },
    {
      name: "pipeline_generation_handoff.log",
      source: join(jobRoot, "logs", "pipeline_generation_handoff.log"),
      required: true,
      role: "download-to-checksum generation handoff",
    },
    {
      name: "manual_endpoint_generation_handoff.log",
      source: join(jobRoot, "logs", "manual_endpoint_generation_handoff.log"),
      required: true,
      role: "bounded endpoint, HTTP protocol, and worker tuning evidence",
    },
    {
      name: "pipeline_watchdog_status.json",
      source: join(jobRoot, "logs", "pipeline_watchdog_status.json"),
      required: false,
      role: "last watchdog process snapshot",
    },
    {
      name: "qtail_web_services.log",
      source: join(jobRoot, "logs", "qtail-web-services.log"),
      required: true,
      role: "dual-port DROID page supervision and recovery",
    },
    {
      name: "qtail_droid_terminal_launcher.log",
      source: join(jobRoot, "logs", "qtail_droid_terminal_launcher.log"),
      required: false,
      role: "scheduled terminal launcher supervision",
    },
    {
      name: "qtail_droid_launchd_stderr.log",
      source: join(jobRoot, "logs", "qtail_droid_launchd_stderr.log"),
      required: false,
      role: "scheduled launcher stderr history",
    },
    {
      name: "qtail_droid_launchd_stdout.log",
      source: join(jobRoot, "logs", "qtail_droid_launchd_stdout.log"),
      required: false,
      role: "scheduled launcher stdout history",
    },
    {
      name: "qtail_uniclash_guard_stderr.log",
      source: join(jobRoot, "logs", "qtail_uniclash_guard_stderr.log"),
      required: false,
      role: "UniClash transport guard stderr history",
    },
    {
      name: "qtail_uniclash_guard_stdout.log",
      source: join(jobRoot, "logs", "qtail_uniclash_guard_stdout.log"),
      required: false,
      role: "UniClash transport guard stdout history",
    },
    {
      name: "qtail_web_services_local.log",
      source: join(jobRoot, "logs", "qtail_web_services_local.log"),
      required: false,
      role: "local web-service supervision history",
    },
  ];
  const entries = [];
  const missingRequired = [];
  for (const item of sources) {
    if (!existsSync(item.source)) {
      if (item.required) missingRequired.push(item.source);
      continue;
    }
    const destination = join(snapshotRoot, item.name);
    await copyFile(item.source, destination);
    const text = await readFile(destination, "utf8");
    entries.push({
      ...(await artifactEntry(destination)),
      source: item.source,
      role: item.role,
      required: item.required,
      line_count: text.length === 0 ? 0 : text.split("\n").length,
    });
  }
  assert(
    missingRequired.length === 0,
    `required process logs are missing: ${missingRequired.join(", ")}`,
  );
  const manifestPath = join(resultRoot, "droid_process_log_manifest.json");
  await atomicWriteJson(manifestPath, {
    status: "complete",
    generated_at: new Date().toISOString(),
    contract: {
      snapshot_is_immutable: true,
      live_logs_continue_after_snapshot: true,
      required_log_count: sources.filter((item) => item.required).length,
      captured_required_log_count: entries.filter((entry) =>
        sources.some((item) =>
          item.required && item.source === entry.source
        )
      ).length,
      optional_log_count: sources.filter((item) => !item.required).length,
      captured_optional_log_count: entries.filter((entry) =>
        sources.some((item) =>
          !item.required && item.source === entry.source
        )
      ).length,
    },
    missing_required: missingRequired,
    logs: entries,
  });
  return {
    manifestPath,
    artifacts: [manifestPath, ...entries.map((entry) => entry.path)],
    entries,
  };
}

async function mergeArtifactManifest(
  manifestPath,
  {additions, exclusions = [], repoRoot, formalRoot = dirname(manifestPath)},
) {
  assert(repoRoot, "repoRoot is required for formal manifest merging");
  const original = await readJson(manifestPath);
  const byPath = new Map(
    (Array.isArray(original.artifacts) ? original.artifacts : [])
      .map((entry) => [entry.path, entry]),
  );
  for (const path of exclusions) {
    byPath.delete(path);
  }
  await atomicWriteJson(manifestPath, {
    ...original,
    artifacts: [...byPath.values()],
  });
  const command = [
    join(repoRoot, "tools", "qtail_merge_droid_artifact_manifest.py"),
    "--manifest",
    manifestPath,
    "--formal-droid-root",
    formalRoot,
  ];
  for (const path of additions) {
    command.push("--add", path);
  }
  const merged = spawnSync(
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
    command,
    {encoding: "utf8"},
  );
  if (merged.status !== 0) {
    await atomicWriteJson(manifestPath, original);
    throw new Error(
      `formal artifact manifest merge failed: ${
        merged.stderr || merged.stdout
      }`,
    );
  }
}

async function probeUrl(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(url, {
      method: "HEAD",
      cache: "no-store",
      signal: controller.signal,
    });
    return {url, status: response.status, ok: response.status === 200};
  } finally {
    clearTimeout(timeout);
  }
}

async function collectArtifactUrls(browser, pageUrl) {
  const context = await browser.newContext({
    viewport: {width: 1280, height: 900},
    locale: "zh-CN",
  });
  try {
    const page = await context.newPage();
    await page.goto(pageUrl, {waitUntil: "networkidle", timeout: 30_000});
    await page.waitForFunction(
      () =>
        !document.querySelector("#completion-count")?.textContent?.startsWith("0 /")
        && document.querySelectorAll(".formal-artifact-row").length >= 64,
      undefined,
      {timeout: 20_000},
    );
    return await page.locator(
      "a.artifact, .formal-artifact-row:not(.wait) .formal-artifact-link",
    ).evaluateAll((links) =>
      links.map((link) => link.href)
    );
  } finally {
    await context.close();
  }
}

async function inspectViewport(browser, {
  pageUrl,
  viewport,
  expectedCompletion = null,
  expectedStatus = null,
  requireIntermediate = false,
  requireResults = false,
  requireCommitted = false,
  screenshotPath = null,
}) {
  const context = await browser.newContext({
    viewport,
    deviceScaleFactor: 1,
    locale: "zh-CN",
  });
  const page = await context.newPage();
  const consoleErrors = [];
  const pageErrors = [];
  const failedResponses = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) {
      failedResponses.push(`${response.status()} ${response.url()}`);
    }
  });

  await page.goto(`${pageUrl}?qa=${Date.now()}`, {
    waitUntil: "networkidle",
    timeout: 30_000,
  });
  await page.waitForFunction(
    () => {
      const value = document.querySelector("#completion-count")?.textContent || "";
      return !value.startsWith("0 /");
    },
    undefined,
    {timeout: 20_000},
  );
  if (expectedCompletion) {
    await page.waitForFunction(
      (expected) => document.querySelector("#completion-count")?.textContent?.trim() === expected,
      expectedCompletion,
      {timeout: 20_000},
    );
  }
  if (expectedStatus) {
    await page.waitForFunction(
      (expected) => document.querySelector("#status-text")?.textContent?.trim() === expected,
      expectedStatus,
      {timeout: 20_000},
    );
  }

  const snapshot = await page.evaluate(() => ({
    title: document.title,
    h1: document.querySelector("h1")?.textContent?.trim() || "",
    completion: document.querySelector("#completion-count")?.textContent?.trim() || "",
    status: document.querySelector("#status-text")?.textContent?.trim() || "",
    capacityGate: document.querySelector("#capacity-gate")?.textContent?.trim() || "",
    capacityAudit: document.querySelector("#audit-capacity")?.textContent?.trim() || "",
    migrationState: document.querySelector("#openx-migration-state")?.textContent?.trim() || "",
    overviewBoundary: document.querySelector(".overview .claim-banner")?.textContent?.trim() || "",
    claimBoundary: document.querySelector("#claim-boundary-card")?.textContent?.trim() || "",
    statisticalBoundary: document.querySelector("#statistical-boundary-card")?.textContent?.trim() || "",
    holdoutAudit: document.querySelector("#holdout-audit-card")?.textContent?.trim() || "",
    historySummary: document.querySelector("#history-summary")?.textContent?.trim() || "",
    historyCanvas: (() => {
      const canvas = document.querySelector("#history-chart");
      if (!canvas) return null;
      const context = canvas.getContext("2d");
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let coloredSamples = 0;
      for (let index = 0; index < pixels.length; index += 64) {
        const red = pixels[index];
        const green = pixels[index + 1];
        const blue = pixels[index + 2];
        if (
          (green > 120 && blue > 90)
          || (red > 150 && green > 80)
        ) coloredSamples += 1;
      }
      return {
        width: canvas.width,
        height: canvas.height,
        pointCount: Number(canvas.dataset.pointCount || 0),
        sourceSampleCount: Number(canvas.dataset.sourceSampleCount || 0),
        coloredSamples,
      };
    })(),
    featureShards: document.querySelector("#feature-shards")?.textContent?.trim() || "",
    featureRecords: document.querySelector("#feature-records")?.textContent?.trim() || "",
    featureBytes: document.querySelector("#feature-bytes")?.textContent?.trim() || "",
    featureScanRate: document.querySelector("#feature-scan-rate")?.textContent?.trim() || "",
    featureErrors: document.querySelector("#feature-errors")?.textContent?.trim() || "",
    featureCache: document.querySelector("#feature-cache")?.textContent?.trim() || "",
    checkpointGridSummary: document.querySelector("#checkpoint-grid-summary")?.textContent?.trim() || "",
    checkpointGridRows: [...document.querySelectorAll("#checkpoint-grid-rows tr")].map(
      (row) => [...row.querySelectorAll("td")].map((cell) => cell.textContent?.trim() || ""),
    ),
    checksumVerified: document.querySelector("#checksum-verified")?.textContent?.trim() || "",
    uniclashIsolation: document.querySelector("#uniclash-isolation")?.textContent?.trim() || "",
    uniclashIsolationNote: document.querySelector("#uniclash-isolation-note")?.textContent?.trim() || "",
    dataContinuityAudit: document.querySelector("#audit-data-continuity")?.textContent?.trim() || "",
    protocolAudit: document.querySelector("#audit-protocol")?.textContent?.trim() || "",
    trainingOrderAudit: document.querySelector("#audit-training-order")?.textContent?.trim() || "",
    downloadMarkerAudit: document.querySelector("#audit-download-marker")?.textContent?.trim() || "",
    singleWriterAudit: document.querySelector("#audit-single-writer")?.textContent?.trim() || "",
    runtimeProcessAudit: document.querySelector("#audit-runtime-process")?.textContent?.trim() || "",
    finalSealAudit: document.querySelector("#audit-final-seal")?.textContent?.trim() || "",
    checksumVpnAudit: document.querySelector("#audit-checksum-vpn")?.textContent?.trim() || "",
    incrementalClosure: document.querySelector("#audit-incremental-closure")?.textContent?.trim() || "",
    releaseMilestones: document.querySelector("#audit-release-milestones")?.textContent?.trim() || "",
    runtimeHandoff: document.querySelector("#runtime-handoff")?.textContent?.trim() || "",
    runtimeGeneration: document.querySelector("#runtime-generation")?.textContent?.trim() || "",
    runtimePrewarm: document.querySelector("#runtime-prewarm")?.textContent?.trim() || "",
    runtimeSupervision: document.querySelector("#runtime-supervision")?.textContent?.trim() || "",
    runtimeMount: document.querySelector("#runtime-mount")?.textContent?.trim() || "",
    preflightInput: document.querySelector("#preflight-input")?.textContent?.trim() || "",
    preflightRecords: document.querySelector("#preflight-records")?.textContent?.trim() || "",
    preflightDevice: document.querySelector("#preflight-device")?.textContent?.trim() || "",
    preflightCompute: document.querySelector("#preflight-compute")?.textContent?.trim() || "",
    preflightResume: document.querySelector("#preflight-resume")?.textContent?.trim() || "",
    preflightResumeNote: document.querySelector("#preflight-resume-note")?.textContent?.trim() || "",
    preflightGate: document.querySelector("#preflight-gate")?.textContent?.trim() || "",
    preflightBoundary: document.querySelector("#preflight-boundary")?.textContent?.trim() || "",
    forecastInput: document.querySelector("#forecast-input")?.textContent?.trim() || "",
    forecastTail: document.querySelector("#forecast-tail")?.textContent?.trim() || "",
    forecastTailNote: document.querySelector("#forecast-tail-note")?.textContent?.trim() || "",
    forecastExtreme: document.querySelector("#forecast-extreme")?.textContent?.trim() || "",
    forecastRare: document.querySelector("#forecast-rare")?.textContent?.trim() || "",
    forecastRareNote: document.querySelector("#forecast-rare-note")?.textContent?.trim() || "",
    forecastBoundary: document.querySelector("#forecast-boundary")?.textContent?.trim() || "",
    canaryInput: document.querySelector("#canary-input")?.textContent?.trim() || "",
    canaryRecords: document.querySelector("#canary-records")?.textContent?.trim() || "",
    canaryCompute: document.querySelector("#canary-compute")?.textContent?.trim() || "",
    canaryDevice: document.querySelector("#canary-device")?.textContent?.trim() || "",
    canaryResume: document.querySelector("#canary-resume")?.textContent?.trim() || "",
    canaryHashes: document.querySelector("#canary-hashes")?.textContent?.trim() || "",
    canaryTail: document.querySelector("#canary-tail")?.textContent?.trim() || "",
    canaryTailNote: document.querySelector("#canary-tail-note")?.textContent?.trim() || "",
    canaryExtreme: document.querySelector("#canary-extreme")?.textContent?.trim() || "",
    canaryRare: document.querySelector("#canary-rare")?.textContent?.trim() || "",
    canaryRareNote: document.querySelector("#canary-rare-note")?.textContent?.trim() || "",
    canaryBoundary: document.querySelector("#canary-boundary")?.textContent?.trim() || "",
    trainedShards: document.querySelector("#trained-shards")?.textContent?.trim() || "",
    parseRate: document.querySelector("#parse-rate")?.textContent?.trim() || "",
    tailGain: document.querySelector("#tail-gain")?.textContent?.trim() || "",
    tailCi: document.querySelector("#tail-ci")?.textContent?.trim() || "",
    hypothesisGate: document.querySelector("#hypothesis-gate")?.textContent?.trim() || "",
    hypothesisGateNote: document.querySelector("#hypothesis-gate-note")?.textContent?.trim() || "",
    resultBoundary: document.querySelector("#result-boundary")?.textContent?.trim() || "",
    rareCoverageBoundary: document.querySelector("#rare-coverage-boundary")?.textContent?.trim() || "",
    rareCoverageSummary: document.querySelector("#rare-coverage-summary")?.textContent?.trim() || "",
    rareCoverageRows: [...document.querySelectorAll("#rare-coverage-rows tr")].map(
      (row) => [...row.querySelectorAll("td")].map((cell) => cell.textContent?.trim() || ""),
    ),
    valueStatusNote: document.querySelector("#value-status-note")?.textContent?.trim() || "",
    valueEvidenceState: document.querySelector("#value-evidence-state")?.textContent?.trim() || "",
    valueEvidenceNote: document.querySelector("#value-evidence-note")?.textContent?.trim() || "",
    valueTechnicalState: document.querySelector("#value-technical-state")?.textContent?.trim() || "",
    valueTechnicalNote: document.querySelector("#value-technical-note")?.textContent?.trim() || "",
    valueCommercialState: document.querySelector("#value-commercial-state")?.textContent?.trim() || "",
    valueCommercialNote: document.querySelector("#value-commercial-note")?.textContent?.trim() || "",
    valueBoundary: document.querySelector("#value-boundary")?.textContent?.trim() || "",
    artifactStates: [...document.querySelectorAll("a.artifact")].map((link) => ({
      href: link.getAttribute("href") || "",
      state: link.querySelector(".artifact-state")?.textContent?.trim() || "",
      waiting: link.classList.contains("waiting"),
      title: link.getAttribute("title") || "",
    })),
    formalArtifactSummary: document.querySelector("#formal-artifact-summary")?.textContent?.trim() || "",
    formalArtifactContract: (() => {
      const ledger = document.querySelector("#formal-artifact-ledger");
      return {
        requiredCount: Number(ledger?.dataset.requiredCount || 0),
        sealedCount: Number(ledger?.dataset.sealedCount || 0),
        generatedCount: Number(ledger?.dataset.generatedCount || 0),
        waitCount: Number(ledger?.dataset.waitCount || 0),
        rows: [...document.querySelectorAll(".formal-artifact-row")].map((row) => ({
          path: row.dataset.path || "",
          href: row.querySelector(".formal-artifact-link")?.getAttribute("href") || "",
          state: row.dataset.state || "",
          badge: row.querySelector(".formal-artifact-state")?.textContent?.trim() || "",
        })),
      };
    })(),
    innerWidth: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }));

  assert(snapshot.title.includes("Q-Tail DROID 全量训练"), "unexpected page title");
  assert(snapshot.h1.includes("DROID 全量数据"), "unexpected page heading");
  assert(
    Math.max(snapshot.scrollWidth, snapshot.bodyScrollWidth) <= snapshot.innerWidth + 1,
    `horizontal overflow at ${viewport.width}px`,
  );
  assert(consoleErrors.length === 0, `browser console errors: ${consoleErrors.join(" | ")}`);
  assert(pageErrors.length === 0, `browser page errors: ${pageErrors.join(" | ")}`);
  assert(failedResponses.length === 0, `failed page resources: ${failedResponses.join(" | ")}`);
  assert(
    snapshot.overviewBoundary.includes("droid_policy_learning") &&
      snapshot.overviewBoundary.includes("仅按固定提交封存为复现参考") &&
      snapshot.overviewBoundary.includes("不参与本次 AllocationHead 优化"),
    `official policy-backend usage boundary is missing: ${snapshot.overviewBoundary}`,
  );
  assert(
    snapshot.hypothesisGateNote.includes("gain ≥ 2 pp") &&
      snapshot.hypothesisGateNote.includes("CI low ≥ 2 pp") &&
      snapshot.hypothesisGateNote.includes("extreme reduction > 0") &&
      snapshot.hypothesisGateNote.includes("结果方向不影响实验完成"),
    `visible preregistered outcome threshold is inaccurate: ${snapshot.hypothesisGateNote}`,
  );
  if (!requireResults) {
    assert(
      snapshot.valueEvidenceState === "WITHHELD" &&
        snapshot.valueTechnicalState === "等待正式裁决" &&
        snapshot.valueCommercialState === "不可主张效果",
      `unfinished page leaked a technical or commercial claim: ${
        JSON.stringify({
          evidence: snapshot.valueEvidenceState,
          technical: snapshot.valueTechnicalState,
          commercial: snapshot.valueCommercialState,
        })
      }`,
    );
    assert(
      snapshot.valueStatusNote.includes("WITHHELD") &&
        snapshot.valueStatusNote.includes("ROI") &&
        snapshot.valueEvidenceNote.includes("4,102") &&
        snapshot.valueEvidenceNote.includes("4,096") &&
        snapshot.valueEvidenceNote.includes("187,891"),
      `unfinished value gate is incomplete: ${snapshot.valueStatusNote} | ${
        snapshot.valueEvidenceNote
      }`,
    );
    assert(
      snapshot.valueBoundary.includes("AllocationHead") &&
        snapshot.valueBoundary.includes("policy 成功率") &&
        snapshot.valueBoundary.includes("ROI") &&
        snapshot.valueBoundary.includes("不构成"),
      `technical/commercial claim boundary is incomplete: ${
        snapshot.valueBoundary
      }`,
    );
  }
  if (requireIntermediate) {
    assert(
      snapshot.capacityGate.includes("净余量") &&
        snapshot.capacityGate.includes("通过"),
      `ORICO capacity headroom is missing or failed: ${snapshot.capacityGate}`,
    );
    assert(
      snapshot.capacityAudit.includes("5% 安全余量") &&
        snapshot.capacityAudit.includes("净余量"),
      `ORICO capacity audit is incomplete: ${snapshot.capacityAudit}`,
    );
    assert(
      snapshot.migrationState.includes("数据、代码和结果目录均已驻留 ORICO") &&
        snapshot.migrationState.includes("不重复下载"),
      `ORICO residency evidence is incomplete: ${snapshot.migrationState}`,
    );
    assert(
      snapshot.holdoutAudit.includes("seed=11") &&
        snapshot.holdoutAudit.includes("820-shard") &&
        snapshot.holdoutAudit.includes("16781c97…8a767") &&
        snapshot.holdoutAudit.includes("划分不因结果改变"),
      `preregistered holdout boundary is missing: ${snapshot.holdoutAudit}`,
    );
    assert(
      snapshot.overviewBoundary.includes("不是独立因果证据") &&
        snapshot.overviewBoundary.includes("不冒充机器人 policy 成功率") &&
        snapshot.claimBoundary.includes("target 与裁决 taxonomy 同源") &&
        snapshot.claimBoundary.includes("Policy tail success") &&
        snapshot.claimBoundary.includes("不等同于外部时间戳或 WORM") &&
        snapshot.statisticalBoundary.includes("不是有效 p 值") &&
        snapshot.statisticalBoundary.includes("不参与完成或支持门禁") &&
        snapshot.statisticalBoundary.includes("不覆盖训练 seed"),
      "claim or statistical boundary regressed",
    );
    assert(
      snapshot.historySummary.includes("retained samples") &&
        snapshot.historySummary.includes("uniform_index_plus_stage_boundaries_v1"),
      `bounded history summary is missing: ${snapshot.historySummary}`,
    );
    assert(
      snapshot.historyCanvas &&
        snapshot.historyCanvas.pointCount >= 2 &&
        snapshot.historyCanvas.pointCount <= 240 &&
        snapshot.historyCanvas.sourceSampleCount >=
          snapshot.historyCanvas.pointCount &&
        snapshot.historyCanvas.width > 0 &&
        snapshot.historyCanvas.height > 0 &&
        snapshot.historyCanvas.coloredSamples > 20,
      `history canvas is blank or unbound: ${JSON.stringify(snapshot.historyCanvas)}`,
    );
    assert(snapshot.featureShards !== "等待下载", "feature-shard progress is still waiting");
    assert(snapshot.featureRecords !== "等待下载", "decoded-episode progress is still waiting");
    assert(snapshot.featureBytes !== "等待下载", "represented-byte progress is still waiting");
    assert(snapshot.featureScanRate.includes("full scans"), "full-scan progress is missing");
    assert(snapshot.featureErrors.includes("parse errors"), "parse-error progress is missing");
    assert(
      snapshot.featureErrors.includes("official shardLengths match"),
      "official record-count audit is missing",
    );
    assert(
      snapshot.featureCache.includes("valid cache") &&
        snapshot.featureCache.includes("excluded stale") &&
        snapshot.featureCache.includes("training input manifest only") &&
        !snapshot.featureCache.includes("BLOCKED"),
      `feature-cache selection boundary is missing: ${snapshot.featureCache}`,
    );
    const expectedCheckpointState = snapshot.checkpointGridSummary.startsWith(
      "20 / 20 VERIFIED",
    )
      ? "VERIFIED"
      : snapshot.checkpointGridSummary.startsWith("20 / 20 SAVED")
        ? "SAVED"
        : "WAIT";
    assert(
      snapshot.checkpointGridRows.length === 4 &&
        JSON.stringify(
          snapshot.checkpointGridRows.map((row) => row[0]),
        ) === JSON.stringify([
          "Evaluation · Source",
          "Evaluation · Q-Tail",
          "Deployment · Source",
          "Deployment · Q-Tail",
        ]) &&
        snapshot.checkpointGridRows.every((row) =>
          row.length === 6 &&
          row.slice(1).every((state) =>
            ["WAIT", "SAVED", "VERIFIED", "MISSING"].includes(state)
          ) &&
          row.slice(1).every(
            (state) => state === expectedCheckpointState,
          )
        ),
      `formal checkpoint grid is malformed: ${
        JSON.stringify(snapshot.checkpointGridRows)
      }`,
    );
    assert(
      snapshot.checksumVerified.includes("/ 4,102"),
      "official MD5 verification progress is missing",
    );
    assert(
      snapshot.uniclashIsolation === "PASS · Core ON · TUN OFF",
      `UniClash isolation gate failed: ${snapshot.uniclashIsolation}`,
    );
    assert(
      snapshot.uniclashIsolationNote.includes("live blocked 0") &&
        /cumulative blocked \d+/.test(snapshot.uniclashIsolationNote) &&
        snapshot.uniclashIsolationNote.includes("timeline") &&
        snapshot.uniclashIsolationNote.includes("clean") &&
        snapshot.uniclashIsolationNote.includes("runtime process anomalies") &&
        snapshot.uniclashIsolationNote.includes("VPN route violations 0"),
      `UniClash guard has blocked transfers: ${snapshot.uniclashIsolationNote}`,
    );
    assert(
      snapshot.uniclashIsolationNote.includes("守护前路由证据缺失") &&
        snapshot.uniclashIsolationNote.includes("classifier gap epochs 1"),
      `transport evidence boundary is not visible: ${snapshot.uniclashIsolationNote}`,
    );
    assert(
      snapshot.runtimeMount.includes("/Volumes/ORICO") &&
        snapshot.runtimeMount.includes("真实 APFS mountpoint") &&
        snapshot.runtimeMount.includes("sleep 0") &&
        snapshot.runtimeMount.includes("disk sleep 0") &&
        snapshot.runtimeMount.includes("ExternalMedia ON") &&
        snapshot.runtimeMount.includes("跨天运行门禁通过") &&
        !snapshot.runtimeMount.includes("BLOCKED"),
      `ORICO long-run power gate is missing: ${snapshot.runtimeMount}`,
    );
    assert(
      snapshot.dataContinuityAudit.startsWith("PASS ·") &&
        snapshot.dataContinuityAudit.includes("已验证对象下降 0") &&
        snapshot.dataContinuityAudit.includes("已提交缓存计数下降 0") &&
        snapshot.dataContinuityAudit.includes("非数据丢失"),
      `committed data continuity is unclear or failed: ${snapshot.dataContinuityAudit}`,
    );
    assert(
      snapshot.artifactStates.length >= 90 &&
        snapshot.artifactStates.every(
          (artifact) =>
            artifact.href &&
            ["READY", "WAIT"].includes(artifact.state) &&
            artifact.waiting === (artifact.state === "WAIT") &&
            artifact.title.startsWith(`${artifact.state} ·`),
        ) &&
        snapshot.artifactStates.some((artifact) => artifact.state === "READY") &&
        (
          requireCommitted ||
          snapshot.artifactStates.some((artifact) => artifact.state === "WAIT")
        ),
      `artifact READY/WAIT state contract failed: ${
        JSON.stringify(snapshot.artifactStates.slice(0, 5))
      }`,
    );
    const formalRows = snapshot.formalArtifactContract.rows;
    const formalStates = formalRows.reduce((counts, row) => {
      counts[row.state] = (counts[row.state] || 0) + 1;
      return counts;
    }, {});
    const formalHrefs = new Set(formalRows.map((row) => row.href));
    const formalCheckpointRows = formalRows.filter((row) =>
      row.href.includes("/intermediate_checkpoints/")
    );
    const requiredFormalGates = [
      "results/qtail_droid_full/uniclash_checksum_handoff_gate.json",
      "results/qtail_droid_full/uniclash_pre_environment_gate.json",
      "results/qtail_droid_full/uniclash_pre_training_gate.json",
    ];
    assert(
      snapshot.formalArtifactContract.requiredCount >= 64 &&
        formalRows.length === snapshot.formalArtifactContract.requiredCount &&
        formalHrefs.size === formalRows.length &&
        formalRows.every((row) =>
          row.path &&
          row.href.startsWith("results/qtail_droid_full/") &&
          ["SEALED", "GENERATED", "WAIT"].includes(row.state) &&
          row.badge === row.state
        ) &&
        (formalStates.SEALED || 0) === snapshot.formalArtifactContract.sealedCount &&
        (formalStates.GENERATED || 0) === snapshot.formalArtifactContract.generatedCount &&
        (formalStates.WAIT || 0) === snapshot.formalArtifactContract.waitCount &&
        formalCheckpointRows.length === 20 &&
        requiredFormalGates.every((href) => formalHrefs.has(href)) &&
        snapshot.formalArtifactSummary.includes(
          `${snapshot.formalArtifactContract.requiredCount} / ${snapshot.formalArtifactContract.requiredCount}`
        ),
      `formal artifact ledger contract failed: ${JSON.stringify({
        summary: snapshot.formalArtifactSummary,
        counts: snapshot.formalArtifactContract,
        checkpointRows: formalCheckpointRows.length,
        requiredFormalGates: requiredFormalGates.map((href) => [href, formalHrefs.has(href)]),
      })}`,
    );
    const finalQaArtifact = snapshot.artifactStates.find(
      (artifact) =>
        artifact.href === "results/qtail_droid_full/final_page_qa.json",
    );
    const finalQaReady = requireCommitted || expectedCompletion === "9 / 9";
    assert(
      finalQaReady
        ? finalQaArtifact?.state === "READY"
        : (
          finalQaArtifact?.state === "WAIT" &&
          finalQaArtifact.title.includes("最终 QA 尚未成功完成")
        ),
      `final QA artifact state is inconsistent with the requested phase: ${
        JSON.stringify(finalQaArtifact)
      }`,
    );
    if (requireCommitted) {
      const requiredReadyArtifacts = new Set([
        "results/qtail_droid_full/final_page_qa.json",
        "results/qtail_droid_full/final_page_desktop.png",
        "results/qtail_droid_full/final_page_mobile.png",
        "results/qtail_droid_full/latest_final.json",
        "results/qtail_droid_full/completion_audit_final.json",
      ]);
      const byHref = new Map(
        snapshot.artifactStates.map((artifact) => [
          artifact.href,
          artifact,
        ]),
      );
      for (const href of requiredReadyArtifacts) {
        assert(
          byHref.get(href)?.state === "READY",
          `committed page artifact is not READY: ${href}`,
        );
      }
    }
    assert(
      snapshot.protocolAudit.startsWith("39 / 39 PASS") &&
        snapshot.protocolAudit.includes("ENV 9 / 9 PASS") &&
        snapshot.protocolAudit.includes("三态结论"),
      `protocol/environment self-tests are not visible as 39/39 and 9/9: ${snapshot.protocolAudit}`,
    );
    assert(
      snapshot.trainingOrderAudit.startsWith("11 / 11 PASS") &&
        snapshot.trainingOrderAudit.includes("187,891") &&
        snapshot.trainingOrderAudit.includes("optimizer"),
      `formal pre-optimizer gate order is missing: ${snapshot.trainingOrderAudit}`,
    );
    assert(
      snapshot.downloadMarkerAudit.startsWith(
        "MARKER 8 / 8 · MIRROR 8 / 8",
      ) &&
        snapshot.downloadMarkerAudit.includes("LIVE PARTIAL REJECTED"),
      `download and mirror controls are not visible as 8/8: ${snapshot.downloadMarkerAudit}`,
    );
    assert(
      snapshot.singleWriterAudit.startsWith("13 / 13 PASS") &&
        snapshot.singleWriterAudit.includes("双进程并发写入已拒绝") &&
        snapshot.singleWriterAudit.includes("容量熔断已验证") &&
        snapshot.singleWriterAudit.includes("硬绑定 en1"),
      `single-writer controls or activation boundary are missing: ${snapshot.singleWriterAudit}`,
    );
    assert(
      snapshot.runtimeProcessAudit.startsWith("16 / 16 PASS") &&
        snapshot.runtimeProcessAudit.includes("旧 PID") &&
        snapshot.runtimeProcessAudit.includes("无 ORICO 写权限替换") &&
        snapshot.runtimeProcessAudit.includes("伪造预热心跳") &&
        snapshot.runtimeProcessAudit.includes("过期心跳") &&
        snapshot.runtimeProcessAudit.includes("guard 缺失") &&
        snapshot.runtimeProcessAudit.includes("handoff 自愈") &&
        snapshot.runtimeProcessAudit.includes("唯一所有权") &&
        snapshot.runtimeProcessAudit.includes("损坏下载 marker") &&
        snapshot.runtimeProcessAudit.includes("checksum 收敛"),
      `runtime process destructive controls are missing: ${snapshot.runtimeProcessAudit}`,
    );
    if (
      requireIntermediate &&
      !requireResults &&
      expectedCompletion !== "8 / 9"
    ) {
      const activePrewarm =
        snapshot.runtimePrewarm.includes("PID MATCH") &&
        snapshot.runtimePrewarm.includes("HEARTBEAT PASS");
      const terminalPrewarm =
        snapshot.runtimePrewarm.includes("checksum_verified_exit") &&
        snapshot.runtimePrewarm.includes("TERMINAL") &&
        snapshot.runtimePrewarm.includes("HEARTBEAT PASS");
      assert(
        (activePrewarm || terminalPrewarm) &&
          !snapshot.runtimePrewarm.includes("BLOCKED"),
        `active or terminal prewarm state is not visible: ${snapshot.runtimePrewarm}`,
      );
    }
    const artifactSealMatch = snapshot.finalSealAudit.match(
      /(\d+) baseline formal artifacts · 当前 required (\d+) · 已封存 (\d+) · 已生成待封存 (\d+) · 未生成 (\d+)/,
    );
    assert(
      artifactSealMatch !== null &&
        Number(artifactSealMatch[1]) === 64 &&
        Number(artifactSealMatch[2]) ===
          Number(artifactSealMatch[3]) +
          Number(artifactSealMatch[4]) +
          Number(artifactSealMatch[5]),
      `artifact seal counts do not close to the dynamic required set: ${snapshot.finalSealAudit}`,
    );
    assert(
      snapshot.finalSealAudit.includes("38 / 38 marker") &&
        snapshot.finalSealAudit.includes("15 / 15 projection") &&
        snapshot.finalSealAudit.includes("8 / 8 manifest") &&
        snapshot.finalSealAudit.includes("11 / 11 shell") &&
        snapshot.finalSealAudit.includes("64 baseline formal artifacts") &&
        snapshot.finalSealAudit.includes("当前 required") &&
        snapshot.finalSealAudit.includes("已封存") &&
        snapshot.finalSealAudit.includes("已生成待封存") &&
        snapshot.finalSealAudit.includes("未生成") &&
        snapshot.finalSealAudit.includes("正式 committed 后才冻结 9/9"),
      `final sealing controls are missing: ${snapshot.finalSealAudit}`,
    );
    assert(
      snapshot.checksumVpnAudit.startsWith("PASS") &&
        snapshot.checksumVpnAudit.includes("LIVE 10 / 10") &&
        snapshot.checksumVpnAudit.includes("CONTROLS 13 / 13") &&
        snapshot.checksumVpnAudit.includes("Core ON") &&
        snapshot.checksumVpnAudit.includes("en1"),
      `checksum VPN gate is missing or incomplete: ${snapshot.checksumVpnAudit}`,
    );
    assert(
      snapshot.incrementalClosure.startsWith("PASS ·") &&
        snapshot.incrementalClosure.includes("MD5 objects") &&
        snapshot.incrementalClosure.includes("shards") &&
        snapshot.incrementalClosure.includes("records") &&
        snapshot.incrementalClosure.includes("deferred") &&
        snapshot.incrementalClosure.includes("checks 13/13") &&
        snapshot.incrementalClosure.includes("controls 7/7"),
      `incremental MD5/record/cache closure is missing: ${snapshot.incrementalClosure}`,
    );
    assert(
      snapshot.releaseMilestones.includes("1.0.0") &&
        snapshot.releaseMilestones.includes("1.0.1") &&
        snapshot.releaseMilestones.includes("objects"),
      `per-release milestone status is missing: ${snapshot.releaseMilestones}`,
    );
    assert(
      (
        snapshot.runtimeHandoff.startsWith("1 process") ||
        (
          snapshot.runtimeHandoff.startsWith("0 processes") &&
          snapshot.runtimeHandoff.includes("TERMINAL")
        )
      ) &&
        snapshot.runtimeHandoff.includes("TARGET MATCH") &&
        snapshot.runtimeHandoff.includes("1s POLL"),
      `download-generation handoff is not bound to the live pipeline: ${snapshot.runtimeHandoff}`,
    );
    if (requireResults || requireCommitted) {
      assert(
        snapshot.runtimeGeneration.startsWith("HASH MATCH") &&
          snapshot.runtimeGeneration.includes("PID"),
        `formal/final pipeline generation is not hash-bound: ${snapshot.runtimeGeneration}`,
      );
    } else {
      assert(
        (
          snapshot.runtimeGeneration.startsWith("HASH MATCH") ||
          (
            snapshot.runtimeGeneration.startsWith("HANDOFF PENDING") &&
            snapshot.runtimeGeneration.includes("legacy marker") &&
            snapshot.runtimeGeneration.includes("download-only")
          )
        ) &&
          !snapshot.runtimeGeneration.includes("BLOCKED"),
        `live pipeline generation state is invalid: ${snapshot.runtimeGeneration}`,
      );
    }
    assert(
      snapshot.runtimeSupervision.startsWith("PASS") &&
        snapshot.runtimeSupervision.includes("com.qtail.droid-full-pipeline") &&
        snapshot.runtimeSupervision.includes("com.qtail.uniclash-transport-guard"),
      `launchd reboot supervision is missing: ${snapshot.runtimeSupervision}`,
    );
    assert(snapshot.preflightInput === "8 shards", "real-TFRecord preflight shard count is missing");
    const normalizedPreflightRecords = snapshot.preflightRecords.replaceAll(",", "");
    assert(
      normalizedPreflightRecords.includes("16 records") &&
        /\b[1-9]\d* bytes\b/.test(normalizedPreflightRecords) &&
        normalizedPreflightRecords.includes("cap 2/shard"),
      `real-TFRecord preflight input is invalid: ${snapshot.preflightRecords}`,
    );
    assert(snapshot.preflightDevice === "MPS", "preflight did not run on MPS");
    assert(
      snapshot.preflightCompute.includes("Source 50 = Q-Tail 50 updates"),
      `preflight compute equality is missing: ${snapshot.preflightCompute}`,
    );
    assert(snapshot.preflightResume === "4 / 4", "checkpoint-resume preflight is incomplete");
    assert(
      snapshot.preflightResumeNote.includes("device match") &&
        snapshot.preflightResumeNote.includes("optimizer match"),
      `checkpoint-resume contract is invalid: ${snapshot.preflightResumeNote}`,
    );
    assert(snapshot.preflightGate === "WITHHELD", "bounded preflight leaked into the formal claim");
    assert(
      snapshot.preflightBoundary.includes("not_scientific_evidence") ||
        snapshot.preflightBoundary.includes("not scientific") ||
        snapshot.preflightBoundary.includes("only"),
      `preflight claim boundary is missing: ${snapshot.preflightBoundary}`,
    );
    assert(snapshot.forecastInput === "908 shards", "forecast shard count is missing");
    assert(snapshot.forecastTail === "+11.75 pp", "forecast tail gain is missing");
    assert(
      snapshot.forecastTailNote.includes("relative +35.9%") &&
        snapshot.forecastTailNote.includes("CI [+8.44, +14.81] pp"),
      `forecast relative gain or CI is missing: ${snapshot.forecastTailNote}`,
    );
    assert(
      snapshot.forecastExtreme === "+26.32 pp",
      "forecast extreme-underallocation reduction is missing",
    );
    assert(snapshot.forecastRare === "SLOWER", "negative rare-fingerprint forecast is hidden");
    assert(
      snapshot.forecastRareNote.includes("-4.21") &&
        snapshot.forecastRareNote.includes("-0.05"),
      `forecast rare-fingerprint range is missing: ${snapshot.forecastRareNote}`,
    );
    assert(
      snapshot.forecastBoundary.includes("预测性工程证据，不是正式完成门禁") &&
        snapshot.forecastBoundary.includes("缺少 1.0.1") &&
        snapshot.forecastBoundary.includes("不等于机器人 policy tail success"),
      "forecast claim boundary is incomplete",
    );
    assert(
      snapshot.canaryInput === "2,505 shards",
      `frozen scalability canary input is missing: ${snapshot.canaryInput}`,
    );
    assert(
      snapshot.canaryRecords.includes("113,426 records") &&
        snapshot.canaryRecords.includes("1.0.0 complete / 1.0.1 partial"),
      `frozen canary record/release boundary is missing: ${snapshot.canaryRecords}`,
    );
    assert(
      snapshot.canaryCompute === "2,000 = 2,000" &&
        snapshot.canaryDevice === "MPS · 1,000 updates/stage",
      `frozen canary equal-compute contract is missing: ${
        snapshot.canaryCompute
      } / ${snapshot.canaryDevice}`,
    );
    assert(
      snapshot.canaryResume === "4 / 4" &&
        snapshot.canaryHashes.includes("20 checkpoints") &&
        snapshot.canaryHashes.includes("hashes stable"),
      `frozen canary resume/hash evidence is missing: ${
        snapshot.canaryResume
      } / ${snapshot.canaryHashes}`,
    );
    assert(
      snapshot.canaryTail === "+14.23 pp" &&
        snapshot.canaryTailNote.includes("diagnostic only") &&
        snapshot.canaryTailNote.includes("32.76% → 47.00%") &&
        snapshot.canaryExtreme === "+29.41 pp",
      `bounded allocation diagnostics are missing or mislabeled: ${
        snapshot.canaryTail
      } / ${snapshot.canaryTailNote} / ${snapshot.canaryExtreme}`,
    );
    assert(
      snapshot.canaryRare === "MIXED / NEGATIVE" &&
        snapshot.canaryRareNote.includes("-2.60") &&
        snapshot.canaryRareNote.includes("+0.22"),
      `negative canary rare-fingerprint result is hidden: ${
        snapshot.canaryRare
      } / ${snapshot.canaryRareNote}`,
    );
    assert(
      snapshot.canaryBoundary.includes("工程演练通过") &&
        snapshot.canaryBoundary.includes("DROID 1.0.1 尚不完整") &&
        snapshot.canaryBoundary.includes("不得升级为全量 claim"),
      `frozen canary claim boundary is incomplete: ${
        snapshot.canaryBoundary
      }`,
    );
  }
  if (requireResults) {
    assert(
      snapshot.incrementalClosure.includes("4,102 MD5 objects") &&
      snapshot.incrementalClosure.includes("4,096 shards") &&
        snapshot.incrementalClosure.includes("187,891 records") &&
        snapshot.incrementalClosure.includes("deferred 0") &&
        snapshot.incrementalClosure.includes("formal gate OPEN"),
      `full MD5/record/cache closure is not visible: ${snapshot.incrementalClosure}`,
    );
    assert(
      snapshot.releaseMilestones.includes("1.0.0 COMPLETE") &&
        snapshot.releaseMilestones.includes("1.0.1 COMPLETE"),
      `per-release milestones are not complete: ${snapshot.releaseMilestones}`,
    );
    assert(
      snapshot.checkpointGridSummary.startsWith("20 / 20 VERIFIED") &&
        snapshot.checkpointGridRows.every(
          (row) => row.slice(1).every((state) => state === "VERIFIED"),
        ),
      `formal 20-checkpoint grid is not fully verified: ${
        snapshot.checkpointGridSummary
      }`,
    );
    assert(snapshot.trainedShards !== "等待", "trained shard result is still waiting");
    assert(snapshot.parseRate !== "等待", "parse-rate result is still waiting");
    assert(snapshot.tailGain !== "等待", "tail-gain result is still waiting");
    assert(
      snapshot.tailCi.includes("relative") &&
        snapshot.tailCi.includes("stratified") &&
        snapshot.tailCi.includes(" pp"),
      "tail-share result does not distinguish relative gain and percentage points",
    );
    assert(!snapshot.resultBoundary.includes("当前尚未"), "claim boundary still shows pre-training placeholder");
    assert(
      snapshot.resultBoundary.includes("AllocationHead") &&
        snapshot.resultBoundary.includes("不是机器人 policy 训练") &&
        snapshot.resultBoundary.includes("不是有效 p 值") &&
        snapshot.resultBoundary.includes("相互独立"),
      `fixed formal-result claim boundary is missing: ${snapshot.resultBoundary}`,
    );
    assert(
      snapshot.rareCoverageBoundary.includes("not semantic task coverage") &&
        snapshot.rareCoverageBoundary.includes("not") &&
        snapshot.rareCoverageBoundary.includes("robot-policy success"),
      `rare-fingerprint claim boundary is missing: ${snapshot.rareCoverageBoundary}`,
    );
    const rareCoverageVisible =
      snapshot.rareCoverageSummary.includes("rare fingerprints") &&
      snapshot.rareCoverageSummary.includes("50% coverage") &&
      snapshot.rareCoverageRows.length === 7 &&
      JSON.stringify(snapshot.rareCoverageRows.map((row) => row[0])) ===
        JSON.stringify(["10", "25", "50", "100", "200", "400", "800"]) &&
      snapshot.rareCoverageRows.every(
        (row) => row.length === 4 && row[1].endsWith("%") &&
          row[2].endsWith("%") && row[3].endsWith(" pp"),
      );
    const noEligibleRareCoverageVisible =
      snapshot.rareCoverageSummary.includes("no eligible fingerprints") &&
      snapshot.rareCoverageSummary.includes("N/A") &&
      snapshot.rareCoverageRows.length === 0;
    assert(
      rareCoverageVisible || noEligibleRareCoverageVisible,
      `rare-fingerprint coverage state is incomplete: ${snapshot.rareCoverageSummary}`,
    );
    const formalOutcome = snapshot.hypothesisGate.toUpperCase();
    assert(
      ["SUPPORTED", "INCONCLUSIVE", "NOT_SUPPORTED"].includes(formalOutcome),
      `formal three-state outcome is invalid: ${snapshot.hypothesisGate}`,
    );
    assert(
      snapshot.valueEvidenceState === `FORMAL · ${formalOutcome}` &&
        snapshot.valueEvidenceNote.includes("4,096 shards") &&
        snapshot.valueEvidenceNote.includes("187,891 records") &&
        snapshot.valueEvidenceNote.includes("Source 40,000 = Q-Tail 40,000 optimizer updates"),
      `formal evidence-to-value binding is incomplete: ${
        snapshot.valueEvidenceState
      } | ${snapshot.valueEvidenceNote}`,
    );
    const expectedValueStates = {
      SUPPORTED: ["长尾分配目标获支持", "可进入受限客户试点"],
      INCONCLUSIVE: ["证据尚不确定", "仅限探索性试点"],
      NOT_SUPPORTED: ["预注册目标未获支持", "不得销售效果提升"],
    };
    assert(
      snapshot.valueTechnicalState === expectedValueStates[formalOutcome][0] &&
        snapshot.valueCommercialState === expectedValueStates[formalOutcome][1],
      `formal outcome was mapped to the wrong value decision: ${
        JSON.stringify({
          outcome: formalOutcome,
          technical: snapshot.valueTechnicalState,
          commercial: snapshot.valueCommercialState,
        })
      }`,
    );
    assert(
      snapshot.valueStatusNote.includes(formalOutcome) &&
        snapshot.valueTechnicalNote.toLowerCase().includes("tail allocation") &&
        snapshot.valueCommercialNote.length > 30 &&
        snapshot.valueBoundary.includes(formalOutcome) &&
        snapshot.valueBoundary.includes("AllocationHead") &&
        snapshot.valueBoundary.includes("不构成机器人 policy 成功率") &&
        snapshot.valueBoundary.includes("ROI"),
      `formal technical/commercial interpretation is incomplete: ${
        snapshot.valueStatusNote
      } | ${snapshot.valueBoundary}`,
    );
  }
  if (screenshotPath) {
    await page.screenshot({path: screenshotPath, fullPage: true});
  }
  await context.close();
  return {
    viewport,
    ...snapshot,
    artifact_ready: snapshot.artifactStates.filter(
      (artifact) => artifact.state === "READY",
    ).length,
    artifact_wait: snapshot.artifactStates.filter(
      (artifact) => artifact.state === "WAIT",
    ).length,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    failed_responses: failedResponses,
  };
}

async function verifyValueDecisionStates(browser, {
  pageUrl,
  baseLatest,
  baseTraining,
}) {
  const fixtures = [
    {
      outcome: "supported",
      sourceTailShare: 0.30,
      qtailTailShare: 0.35,
      ciLow: 3.5,
      ciHigh: 6.5,
      extremeReduction: 8.0,
      technical: "长尾分配目标获支持",
      commercial: "可进入受限客户试点",
    },
    {
      outcome: "inconclusive",
      sourceTailShare: 0.30,
      qtailTailShare: 0.315,
      ciLow: 1.0,
      ciHigh: 3.0,
      extremeReduction: 5.0,
      technical: "证据尚不确定",
      commercial: "仅限探索性试点",
    },
    {
      outcome: "not_supported",
      sourceTailShare: 0.30,
      qtailTailShare: 0.29,
      ciLow: -2.0,
      ciHigh: 0.0,
      extremeReduction: 0.0,
      technical: "预注册目标未获支持",
      commercial: "不得销售效果提升",
    },
  ];
  const controls = [];
  for (const fixture of fixtures) {
    const latest = JSON.parse(JSON.stringify(baseLatest));
    const training = JSON.parse(JSON.stringify(baseTraining));
    const effect = training.effect_metrics;
    effect.source_pred_tail_share = fixture.sourceTailShare;
    effect.qtail_pred_tail_share = fixture.qtailTailShare;
    effect.predicted_tail_share_gain_pp =
      (fixture.qtailTailShare - fixture.sourceTailShare) * 100;
    effect.source_extreme_underallocation_rate =
      fixture.extremeReduction / 100;
    effect.qtail_extreme_underallocation_rate = 0;
    effect.extreme_underallocation_reduction_pp =
      fixture.extremeReduction;
    effect.paired_bootstrap.ci95_low_pp = fixture.ciLow;
    effect.paired_bootstrap.ci95_high_pp = fixture.ciHigh;
    effect.hypothesis_gate.outcome = fixture.outcome;
    effect.hypothesis_gate.supported = fixture.outcome === "supported";
    effect.hypothesis_gate.passed = fixture.outcome === "supported";
    training.status = "complete";
    training.shard_count = 4_096;
    training.trajectory_evidence.records_decoded = 187_891;
    training.compute_audit.source_optimizer_updates = 40_000;
    training.compute_audit.qtail_optimizer_updates = 40_000;
    latest.training = training;
    latest.completion_audit = {
      ...(latest.completion_audit || {}),
      experiment_execution_valid: true,
      formal_results_publishable: true,
      outcome_is_completion_gate: false,
    };
    latest.markers = {
      ...(latest.markers || {}),
      droid_training_complete: true,
      final_page_qa_complete: true,
      droid_public_projection_committed: true,
      public_projection_validation: {
        valid: true,
        errors: [],
      },
    };

    const context = await browser.newContext({
      viewport: {width: 1024, height: 768},
      locale: "zh-CN",
    });
    try {
      const page = await context.newPage();
      await page.route(
        "**/results/qtail_droid_full/latest.json*",
        async (route) => {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(latest),
          });
        },
      );
      await page.goto(`${pageUrl}?value-fixture=${fixture.outcome}`, {
        waitUntil: "networkidle",
        timeout: 30_000,
      });
      await page.waitForFunction(
        (outcome) =>
          document.querySelector("#value-evidence-state")?.textContent?.trim()
            === `FORMAL · ${outcome.toUpperCase()}`,
        fixture.outcome,
        {timeout: 20_000},
      );
      const observed = await page.evaluate(() => ({
        evidence:
          document.querySelector("#value-evidence-state")?.textContent?.trim()
          || "",
        technical:
          document.querySelector("#value-technical-state")?.textContent?.trim()
          || "",
        commercial:
          document.querySelector("#value-commercial-state")?.textContent?.trim()
          || "",
        boundary:
          document.querySelector("#value-boundary")?.textContent?.trim()
          || "",
      }));
      assert(
        observed.technical === fixture.technical &&
          observed.commercial === fixture.commercial,
        `value-state fixture mapped incorrectly: ${
          JSON.stringify({fixture, observed})
        }`,
      );
      assert(
        observed.boundary.includes(fixture.outcome.toUpperCase()) &&
          observed.boundary.includes("AllocationHead") &&
          observed.boundary.includes("不构成机器人 policy 成功率") &&
          observed.boundary.includes("ROI"),
        `value-state fixture lost its claim boundary: ${
          JSON.stringify({fixture, observed})
        }`,
      );
      controls.push({
        outcome: fixture.outcome,
        passed: true,
        observed,
      });
    } finally {
      await context.close();
    }
  }
  return {
    version: "qtail_droid_value_decision_ui_selftest_v1",
    status: "passed",
    controls_passed: controls.filter((item) => item.passed).length,
    controls_total: controls.length,
    fixture_scope: "presentation_logic_only_not_formal_training_evidence",
    controls,
  };
}

function validateReleaseMetadataAudit(metadata) {
  const expectedGates = [
    "official_checksum_manifest",
    "both_releases_verified",
    "combined_shards_4096",
    "combined_records_187891",
    "combined_split_bytes_match",
    "step_schemas_identical",
    "training_features_present",
  ];
  assert(
    metadata.version === "droid_release_metadata_audit_v1" &&
      metadata.status === "verified" &&
      metadata.source === "gs://gresearch/robotics/droid",
    "official release metadata audit identity is invalid",
  );
  const gates = metadata.gates || {};
  assert(
    JSON.stringify(Object.keys(gates).sort()) ===
      JSON.stringify([...expectedGates].sort()) &&
      Object.values(gates).every((value) => value === true),
    "official release metadata audit gates are incomplete",
  );
  const combined = metadata.combined_official_metadata || {};
  assert(
    Number(combined.tfrecord_shards) === 4_096 &&
      Number(combined.records) === 187_891 &&
      Number(combined.split_bytes) === 3_700_742_144_299,
    "combined official release metadata is not exact",
  );
  const expectedReleases = new Map([
    ["1.0.0", {
      datasetName: "r2d2_faceblur",
      datasetVersion: "1.4.0",
      shards: 2_048,
      records: 92_233,
      splitBytes: 1_834_749_018_029,
    }],
    ["1.0.1", {
      datasetName: "droid_101",
      datasetVersion: "0.0.1",
      shards: 2_048,
      records: 95_658,
      splitBytes: 1_865_993_126_270,
    }],
  ]);
  const releases = new Map(
    (Array.isArray(metadata.releases) ? metadata.releases : [])
      .map((item) => [String(item.release), item]),
  );
  assert(releases.size === expectedReleases.size, "metadata release set is incomplete");
  for (const [release, expected] of expectedReleases) {
    const item = releases.get(release);
    assert(
      item?.verified === true &&
        item.dataset_name === expected.datasetName &&
        item.dataset_version === expected.datasetVersion &&
        Number(item.official_tfrecord_shards) === expected.shards &&
        Number(item.official_records) === expected.records &&
        Number(item.official_split_bytes) === expected.splitBytes &&
        item.required_training_features_present === true &&
        item.dataset_info_file?.verified === true &&
        item.features_file?.verified === true,
      `official metadata mismatch for release ${release}`,
    );
  }
  assert(
    String(metadata.claim_boundary || "").includes(
      "does not prove that 4,096 TFRecord shards or 187,891 records",
    ),
    "release metadata claim boundary is missing",
  );
}

function validateIncrementalClosure(closure) {
  const current = closure.current_closure || {};
  const checks = closure.checks || {};
  assert(
    closure.format_version === "qtail_droid_incremental_closure_v2" &&
      closure.status === "complete" &&
      closure.formal_full_mirror_gate === true,
    "incremental closure did not become a full closure",
  );
  assert(
    Number(current.verified_objects) === 4_102 &&
      Number(current.completed_tfrecords) === 4_096 &&
      Number(current.listed_verified_caches) === 4_096 &&
      Number(current.decoded_records) === 187_891 &&
      Number(current.transport_partial_files) === 0 &&
      Number(current.deferred_after_snapshot_tfrecords) === 0 &&
      Number(current.missing_from_snapshot_tfrecords) === 0,
    "full MD5/record/cache closure counts are not exact",
  );
  assert(
    Object.keys(checks).length >= 12 &&
      Object.values(checks).every((value) => value === true) &&
      Number(closure.error_count) === 0 &&
      (closure.failed_checks || []).length === 0,
    "full MD5/record/cache closure checks failed",
  );
  assert(
    String(closure.claim_boundary || "").includes(
      "not full-mirror or model-quality evidence until",
    ),
    "incremental closure claim boundary is missing",
  );
}

const PREWARM_STATUS_CONTROL_NAMES = [
  "coverage_error_is_never_complete",
  "empty_snapshot_is_rejected",
  "exact_official_snapshot_is_complete",
  "excess_shards_are_rejected",
  "partial_snapshot_is_caught_up_not_complete",
  "wrong_release_composition_is_rejected",
];

function validatePrewarmStatusContractSelftest(selftest) {
  const checks = selftest.checks || {};
  const observedNames = Object.keys(checks).sort();
  assert(
    selftest.format_version ===
      "qtail_prewarm_status_contract_selftest_v1" &&
      selftest.status === "passed" &&
      Number(selftest.passed) === PREWARM_STATUS_CONTROL_NAMES.length &&
      Number(selftest.total) === PREWARM_STATUS_CONTROL_NAMES.length,
    "prewarm status-scope control summary is invalid",
  );
  assert(
    JSON.stringify(observedNames) ===
      JSON.stringify(PREWARM_STATUS_CONTROL_NAMES) &&
      observedNames.every((name) => checks[name] === true),
    "prewarm status-scope controls are incomplete or failed",
  );
}

async function validateEnvironmentCodeBinding({
  environment,
  selftest,
  report,
  environmentPath,
}) {
  const requiredChecks = new Set([
    "positive_control_completes",
    "one_byte_mirror_mismatch_fails",
    "orchestration_snapshot_code_drift_fails",
    "missing_official_md5_fails",
    "uniclash_violation_fails",
    "transport_classifier_v6_selftest_passes",
    "backend_commit_drift_fails",
    "backend_origin_drift_fails",
    "backend_worktree_dirty_fails",
  ]);
  const checks = selftest.checks || {};
  assert(
    selftest.status === "passed" &&
      selftest.contract_version ===
        "qtail_droid_environment_contract_selftest_v3" &&
      Object.keys(checks).length === requiredChecks.size &&
      Object.keys(checks).every((name) => requiredChecks.has(name)) &&
      Object.values(checks).every((value) => value === true),
    "environment contract self-test is not exact v3 9/9",
  );
  assert(
    environment.status === "complete" &&
      Object.keys(environment.gates || {}).length > 0 &&
      Object.values(environment.gates || {}).every(
        (value) => value === true,
      ),
    "environment manifest gates are incomplete",
  );
  const codeRows = Array.isArray(environment.code) ? environment.code : [];
  assert(codeRows.length > 0, "environment code inventory is empty");
  assert(
    new Set(codeRows.map((item) => item.path)).size === codeRows.length,
    "environment code inventory contains duplicate paths",
  );
  const liveHashes = await Promise.all(
    codeRows.map(async (item) => ({
      item,
      exists: existsSync(String(item.path || "")),
      liveSha256: existsSync(String(item.path || ""))
        ? await sha256(String(item.path))
        : null,
    })),
  );
  assert(
    liveHashes.every(
      ({item, exists, liveSha256}) =>
        item?.exists === true &&
        exists &&
        /^[0-9a-f]{64}$/.test(String(item.sha256 || "")) &&
        liveSha256 === item.sha256,
    ),
    "live critical code no longer matches the environment manifest",
  );
  assert(
    codeRows.some((item) =>
      String(item.path || "").endsWith("/tools/qtail_train_droid_full.py"),
    ),
    "formal trainer is absent from the environment code inventory",
  );
  const snapshot = environment.orchestration_snapshot || {};
  assert(
    snapshot.code_parity_passed === true &&
      Number(snapshot.code_mismatch_count) === 0 &&
      (snapshot.manifest_errors || []).length === 0 &&
      existsSync(String(snapshot.manifest || "")) &&
      await sha256(String(snapshot.manifest)) === snapshot.manifest_sha256,
    "ORICO orchestration snapshot binding is invalid",
  );
  const binding = report.environment_code_binding || {};
  assert(
    binding.required === true &&
      binding.passed === true &&
      binding.manifest === environmentPath &&
      binding.manifest_sha256 === await sha256(environmentPath) &&
      Number(binding.checked_code_entries) === codeRows.length &&
      Number(binding.mismatch_count) === 0 &&
      (binding.errors || []).length === 0 &&
      binding.snapshot_code_parity_passed === true &&
      binding.snapshot_manifest_sha256 === snapshot.manifest_sha256,
    "training report is not bound to the live environment/code generation",
  );
}

function validateFinalEvidence({report, audit, latest}) {
  assert(
    audit.experiment_execution_valid === true &&
      audit.formal_results_publishable === false &&
      audit.outcome_is_completion_gate === false &&
      ["supported", "inconclusive", "not_supported"].includes(
        audit.hypothesis_outcome,
      ),
    "pre-QA publication state is not separated from hypothesis direction",
  );
  const disk = latest.external_storage || {};
  assert(
    disk.capacity_model_version ===
      "official_md5_plus_allocated_resumable_parts_v2" &&
      disk.capacity_gate_passed === true,
    "ORICO authoritative capacity gate is missing or failed",
  );
  assert(
    Number(disk.safety_reserve_bytes) === 185_037_263_258 &&
      Number(disk.required_free_bytes) ===
        Number(disk.remaining_official_bytes) +
          Number(disk.safety_reserve_bytes) &&
      Number(disk.headroom_bytes) ===
        Number(disk.free_bytes) - Number(disk.required_free_bytes) &&
      Number(disk.headroom_bytes) >= 0,
    "ORICO capacity arithmetic or 5% reserve is invalid",
  );
  const history = latest.history_chart || {};
  const historyPoints = Array.isArray(history.points) ? history.points : [];
  assert(
    history.version === "qtail_droid_bounded_history_chart_v1" &&
      history.sampling === "uniform_index_plus_stage_boundaries_v1" &&
      Number(history.max_points) === 240 &&
      Number(history.point_count) === historyPoints.length &&
      historyPoints.length >= 2 &&
      historyPoints.length <= 240 &&
      Number(history.source_sample_count) >= historyPoints.length,
    "bounded hash-chain history view is invalid",
  );
  assert(
    historyPoints.at(-1)?.generated_at ===
      latest.pipeline_timeline?.last_generated_at,
    "bounded history does not preserve the latest hash-chain sample",
  );
  assert(report.status === "complete", "training report is not complete");
  assert(
    report.training_scope === "all_complete_shards_all_decodable_records",
    "training report is not an all-shard/all-record run",
  );
  const formal = report.formal_protocol || {};
  assert(
    formal.locked === true &&
      Number(formal.seed) === 11 &&
      Number(formal.steps_per_stage) === 20_000 &&
      Number(formal.holdout_fraction) === 0.20 &&
      Number(formal.holdout_shards_per_release) === 410 &&
      Number(formal.bootstrap_samples) === 5_000 &&
      Number(formal.randomization_samples) === 5_000 &&
      Number(formal.checkpoint_every_steps) === 5_000 &&
      Number(formal.min_record_parse_rate) === 1 &&
      Number(formal.min_record_scan_complete_rate) === 1 &&
      formal.require_verified_mirror === true &&
      formal.pt_source_sha256 ===
        "59e487af80482215b2c2d4e81e9ccd7471ac6c94c1ef40547596ccb80367e75f" &&
      Number(report.seed) === 11,
    "formal protocol lock is invalid",
  );
  const trajectory = report.trajectory_evidence || {};
  assert(trajectory.full_record_mode === true, "full-record mode is false");
  assert(trajectory.record_parse_rate === 1, "record parse rate is not 100%");
  assert(trajectory.record_scan_complete_rate === 1, "record scan completion is not 100%");
  const input = report.input_audit || {};
  assert(input.required === true && input.verified === true, "verified-mirror input gate failed");
  assert(
    Number(input.formal_expected_object_count) === 4_102 &&
      Number(input.formal_expected_tfrecord_shards) === 4_096 &&
      Number(input.formal_expected_total_bytes) === 3_700_745_265_151 &&
      input.download_marker_matches_current_binding === true &&
      Object.values(input.current_binding_checks || {}).length === 13 &&
      Object.values(input.current_binding_checks || {}).every(Boolean) &&
      Number(input.current_binding_file_error_count) === 0,
    "formal report is not rebound to the current 4,102-object mirror",
  );
  assert(
    Number(input.expected_tfrecord_shards) === Number(input.actual_tfrecord_shards),
    "formal training did not cover every official TFRecord shard",
  );
  const expectedReleases = new Map([
    ["1.0.0", {dataset: "r2d2_faceblur", shards: 2_048, records: 92_233}],
    ["1.0.1", {dataset: "droid_101", shards: 2_048, records: 95_658}],
  ]);
  const releases = new Map(
    (Array.isArray(report.release_composition) ? report.release_composition : [])
      .map((item) => [String(item.release), item]),
  );
  assert(releases.size === expectedReleases.size, "release composition is incomplete");
  for (const [release, expected] of expectedReleases) {
    const item = releases.get(release);
    assert(item, `release ${release} is missing`);
    assert(
      item.official_dataset_name === expected.dataset,
      `release ${release} dataset name is unexpected`,
    );
    assert(item.metadata_status === "verified", `release ${release} metadata is unverified`);
    assert(
      Number(item.observed_tfrecord_shards) === expected.shards &&
        Number(item.official_tfrecord_shards) === expected.shards,
      `release ${release} shard coverage is incomplete`,
    );
    assert(
      Number(item.observed_records_decoded) === expected.records &&
        Number(item.official_expected_records) === expected.records,
      `release ${release} episode count does not match official metadata`,
    );
    assert(
      item.full_shard_coverage === true && item.full_record_count_match === true,
      `release ${release} full-record gate failed`,
    );
  }
  assert(
    [...releases.values()].reduce(
      (sum, item) => sum + Number(item.observed_tfrecord_shards || 0),
      0,
    ) === 4_096,
    "combined release shard count is not 4,096",
  );
  assert(
    [...releases.values()].reduce(
      (sum, item) => sum + Number(item.observed_records_decoded || 0),
      0,
    ) === 187_891,
    "combined release episode count is not 187,891",
  );
  assert(
    [...releases.values()].reduce(
      (sum, item) => sum + Number(item.observed_tfrecord_bytes || 0),
      0,
    ) === Number(report.total_bytes),
    "combined release bytes do not match the training report",
  );
  const compute = report.compute_audit || {};
  assert(Number(report.steps) === 20_000, "training steps are not 20,000 per stage");
  assert(Number(report.total_steps_per_arm) === 40_000, "total steps are not 40,000 per arm");
  assert(Number(compute.source_steps) === 40_000, "source total step count mismatch");
  assert(Number(compute.qtail_steps) === 40_000, "Q-Tail total step count mismatch");
  for (const field of [
    "evaluation_source_steps",
    "evaluation_qtail_steps",
    "deployment_source_steps",
    "deployment_qtail_steps",
  ]) {
    assert(Number(compute[field]) === Number(report.steps), `${field} mismatch`);
  }
  for (const field of [
    "evaluation_source_optimizer_updates",
    "evaluation_qtail_optimizer_updates",
    "deployment_source_optimizer_updates",
    "deployment_qtail_optimizer_updates",
  ]) {
    assert(
      Number(compute[field]) === Number(report.steps),
      `${field} mismatch`,
    );
  }
  assert(
    Number(compute.source_optimizer_updates) === 40_000 &&
      Number(compute.qtail_optimizer_updates) === 40_000,
    "total optimizer-update count differs between arms",
  );
  assert(
    compute.optimizer_update_semantics ===
      "Checkpoint step k stores the state after exactly k optimizer updates; each stage ends at k=steps.",
    "optimizer-update boundary semantics are missing",
  );
  const resume = compute.resume || {};
  const expectedResumeKeys = [
    "deployment_qtail",
    "deployment_source",
    "evaluation_qtail",
    "evaluation_source",
  ];
  assert(
    JSON.stringify(Object.keys(resume).sort()) ===
      JSON.stringify(expectedResumeKeys),
    "per-stage resume audit is incomplete",
  );
  for (const [stage, item] of Object.entries(resume)) {
    assert(
      Number(item.target_step) === Number(report.steps) &&
        Number(item.optimizer_updates_completed) === Number(report.steps) &&
        item.device === compute.training_device &&
        item.optimizer === compute.same_optimizer &&
        (
          item.resumed !== true ||
          (
            item.checkpoint_device === compute.training_device &&
            item.checkpoint_optimizer === compute.same_optimizer &&
            item.checkpoint_environment_fingerprint ===
              compute.checkpoint_environment_fingerprint
          )
        ) &&
        item.environment_fingerprint === compute.checkpoint_environment_fingerprint &&
        item.step_semantics ===
          "Checkpoint step k is the state after exactly k optimizer updates.",
      `${stage} optimizer-update audit mismatch`,
    );
  }
  assert(
    compute.architecture === "AllocationHead(10→32→16→1)",
    "architecture signature is unexpected",
  );
  assert(compute.same_architecture === true, "architecture equality gate failed");
  assert(compute.same_seed === true, "seed equality gate failed");
  assert(compute.same_features === true, "feature equality gate failed");
  assert(compute.same_device === true, "device equality gate failed");
  assert(
    compute.same_environment_fingerprint === true &&
      /^[a-f0-9]{64}$/.test(String(compute.runtime_environment_fingerprint || "")),
    "runtime environment fingerprint gate failed",
  );
  const checkpointEnvironment = compute.checkpoint_environment_contract || {};
  const formalCheckpointBinding =
    checkpointEnvironment.formal_environment_binding || {};
  const codeBinding = report.environment_code_binding || {};
  assert(
    /^[a-f0-9]{64}$/.test(
      String(compute.checkpoint_environment_fingerprint || ""),
    ) &&
      checkpointEnvironment.version === "qtail_checkpoint_environment_v2" &&
      checkpointEnvironment.formal_run === true &&
      JSON.stringify(checkpointEnvironment.runtime_environment) ===
        JSON.stringify(compute.runtime_environment) &&
      formalCheckpointBinding.required === true &&
      formalCheckpointBinding.passed === true &&
      formalCheckpointBinding.environment_manifest_sha256 ===
        codeBinding.manifest_sha256 &&
      formalCheckpointBinding.checked_code_aggregate_sha256 ===
        codeBinding.checked_code_aggregate_sha256 &&
      formalCheckpointBinding.orico_snapshot_manifest_sha256 ===
        codeBinding.snapshot_manifest_sha256 &&
      formalCheckpointBinding.snapshot_code_parity_passed === true,
    "checkpoint environment is not bound to the formal ORICO code snapshot",
  );
  assert(
    compute.same_optimizer === "AdamW(lr=0.002, weight_decay=0.0001)",
    "optimizer equality gate failed",
  );
  assert(compute.same_parameter_count === true, "parameter-count equality gate failed");
  assert(
    Number.isInteger(Number(compute.source_parameter_count)) &&
      Number(compute.source_parameter_count) > 0,
    "source parameter count is not a positive integer",
  );
  assert(
    Number(compute.source_parameter_count) === Number(compute.qtail_parameter_count),
    "source and Q-Tail parameter counts differ",
  );
  const checkpointAudit = report.intermediate_checkpoint_audit || {};
  assert(
    checkpointAudit.status === "complete" &&
      checkpointAudit.paired_feature_signatures_equal === true &&
      checkpointAudit.initialized_state_signatures_equal === true,
    "checkpoint feature/initialization equality audit failed",
  );
  const effect = report.effect_metrics || {};
  const bootstrap = effect.paired_bootstrap || {};
  const randomization = effect.paired_arm_swap_randomization || {};
  const hypothesisGate = effect.hypothesis_gate || {};
  const finiteEffectValues = [
    effect.source_pred_tail_share,
    effect.qtail_pred_tail_share,
    effect.predicted_tail_share_gain_pp,
    effect.source_extreme_underallocation_rate,
    effect.qtail_extreme_underallocation_rate,
    effect.extreme_underallocation_reduction_pp,
    bootstrap.mean_gain_pp,
    bootstrap.ci95_low_pp,
    bootstrap.ci95_high_pp,
    bootstrap.descriptive_fraction_gain_le_zero,
    randomization.observed_gain_pp,
    randomization.diagnostic_exceedance_fraction,
  ].map(Number);
  assert(finiteEffectValues.every(Number.isFinite), "effect metrics contain non-finite values");
  assert(
    effect.tail_definition === "heldout_top_30_percent_by_record_informed_tail_score_v2",
    "tail definition is unexpected",
  );
  assert(
    effect.extreme_definition === "heldout_top_10_percent_by_record_informed_tail_score_v2",
    "extreme definition is unexpected",
  );
  assert(
    effect.evaluation_scope === "deterministic_release_stratified_heldout_shards",
    "effect metrics are not held-out",
  );
  assert(Number(bootstrap.samples) === 5_000, "paired bootstrap does not contain 5,000 samples");
  assert(
    bootstrap.method ===
      "paired_release_stratified_shard_bootstrap_within_draw_renormalization",
    "paired bootstrap method is unexpected",
  );
  assert(
    bootstrap.p_gain_le_zero_is_p_value === false &&
      bootstrap.inference_role ===
        "conditional_percentile_interval_and_descriptive_fraction_only",
    "bootstrap descriptive fraction is mislabeled as a p-value",
  );
  assert(
    JSON.stringify(bootstrap.strata) === JSON.stringify(["1.0.0", "1.0.1"]) &&
      Object.values(bootstrap.strata_counts || {}).reduce(
        (sum, value) => sum + Number(value),
        0,
      ) === Number(report.holdout_evaluation?.holdout_shards),
    "paired bootstrap release strata are invalid",
  );
  assert(
    randomization.version === "paired_shard_arm_swap_diagnostic_v2" &&
      Number(randomization.samples) === 5_000 &&
      randomization.unit === "non_independent_heldout_shard_weight" &&
      randomization.finite_sample_correction === "(k+1)/(B+1)" &&
      randomization.exchangeability_justified_by_experiment_design === false &&
      randomization.inference_role ===
        "dependency_sensitive_descriptive_diagnostic_only" &&
      randomization.conditional_p_value_is_valid_p_value === false &&
      Number(randomization.diagnostic_exceedance_fraction) > 0 &&
      Number(randomization.diagnostic_exceedance_fraction) <= 1 &&
      Number(randomization.conditional_p_value) ===
        Number(randomization.diagnostic_exceedance_fraction) &&
      hypothesisGate.name === "heldout_tail_allocation_outcome_v4" &&
      Number(hypothesisGate.minimum_tail_share_gain_pp) === 2 &&
      hypothesisGate.requires_ci95_low_at_least_minimum === true &&
      hypothesisGate.requires_positive_extreme_underallocation_reduction ===
        true &&
      hypothesisGate.completion_role ===
        "outcome_only_not_experiment_execution_gate" &&
      hypothesisGate.randomization_diagnostic_is_valid_p_value === false,
    "held-out outcome contract is invalid",
  );
  for (const field of [
    "source_pred_tail_share",
    "qtail_pred_tail_share",
    "source_extreme_underallocation_rate",
    "qtail_extreme_underallocation_rate",
  ]) {
    assert(
      Number(effect[field]) >= 0 && Number(effect[field]) <= 1,
      `${field} is outside [0, 1]`,
    );
  }
  assert(
    Number(bootstrap.ci95_low_pp) <= Number(bootstrap.ci95_high_pp),
    "paired bootstrap CI is reversed",
  );
  assert(
    Number(bootstrap.descriptive_fraction_gain_le_zero) >= 0 &&
      Number(bootstrap.descriptive_fraction_gain_le_zero) <= 1,
    "paired bootstrap descriptive fraction is outside [0, 1]",
  );
  const recomputedTailGain =
    (Number(effect.qtail_pred_tail_share) -
      Number(effect.source_pred_tail_share)) * 100;
  const recomputedExtremeReduction =
    (Number(effect.source_extreme_underallocation_rate) -
      Number(effect.qtail_extreme_underallocation_rate)) * 100;
  const recomputedSupported =
    recomputedTailGain >= 2 &&
    Number(bootstrap.ci95_low_pp) >= 2 &&
    recomputedExtremeReduction > 0;
  const recomputedNotSupported =
    Number(bootstrap.ci95_high_pp) < 2 ||
    recomputedExtremeReduction <= 0;
  const recomputedOutcome = recomputedSupported
    ? "supported"
    : recomputedNotSupported
      ? "not_supported"
      : "inconclusive";
  assert(
    Math.abs(
      Number(effect.predicted_tail_share_gain_pp) - recomputedTailGain,
    ) <= 1e-9 &&
      Math.abs(
        Number(effect.extreme_underallocation_reduction_pp) -
          recomputedExtremeReduction,
      ) <= 1e-9 &&
      hypothesisGate.outcome === recomputedOutcome &&
      hypothesisGate.supported === recomputedSupported &&
      hypothesisGate.passed === recomputedSupported &&
      Number(effect.tail_selected_shards) === 246 &&
      Number(effect.tail_total_holdout_shards) === 820 &&
      Number(effect.extreme_selected_shards) === 82 &&
      Number(effect.extreme_total_holdout_shards) === 820,
    "effect metrics, outcome, or exact tail selection do not recompute",
  );
  const holdout = report.holdout_evaluation || {};
  const holdoutRelativePaths = Array.isArray(
    holdout.holdout_relative_paths,
  ) && holdout.holdout_relative_paths.every(
    (value) => typeof value === "string",
  )
    ? holdout.holdout_relative_paths
    : [];
  const holdoutRelativePathSha256 = createHash("sha256")
    .update(holdoutRelativePaths.join("\n"))
    .digest("hex");
  assert(
    holdout.version ===
      "release_stratified_official_relative_path_hash_v2" &&
      holdout.membership_path_scope ===
        "official_release_relative_path" &&
      holdout.holdout_membership_locked === true &&
      holdoutRelativePaths.length === 820 &&
      new Set(holdoutRelativePaths).size === 820 &&
      JSON.stringify(holdoutRelativePaths) ===
        JSON.stringify([...holdoutRelativePaths].sort()) &&
      holdoutRelativePathSha256 ===
        "16781c97f05cc2bdc94837b0ae96942ac9621174d60775d2c6185dae5fd8a767" &&
      holdout.holdout_relative_path_sha256 ===
        holdoutRelativePathSha256 &&
      report.formal_protocol?.holdout_relative_path_sha256 ===
        holdoutRelativePathSha256 &&
      report.formal_protocol?.holdout_membership_path_scope ===
        "official_release_relative_path",
    "holdout split contract is missing",
  );
  assert(
    holdout.normalization_fit === "training_shards_only",
    "feature normalization was not fit on training shards only",
  );
  assert(
    holdout.tail_taxonomy_scope === "training_shards_fit_applied_to_holdout" &&
      holdout.instruction_rarity_fit === "training_shards_only" &&
      holdout.pt_allocation_fit === "training_shards_only",
    "held-out tail transforms or PT allocation use information outside training shards",
  );
  assert(
    Number(holdout.training_shards) + Number(holdout.holdout_shards) === 4_096 &&
      Number(holdout.holdout_shards) === 820 &&
      Number(holdout.requested_holdout_fraction) === 0.20 &&
      Number(holdout.seed) === 11 &&
      Array.isArray(holdout.per_release) &&
      holdout.per_release.length === 2 &&
      holdout.per_release.every(
        (item) =>
          Number(item.total_shards) === 2_048 &&
          Number(item.holdout_shards) === 410 &&
          Number(item.training_shards) === 1_638,
      ),
    "holdout shard counts are invalid",
  );
  const tailContract = report.tail_score_contract || {};
  assert(
    Number(tailContract.transform_fit_row_count) === Number(holdout.training_shards) &&
      Number(tailContract.allocation_fit_row_count) === Number(holdout.training_shards) &&
      tailContract.instruction_document_frequency_fit ===
        "normalization_fit_rows_only",
    "held-out tail-score fit contract is invalid",
  );
  const ptSource = report.pt_source_audit || {};
  assert(Number(ptSource.count) >= 4_096, "empirical PT source is too small");
  assert(
    ptSource.sha256 ===
      "59e487af80482215b2c2d4e81e9ccd7471ac6c94c1ef40547596ccb80367e75f",
    "empirical PT source SHA-256 is invalid",
  );
  const rareCoverage =
    report.rare_instruction_fingerprint_coverage || {};
  const rareCurve = Array.isArray(rareCoverage.curve)
    ? rareCoverage.curve
    : [];
  const rareTime = Array.isArray(rareCoverage.time_to_coverage)
    ? rareCoverage.time_to_coverage
    : [];
  const rareStatus = rareCoverage.status;
  const rareShapeValid =
    (
      rareStatus === "complete" &&
      Number(rareCoverage.rare_holdout_fingerprint_count) > 0 &&
      JSON.stringify(rareCurve.map((item) => Number(item.draw_budget))) ===
        JSON.stringify([10, 25, 50, 100, 200, 400, 800]) &&
      JSON.stringify(
        rareTime.map((item) => Number(item.coverage_threshold)),
      ) === JSON.stringify([0.10, 0.25, 0.50, 0.75])
    ) ||
    (
      rareStatus === "no_eligible_fingerprints" &&
      Number(rareCoverage.rare_holdout_fingerprint_count) === 0 &&
      Number(rareCoverage.unseen_in_training_fingerprint_count) === 0 &&
      rareCurve.length === 0 &&
      rareTime.length === 0 &&
      String(rareCoverage.status_reason || "").length > 0
    );
  assert(
    rareCoverage.version ===
      "heldout_instruction_fingerprint_coverage_v1" &&
      ["complete", "no_eligible_fingerprints"].includes(rareStatus) &&
      rareShapeValid &&
      rareCoverage.metric_role ===
        "auxiliary_descriptive_metric_not_a_completion_gate" &&
      rareCoverage.rarity_fit_scope === "training_shards_only" &&
      rareCoverage.evaluation_scope === "holdout_shards_only" &&
      String(rareCoverage.claim_boundary).includes(
        "not semantic task coverage",
      ) &&
      Number(rareCoverage.training_shards) === 3_276 &&
      Number(rareCoverage.holdout_shards) === 820 &&
      Number(rareCoverage.max_training_shard_document_frequency) === 1 &&
      String(
        rareCoverage.training_document_frequency_sha256 || "",
      ).length === 64,
    "rare-instruction fingerprint coverage contract is invalid",
  );
  for (const item of rareCurve) {
    const sourceCoverage = Number(item.source_expected_coverage);
    const qtailCoverage = Number(item.qtail_expected_coverage);
    assert(
      Number.isFinite(sourceCoverage) &&
        Number.isFinite(qtailCoverage) &&
        sourceCoverage >= 0 &&
        sourceCoverage <= 1 &&
        qtailCoverage >= 0 &&
        qtailCoverage <= 1 &&
        Math.abs(
          Number(item.gain_pp) -
            (qtailCoverage - sourceCoverage) * 100,
        ) <= 1e-9,
      `rare-instruction coverage row is invalid at budget ${item.draw_budget}`,
    );
  }
  const requirements = Array.isArray(audit.requirements) ? audit.requirements : [];
  const isolation = latest.transport_isolation || {};
  const isolationCumulative = isolation.cumulative || {};
  const isolationAdjudication =
    latest.transport_isolation_adjudication || {};
  const isolationAdjudicationValid =
    isolationAdjudication.status === "adjudicated_transport_epochs_v6" &&
    isolationAdjudication.findings?.length >= 5 &&
    isolationAdjudication.findings.every(
      (finding) => finding.data_transfer_violation === false,
    ) &&
    isolationAdjudication.findings.some(
      (finding) =>
        finding.guard_epoch === "droid_transport_root_environment_v3" &&
        finding.coverage_gap === true,
    ) &&
    isolationAdjudication.remediation?.classifier_version ===
      "droid_transport_downloader_descendants_v6_interface_bound_live" &&
    isolationAdjudication.preservation?.archives?.length >= 5 &&
    isolationAdjudication.preservation.archives.every(
      (archive) =>
        archive.sha256 ===
        isolationAdjudication.archive_hashes_actual?.[archive.path],
    ) &&
    isolationAdjudication.preservation.archives.some(
      (archive) => archive.coverage_gap === true,
    );
  assert(
    isolation.status === "passed" || isolation.status === "passed_idle",
    "UniClash transport isolation is not passing",
  );
  assert(isolation.uniclash?.core_running === true, "UniClashCore is not running");
  assert(isolation.uniclash?.tun_enabled === false, "UniClash TUN is not disabled");
  assert(
    Array.isArray(isolation.policy?.guarded_transports) &&
      isolation.policy.guarded_transports.includes("curl") &&
      isolation.policy.guarded_transports.includes("gsutil"),
    "transport guard does not cover curl and gsutil",
  );
  assert(
    isolation.policy?.process_classifier_version ===
      "droid_transport_downloader_descendants_v6_interface_bound_live",
    "transport guard process classifier is not v6 interface-bound",
  );
  assert(
    Number(isolationCumulative.samples) > 0 &&
      (Number(isolationCumulative.blocked_samples) === 0 ||
        isolationAdjudicationValid) &&
      Number(isolationCumulative.forbidden_socket_observations) === 0 &&
      Number(isolationCumulative.wrong_route_observations) === 0 &&
      (isolationCumulative.blocked_pids || []).length === 0 &&
      ((isolationCumulative.violation_events || []).length === 0 ||
        isolationAdjudicationValid),
    "cumulative UniClash transport isolation audit contains violations",
  );
  assert(
    isolationAdjudicationValid,
    "transport classifier epoch adjudication is invalid",
  );
  const waiting = requirements.filter((item) => !item.passed).map((item) => item.id);
  assert(audit.passed_requirements === 8, `expected 8/9 pre-QA gates, found ${audit.passed_requirements}/9`);
  assert(
    waiting.length === 1 && waiting[0] === "final_page_qa",
    `unexpected pre-QA waiting gates: ${waiting.join(", ")}`,
  );
}

function validateEngineeringPreflight({summary, report}) {
  assert(
    summary.format_version === "qtail_droid_engineering_preflight_v2" &&
    summary.status === "passed_engineering_preflight" &&
      summary.scope === "bounded_test_subset_not_scientific_evidence",
    "engineering preflight status or scope is invalid",
  );
  assert(
      Number(summary.input?.shards) === 8 &&
      Number(summary.input?.shards_per_release) === 4 &&
      Number(summary.input?.releases?.["1.0.0"]) === 4 &&
      Number(summary.input?.releases?.["1.0.1"]) === 4 &&
      Number(summary.input?.records_decoded) === 16 &&
      Number(summary.input?.record_cap_per_shard) === 2 &&
      Number(summary.input?.bytes) > 0 &&
      summary.input?.all_local_md5_recomputed_match_official === true &&
      /^[a-f0-9]{64}$/.test(
        String(summary.input?.frozen_relative_paths_sha256 || ""),
      ) &&
      summary.input?.formal_protocol_locked === false,
    "engineering preflight input scope is invalid",
  );
  const compute = summary.compute || {};
  assert(
    compute.device === "mps" &&
      compute.mps_available === true &&
      compute.same_architecture === true &&
      compute.same_seed === true &&
      compute.same_features === true &&
      compute.same_device === true &&
      compute.same_parameter_count === true &&
      compute.same_environment_fingerprint === true &&
      /^[a-f0-9]{64}$/.test(
        String(compute.runtime_environment_fingerprint || ""),
      ) &&
      Number(compute.source_optimizer_updates) === 50 &&
      Number(compute.qtail_optimizer_updates) === 50,
    "engineering preflight same-compute contract failed",
  );
  const resume = summary.resume || {};
  const resumeStages = Object.values(resume.stages || {});
  assert(
    Number(resume.resumed_stage_count) === 4 &&
      Number(resume.stage_count) === 4 &&
      resume.all_checkpoint_devices_match === true &&
      resume.all_checkpoint_optimizers_match === true &&
      resume.all_environment_fingerprints_match === true &&
      resumeStages.length === 4 &&
      resumeStages.every(
        (stage) =>
          stage.resumed === true &&
          Number(stage.resumed_from_step) === 25 &&
          Number(stage.target_step) === 25 &&
          Number(stage.optimizer_updates_completed) === 25 &&
          stage.device === "mps" &&
          stage.checkpoint_device === "mps" &&
          stage.optimizer === stage.checkpoint_optimizer &&
          stage.environment_fingerprint ===
            compute.checkpoint_environment_fingerprint &&
          stage.checkpoint_environment_fingerprint ===
            compute.checkpoint_environment_fingerprint &&
          Number(stage.checkpoint_format_version) === 6 &&
          stage.checkpoint_chain_version === "sha256_parent_v1" &&
          Array.isArray(stage.resume_rejections) &&
          stage.resume_rejections.length === 0 &&
          typeof stage.training_signature === "string" &&
          stage.training_signature.length === 64,
      ),
    "engineering preflight checkpoint-resume contract failed",
  );
  const checkpointChain = summary.checkpoint_chain || {};
  assert(
    Number(checkpointChain.format_version) === 6 &&
      checkpointChain.chain_version === "sha256_parent_v1" &&
      JSON.stringify(checkpointChain.expected_steps) ===
        JSON.stringify([0, 10, 20, 25]) &&
      Number(checkpointChain.expected_checkpoint_count) === 16 &&
      Number(checkpointChain.actual_checkpoint_count) === 16 &&
      checkpointChain.parent_hash_chains_verified === true &&
      checkpointChain.terminal_resume_preserved_checkpoint_hashes === true &&
      Object.keys(checkpointChain.checkpoint_hashes || {}).length === 16 &&
      Object.values(checkpointChain.checkpoint_hashes || {}).every((value) =>
        /^[a-f0-9]{64}$/.test(String(value)),
      ),
    "engineering preflight checkpoint chain is invalid",
  );
  assert(
    summary.formal_marker_isolation?.marker_dir_argument_passed_to_trainer ===
      false &&
      summary.formal_marker_isolation?.unchanged === true &&
      JSON.stringify(summary.formal_marker_isolation?.before) ===
        JSON.stringify(summary.formal_marker_isolation?.after),
    "engineering preflight touched formal completion markers",
  );
  assert(
    summary.scientific_gate?.passed === false &&
      summary.scientific_gate?.expected_for_completion === false &&
      summary.scientific_gate?.disposition ===
        "withheld_from_formal_claim_and_completion_markers",
    "engineering preflight scientific result was not withheld",
  );
  const reportCompute = report.compute_audit || {};
  const reportCheckpoint = report.intermediate_checkpoint_audit || {};
  const reportCheckpointContract = reportCheckpoint.contract || {};
  const releaseComposition = Object.fromEntries(
    (report.release_composition || []).map((item) => [item.release, item]),
  );
  assert(
    report.status === "complete" &&
    report.training_scope === "bounded_test_subset" &&
      report.formal_protocol?.locked === false &&
      Number(report.shard_count) === 8 &&
      Number(report.total_bytes) === Number(summary.input?.bytes) &&
      Number(releaseComposition["1.0.0"]?.observed_tfrecord_shards) === 4 &&
      Number(releaseComposition["1.0.1"]?.observed_tfrecord_shards) === 4 &&
      Number(releaseComposition["1.0.0"]?.observed_records_decoded) === 8 &&
      Number(releaseComposition["1.0.1"]?.observed_records_decoded) === 8 &&
      reportCompute.training_device === "mps" &&
      reportCompute.mps_available === true &&
      reportCompute.same_architecture === true &&
      reportCompute.same_seed === true &&
      reportCompute.same_features === true &&
      reportCompute.same_device === true &&
      reportCompute.same_parameter_count === true &&
      reportCompute.same_environment_fingerprint === true &&
      reportCompute.runtime_environment_fingerprint ===
        compute.runtime_environment_fingerprint &&
      Number(reportCompute.source_optimizer_updates) === 50 &&
      Number(reportCompute.qtail_optimizer_updates) === 50 &&
      reportCheckpoint.status === "complete" &&
      Number(reportCheckpoint.actual_checkpoint_count) === 16 &&
      Number(reportCheckpointContract.expected_checkpoint_count) === 16 &&
      reportCheckpointContract.checkpoint_format_version === 6 &&
      reportCheckpointContract.checkpoint_chain_version ===
        "sha256_parent_v1" &&
      reportCheckpointContract.parent_checkpoint_hash_chains_verified ===
        true,
    "engineering preflight report does not match its bounded scope",
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  assert(existsSync(args.chrome), `Chrome executable not found: ${args.chrome}`);
  const resultRoot = join(args.jobRoot, "results", "qtail_droid_full");
  const markerRoot = join(args.jobRoot, "manifests");
  const finalMarker = join(markerRoot, "FINAL_PAGE_QA_COMPLETE");
  const previewMarker = join(markerRoot, "FINAL_PAGE_QA_PREVIEW");
  const publicProjectionMarker = join(
    markerRoot,
    "DROID_PUBLIC_PROJECTION_COMMITTED",
  );
  const postcommitMarker = join(
    markerRoot,
    "DROID_POSTCOMMIT_PAGE_QA_COMPLETE",
  );
  const postcommitRunLock = join(
    markerRoot,
    ".DROID_POSTCOMMIT_PAGE_QA_RUN.lock",
  );
  const qaPath = join(resultRoot, "final_page_qa.json");
  const desktopScreenshot = join(resultRoot, "final_page_desktop.png");
  const mobileScreenshot = join(resultRoot, "final_page_mobile.png");
  const postcommitQaPath = join(
    resultRoot,
    "final_page_postcommit_qa.json",
  );
  const postcommitDesktopScreenshot = join(
    resultRoot,
    "final_page_postcommit_desktop.png",
  );
  const postcommitMobileScreenshot = join(
    resultRoot,
    "final_page_postcommit_mobile.png",
  );
  let stalePreviewRemoved = false;
  const staleFinalRemoved = false;
  let releasePostcommitRunLock = null;
  if (args.postCommitReadOnly) {
    assert(
      existsSync(finalMarker) && existsSync(publicProjectionMarker),
      "postcommit QA requires committed final and public projection markers",
    );
    const projectionState = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
        "--job-root",
        args.jobRoot,
        "--stage",
        "final",
        "--validate-projection",
      ],
      {encoding: "utf8"},
    );
    assert(
      projectionState.status === 0,
      `postcommit QA cannot validate the 9/9 projection: ${
        projectionState.stderr || projectionState.stdout
      }`,
    );
    if (existsSync(postcommitMarker)) {
      const completedState = spawnSync(
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        [
          join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
          "--job-root",
          args.jobRoot,
          "--stage",
          "final",
        ],
        {encoding: "utf8"},
      );
      if (completedState.status === 0) {
        process.stdout.write(`${JSON.stringify({
          status: "already_complete_postcommit_read_only",
          marker: postcommitMarker,
        }, null, 2)}\n`);
        return;
      }
      throw new Error(
        "an existing postcommit QA marker is invalid; "
        + "the parent pipeline must invalidate it before rerunning QA",
      );
    }
    const parentCommand = spawnSync(
      "ps",
      ["-p", String(process.ppid), "-o", "command="],
      {encoding: "utf8"},
    ).stdout.trim();
    assert(
      parentCommand.includes("qtail_orico_full_pipeline.sh"),
      "postcommit QA must be owned by qtail_orico_full_pipeline.sh",
    );
    releasePostcommitRunLock = await acquireExclusiveRunLock(
      postcommitRunLock,
    );
    if (existsSync(postcommitMarker)) {
      const completedState = spawnSync(
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        [
          join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
          "--job-root",
          args.jobRoot,
          "--stage",
          "final",
        ],
        {encoding: "utf8"},
      );
      if (completedState.status === 0) {
        await releasePostcommitRunLock();
        releasePostcommitRunLock = null;
        process.stdout.write(`${JSON.stringify({
          status: "already_complete_postcommit_read_only",
          marker: postcommitMarker,
        }, null, 2)}\n`);
        return;
      }
      await releasePostcommitRunLock();
      releasePostcommitRunLock = null;
      throw new Error(
        "an existing postcommit QA marker is invalid; "
        + "the parent pipeline must invalidate it before rerunning QA",
      );
    }
  } else if (!args.smoke) {
    if (existsSync(finalMarker)) {
      const completedState = spawnSync(
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        [
          join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
          "--job-root",
          args.jobRoot,
          "--stage",
          "final",
        ],
        {encoding: "utf8"},
      );
      if (completedState.status === 0) {
        process.stdout.write(`${JSON.stringify({
          status: "already_complete_read_only",
          marker: finalMarker,
          public_projection_marker: publicProjectionMarker,
        }, null, 2)}\n`);
        return;
      }
      throw new Error(
        "an existing final marker is invalid or incomplete; "
        + "the parent pipeline must invalidate it before QA",
      );
    }
    const parentCommand = spawnSync(
      "ps",
      ["-p", String(process.ppid), "-o", "command="],
      {encoding: "utf8"},
    ).stdout.trim();
    assert(
      parentCommand.includes("qtail_orico_full_pipeline.sh"),
      "non-smoke final QA must be owned by qtail_orico_full_pipeline.sh",
    );
    assert(
      !existsSync(publicProjectionMarker),
      "public projection marker exists without a final marker",
    );
    stalePreviewRemoved = existsSync(previewMarker);
    if (stalePreviewRemoved) {
      await unlink(previewMarker);
    }
  }
  if (stalePreviewRemoved) {
    const recoveryRefresh = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_droid_full_progress.py"),
        "--job-root",
        args.jobRoot,
      ],
      {encoding: "utf8"},
    );
    assert(
      recoveryRefresh.status === 0,
      `stale-preview recovery refresh failed: ${recoveryRefresh.stderr || recoveryRefresh.stdout}`,
    );
  }
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: args.chrome,
      headless: true,
    });
  } catch (error) {
    if (releasePostcommitRunLock) {
      await releasePostcommitRunLock();
    }
    throw error;
  }

  if (args.smoke) {
    try {
      validateEngineeringPreflight({
        summary: await readJson(
          join(resultRoot, "droid_preflight_training_smoke.json"),
        ),
        report: await readJson(
          join(resultRoot, "droid_preflight_training_smoke_report.json"),
        ),
      });
      const liveCompletionAudit = await readJson(
        join(resultRoot, "completion_audit.json"),
      );
      const liveIntermediateEvidence =
        (liveCompletionAudit.requirements || []).find(
          (item) => item?.id === "intermediate_artifacts",
        )?.evidence || {};
      const liveProjectionContracts = [
        {
          artifact: "droid_environment_contract_selftest.json",
          field: "environment_selftest_valid",
        },
        {
          artifact: "droid_download_marker_selftest.json",
          field: "download_marker_selftest_valid",
        },
        {
          artifact: "droid_mirror_verifier_selftest.json",
          field: "mirror_verifier_selftest_valid",
        },
        {
          artifact: "droid_training_gate_order_selftest.json",
          field: "training_gate_order_selftest_valid",
        },
        {
          artifact: "droid_downloader_single_writer_selftest.json",
          field: "downloader_single_writer_selftest_valid",
        },
        {
          artifact: "droid_runtime_process_contract_selftest.json",
          field: "runtime_process_contract_selftest_valid",
        },
        {
          artifact: "uniclash_pre_checksum_gate.json",
          field: "uniclash_pre_checksum_gate_valid",
        },
        {
          artifact: "uniclash_pre_checksum_gate_selftest.json",
          field: "uniclash_pre_checksum_gate_selftest_valid",
        },
        {
          artifact: "droid_live_partial_marker_rejection.json",
          field: "live_partial_marker_rejection_valid",
        },
      ];
      const completionProjectionContracts = [];
      for (const contract of liveProjectionContracts) {
        const directArtifact = await readJson(
          join(resultRoot, contract.artifact),
        );
        const projection = liveIntermediateEvidence[contract.field];
        assert(
          directArtifact.status === "passed"
            && projection === true,
          `completion audit projection disagrees with passing artifact: ${
            contract.artifact
          } -> ${contract.field}`,
        );
        completionProjectionContracts.push({
          ...contract,
          direct_status: directArtifact.status,
          projected_valid: projection,
        });
      }
      const views = [];
      views.push(await inspectViewport(browser, {
        pageUrl: args.pageUrl,
        viewport: {width: 1440, height: 1000},
        requireIntermediate: true,
      }));
      views.push(await inspectViewport(browser, {
        pageUrl: args.pageUrl,
        viewport: {width: 390, height: 844},
        requireIntermediate: true,
      }));
      const valueDecisionUiSelftest = await verifyValueDecisionStates(
        browser,
        {
          pageUrl: args.pageUrl,
          baseLatest: await readJson(join(resultRoot, "latest.json")),
          baseTraining: await readJson(
            join(resultRoot, "droid_scalability_canary_full_report.json"),
          ),
        },
      );
      const pageRoot = new URL(".", args.pageUrl);
      const readyArtifactHrefs = [
        ...new Set(
          [
            ...views[0].artifactStates
              .filter((artifact) => artifact.state === "READY")
              .map((artifact) => artifact.href),
            ...views[0].formalArtifactContract.rows
              .filter((artifact) => artifact.state !== "WAIT")
              .map((artifact) => artifact.href),
          ],
        ),
      ].sort();
      const urlProbes = [];
      for (const href of readyArtifactHrefs) {
        const probe = await probeUrl(
          new URL(href, pageRoot).toString(),
        );
        urlProbes.push(probe);
        assert(
          probe.ok,
          `READY artifact URL did not return HTTP 200: ${probe.url}`,
        );
      }
      const smokeReport = {
        generated_at: new Date().toISOString(),
        status: "smoke_passed",
        scope: "live_nonformal_desktop_mobile",
        views,
        value_decision_ui_selftest: valueDecisionUiSelftest,
        completion_projection_contracts: completionProjectionContracts,
        ready_artifact_link_count: readyArtifactHrefs.length,
        url_probes: urlProbes,
      };
      await atomicWriteJson(
        join(resultRoot, "live_page_smoke.json"),
        smokeReport,
      );
      process.stdout.write(`${JSON.stringify(smokeReport, null, 2)}\n`);
      return;
    } finally {
      await browser.close();
      if (releasePostcommitRunLock) {
        await releasePostcommitRunLock();
        releasePostcommitRunLock = null;
      }
    }
  }

  if (args.postCommitReadOnly) {
    const postcommitQa = {
      version: "qtail_droid_postcommit_page_qa_v1",
      generated_at: new Date().toISOString(),
      status: "running",
      scope: "final_public_projection_read_only_browser_qa",
      read_only: true,
      page_url: args.pageUrl,
      repo_root: args.repoRoot,
      job_root: args.jobRoot,
      chrome_executable: args.chrome,
      expected_completion: "9 / 9",
      expected_status: "全部完成",
      bootstrap_views: [],
      final_views: [],
      url_probes: [],
    };
    try {
      await atomicWriteJson(postcommitQaPath, postcommitQa);
      postcommitQa.bootstrap_views.push(await inspectViewport(browser, {
        pageUrl: args.pageUrl,
        viewport: {width: 1440, height: 1000},
        expectedCompletion: "9 / 9",
        expectedStatus: "全部完成",
        requireIntermediate: true,
        requireResults: true,
        screenshotPath: postcommitDesktopScreenshot,
      }));
      postcommitQa.bootstrap_views.push(await inspectViewport(browser, {
        pageUrl: args.pageUrl,
        viewport: {width: 390, height: 844},
        expectedCompletion: "9 / 9",
        expectedStatus: "全部完成",
        requireIntermediate: true,
        requireResults: true,
        screenshotPath: postcommitMobileScreenshot,
      }));
      postcommitQa.status = "sealing";
      await atomicWriteJson(postcommitQaPath, postcommitQa);

      postcommitQa.final_views.push(await inspectViewport(browser, {
        pageUrl: args.pageUrl,
        viewport: {width: 1440, height: 1000},
        expectedCompletion: "9 / 9",
        expectedStatus: "全部完成",
        requireIntermediate: true,
        requireResults: true,
        requireCommitted: true,
        screenshotPath: postcommitDesktopScreenshot,
      }));
      postcommitQa.final_views.push(await inspectViewport(browser, {
        pageUrl: args.pageUrl,
        viewport: {width: 390, height: 844},
        expectedCompletion: "9 / 9",
        expectedStatus: "全部完成",
        requireIntermediate: true,
        requireResults: true,
        requireCommitted: true,
        screenshotPath: postcommitMobileScreenshot,
      }));
      const pageRoot = new URL(".", args.pageUrl);
      for (const relative of [
        "results/qtail_droid_full/final_page_postcommit_qa.json",
        "results/qtail_droid_full/final_page_postcommit_desktop.png",
        "results/qtail_droid_full/final_page_postcommit_mobile.png",
      ]) {
        const probe = await probeUrl(
          new URL(relative, pageRoot).toString(),
        );
        postcommitQa.url_probes.push(probe);
        assert(
          probe.ok,
          `postcommit artifact URL did not return HTTP 200: ${probe.url}`,
        );
      }
      postcommitQa.status = "complete";
      postcommitQa.completed_at = new Date().toISOString();
      postcommitQa.evidence = {
        final_marker: finalMarker,
        final_marker_sha256: await sha256(finalMarker),
        public_projection_marker: publicProjectionMarker,
        public_projection_marker_sha256: await sha256(
          publicProjectionMarker,
        ),
        desktop_screenshot: postcommitDesktopScreenshot,
        mobile_screenshot: postcommitMobileScreenshot,
      };
      await atomicWriteJson(postcommitQaPath, postcommitQa);

      const commit = spawnSync(
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        [
          join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
          "--job-root",
          args.jobRoot,
          "--stage",
          "final",
          "--commit-postcommit-qa",
        ],
        {encoding: "utf8"},
      );
      assert(
        commit.status === 0,
        `postcommit QA marker commit failed: ${
          commit.stderr || commit.stdout
        }`,
      );
      const completeState = spawnSync(
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        [
          join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
          "--job-root",
          args.jobRoot,
          "--stage",
          "final",
        ],
        {encoding: "utf8"},
      );
      assert(
        completeState.status === 0,
        `postcommit 9/9 state did not verify: ${
          completeState.stderr || completeState.stdout
        }`,
      );
      process.stdout.write(`${JSON.stringify({
        status: "postcommit_page_qa_complete",
        marker: postcommitMarker,
        qa: postcommitQaPath,
        screenshots: [
          postcommitDesktopScreenshot,
          postcommitMobileScreenshot,
        ],
      }, null, 2)}\n`);
      return;
    } catch (error) {
      await unlink(postcommitMarker).catch(() => {});
      postcommitQa.status = "failed";
      postcommitQa.failed_at = new Date().toISOString();
      postcommitQa.error = (
        error instanceof Error ? error.message : String(error)
      );
      await atomicWriteJson(postcommitQaPath, postcommitQa).catch(() => {});
      throw error;
    } finally {
      await browser.close();
    }
  }

  let previewCreated = false;
  let markerCreated = false;
  const qa = {
    generated_at: new Date().toISOString(),
    status: "running",
    page_url: args.pageUrl,
    repo_root: args.repoRoot,
    job_root: args.jobRoot,
    chrome_executable: args.chrome,
    playwright_core: JSON.parse(
      await readFile(join(args.repoRoot, "node_modules", "playwright-core", "package.json"), "utf8"),
    ).version,
    stale_preview_removed: stalePreviewRemoved,
    pre_marker_views: [],
    final_views: [],
    url_probes: [],
  };

  try {
    assert(
      existsSync(join(markerRoot, "DROID_TRAINING_COMPLETE")),
      "DROID_TRAINING_COMPLETE marker is missing",
    );
    const reportPath = join(resultRoot, "droid_full_training_report.json");
    const auditPath = join(resultRoot, "completion_audit.json");
    const latestPath = join(resultRoot, "latest.json");
    const artifactManifestPath = join(resultRoot, "droid_artifact_manifest.json");
    const trainingArtifactManifestPath = join(
      resultRoot,
      "droid_training_artifact_manifest.json",
    );
    const releaseMetadataAuditPath = join(
      resultRoot,
      "droid_release_metadata_audit.json",
    );
    const featureCacheManifestPath = join(resultRoot, "droid_feature_cache_manifest.json");
    const featureCachePartialVerificationPath = join(
      resultRoot,
      "droid_feature_cache_partial_verification.json",
    );
    const incrementalClosurePath = join(
      resultRoot,
      "droid_incremental_closure_audit.json",
    );
    const incrementalClosureSelftestPath = join(
      resultRoot,
      "droid_incremental_closure_selftest.json",
    );
    const releaseMilestoneStatusPath = join(
      resultRoot,
      "droid_release_milestone_status.json",
    );
    const releaseMilestonePaths = ["1.0.0", "1.0.1"].map((release) =>
      join(
        resultRoot,
        "release_milestones",
        `droid_release_${release}_complete.json`,
      )
    );
    const featureCacheVerificationPath = join(resultRoot, "droid_feature_cache_verification.json");
    const protocolSelftestPath = join(resultRoot, "droid_protocol_selftest.json");
    const environmentManifestPath = join(
      resultRoot,
      "droid_environment_manifest.json",
    );
    const environmentContractSelftestPath = join(
      resultRoot,
      "droid_environment_contract_selftest.json",
    );
    const prewarmStatusContractSelftestPath = join(
      resultRoot,
      "droid_prewarm_status_contract_selftest.json",
    );
    const downloadMarkerSelftestPath = join(
      resultRoot,
      "droid_download_marker_selftest.json",
    );
    const mirrorVerifierSelftestPath = join(
      resultRoot,
      "droid_mirror_verifier_selftest.json",
    );
    const trainingGateOrderSelftestPath = join(
      resultRoot,
      "droid_training_gate_order_selftest.json",
    );
    const downloaderSingleWriterSelftestPath = join(
      resultRoot,
      "droid_downloader_single_writer_selftest.json",
    );
    const runtimeProcessContractSelftestPath = join(
      resultRoot,
      "droid_runtime_process_contract_selftest.json",
    );
    const preChecksumGatePath = join(
      resultRoot,
      "uniclash_pre_checksum_gate.json",
    );
    const preChecksumGateSelftestPath = join(
      resultRoot,
      "uniclash_pre_checksum_gate_selftest.json",
    );
    const livePartialMarkerRejectionPath = join(
      resultRoot,
      "droid_live_partial_marker_rejection.json",
    );
    const downloadMarkerPath = join(
      args.jobRoot,
      "manifests",
      "DROID_DOWNLOAD_COMPLETE",
    );
    const checkpointPath = join(resultRoot, "qtail_droid_allocation_head.pt");
    const checkpointManifestPath = join(
      resultRoot,
      "droid_intermediate_checkpoint_manifest.json",
    );
    const rareCoveragePath = join(
      resultRoot,
      "droid_rare_instruction_fingerprint_coverage.json",
    );
    const liveGuardPath = join(resultRoot, "uniclash_transport_guard.json");
    const finalGuardSnapshotPath = join(
      resultRoot,
      "uniclash_transport_guard_final.json",
    );
    const liveProgressSamplesPath = join(
      resultRoot,
      "download_progress_samples.json",
    );
    const finalProgressSamplesPath = join(
      resultRoot,
      "download_progress_samples_final.json",
    );
    const processLogManifestPath = join(
      resultRoot,
      "droid_process_log_manifest.json",
    );
    const liveTimelinePath = join(resultRoot, "pipeline_timeline.json");
    const currentTimelineVerificationPath = join(
      resultRoot,
      "pipeline_timeline_current_verification.json",
    );
    const finalTimelinePath = join(
      resultRoot,
      "pipeline_timeline_final.json",
    );
    const finalTimelineVerificationPath = join(
      resultRoot,
      "pipeline_timeline_final_verification.json",
    );
    const liveGuard = await readJson(liveGuardPath);
    await atomicWriteJson(finalGuardSnapshotPath, {
      ...liveGuard,
      immutable_snapshot: {
        captured_at: new Date().toISOString(),
        source: liveGuardPath,
        purpose: "Final artifact integrity; the live guard continues updating.",
      },
    });
    const liveProgressSamples = await readJson(liveProgressSamplesPath);
    await atomicWriteJson(finalProgressSamplesPath, {
      ...liveProgressSamples,
      immutable_snapshot: {
        captured_at: new Date().toISOString(),
        source: liveProgressSamplesPath,
        purpose: "Complete download history at final page QA.",
      },
    });
    const processLogSnapshot = await snapshotProcessLogs({
      jobRoot: args.jobRoot,
      repoRoot: args.repoRoot,
      resultRoot,
    });
    assert(
      processLogSnapshot.manifestPath === processLogManifestPath &&
        processLogSnapshot.entries.length >= 6 &&
        processLogSnapshot.entries.every((entry) =>
          entry.bytes >= 0 &&
          entry.line_count >= 0 &&
          String(entry.sha256 || "").length === 64
        ),
      "final process-log snapshot is incomplete",
    );
    const finalGuardSnapshot = await readJson(finalGuardSnapshotPath);
    const finalGuardAgeSeconds =
      (Date.now() - Date.parse(finalGuardSnapshot.generated_at)) / 1000;
    assert(
      ["passed", "passed_idle"].includes(finalGuardSnapshot.status) &&
        finalGuardSnapshot.policy?.uniclash_core_must_continue === true &&
        finalGuardSnapshot.uniclash?.core_running === true &&
        finalGuardSnapshot.uniclash?.tun_enabled === false &&
        Number(
          finalGuardSnapshot.cumulative?.forbidden_socket_observations,
        ) === 0 &&
        Number(finalGuardSnapshot.cumulative?.wrong_route_observations) ===
          0 &&
        Number.isFinite(finalGuardAgeSeconds) &&
        finalGuardAgeSeconds <= 10 &&
        (finalGuardSnapshot.global_violations || []).length === 0,
      "immutable UniClash transport snapshot does not pass",
    );
    const report = await readJson(reportPath);
    const environmentManifest = await readJson(environmentManifestPath);
    const environmentContractSelftest = await readJson(
      environmentContractSelftestPath,
    );
    await validateEnvironmentCodeBinding({
      environment: environmentManifest,
      selftest: environmentContractSelftest,
      report,
      environmentPath: environmentManifestPath,
    });
    const releaseMetadataAudit = await readJson(releaseMetadataAuditPath);
    validateReleaseMetadataAudit(releaseMetadataAudit);
    const rareCoverageArtifact = await readJson(rareCoveragePath);
    const cacheVerification = await readJson(featureCacheVerificationPath);
    const incrementalClosure = await readJson(incrementalClosurePath);
    validateIncrementalClosure(incrementalClosure);
    const incrementalClosureSelftest = await readJson(
      incrementalClosureSelftestPath,
    );
    assert(
      incrementalClosureSelftest.status === "passed" &&
        incrementalClosureSelftest.format_version ===
          "qtail_droid_incremental_closure_selftest_v2" &&
        Object.keys(incrementalClosureSelftest.checks || {}).length === 7 &&
        Object.values(incrementalClosureSelftest.checks || {}).every(
          (value) => value === true,
        ),
      "incremental closure positive/negative controls failed",
    );
    const releaseMilestoneStatus = await readJson(
      releaseMilestoneStatusPath,
    );
    assert(
      releaseMilestoneStatus.status === "complete" &&
        Number(releaseMilestoneStatus.completed_release_count) === 2 &&
        (releaseMilestoneStatus.releases || []).length === 2 &&
        (releaseMilestoneStatus.releases || []).every(
          (item) =>
            item.status === "complete" &&
            Object.values(item.checks || {}).every(
              (value) => value === true,
            ),
        ),
      "per-release milestone status is incomplete",
    );
    for (const milestonePath of releaseMilestonePaths) {
      const milestone = await readJson(milestonePath);
      assert(
        milestone.status === "complete" &&
          milestone.immutable === true &&
          Object.values(milestone.checks || {}).every(
            (value) => value === true,
          ),
        `release milestone is invalid: ${milestonePath}`,
      );
    }
    const protocolSelftest = await readJson(protocolSelftestPath);
    const prewarmStatusContractSelftest = await readJson(
      prewarmStatusContractSelftestPath,
    );
    validatePrewarmStatusContractSelftest(
      prewarmStatusContractSelftest,
    );
    const downloadMarkerSelftest = await readJson(
      downloadMarkerSelftestPath,
    );
    const mirrorVerifierSelftest = await readJson(
      mirrorVerifierSelftestPath,
    );
    const trainingGateOrderSelftest = await readJson(
      trainingGateOrderSelftestPath,
    );
    const downloaderSingleWriterSelftest = await readJson(
      downloaderSingleWriterSelftestPath,
    );
    const runtimeProcessContractSelftest = await readJson(
      runtimeProcessContractSelftestPath,
    );
    const preChecksumGate = await readJson(preChecksumGatePath);
    const preChecksumGateSelftest = await readJson(
      preChecksumGateSelftestPath,
    );
    const livePartialMarkerRejection = await readJson(
      livePartialMarkerRejectionPath,
    );
    assert(
      downloadMarkerSelftest.status === "passed" &&
        Number(downloadMarkerSelftest.controls_passed) === 8 &&
        Number(downloadMarkerSelftest.controls_total) === 8 &&
        (downloadMarkerSelftest.controls || []).length === 8 &&
        (downloadMarkerSelftest.controls || []).every(
          (control) => control?.passed === true,
        ),
      "download completion marker positive/negative controls failed",
    );
    assert(
      mirrorVerifierSelftest.status === "passed" &&
        Number(mirrorVerifierSelftest.controls_passed) === 8 &&
        Number(mirrorVerifierSelftest.controls_total) === 8 &&
        (mirrorVerifierSelftest.controls || []).length === 8 &&
        (mirrorVerifierSelftest.controls || []).every(
          (control) => control?.passed === true,
        ),
      "final mirror verifier positive/negative controls failed",
    );
    assert(
      trainingGateOrderSelftest.version ===
        "qtail_droid_training_gate_order_selftest_v2" &&
        trainingGateOrderSelftest.status === "passed" &&
        Number(trainingGateOrderSelftest.controls_passed) === 11 &&
        Number(trainingGateOrderSelftest.controls_total) === 11 &&
        (trainingGateOrderSelftest.controls || []).length === 11 &&
        (trainingGateOrderSelftest.controls || []).every(
          (control) => control?.passed === true,
        ),
      "formal pre-optimizer gate-order controls failed",
    );
    assert(
      downloaderSingleWriterSelftest.status === "passed" &&
        Number(downloaderSingleWriterSelftest.checks_passed) === 13 &&
        Number(downloaderSingleWriterSelftest.checks_total) === 13 &&
        Object.keys(downloaderSingleWriterSelftest.checks || {}).length === 13 &&
        Object.values(downloaderSingleWriterSelftest.checks || {}).every(
          (value) => value === true,
        ),
      "downloader single-writer positive/negative controls failed",
    );
    assert(
      runtimeProcessContractSelftest.status === "passed" &&
        runtimeProcessContractSelftest.control ===
          "droid_runtime_process_contract_v11" &&
        Number(runtimeProcessContractSelftest.checks_passed) === 16 &&
        Number(runtimeProcessContractSelftest.checks_total) === 16 &&
        Object.keys(runtimeProcessContractSelftest.checks || {}).length === 16 &&
        Object.values(runtimeProcessContractSelftest.checks || {}).every(
          (value) => value === true,
        ),
      "runtime process positive/negative controls failed",
    );
    assert(
      preChecksumGate.status === "passed" &&
        Number(preChecksumGate.checks_passed) === 10 &&
        Number(preChecksumGate.checks_total) === 10 &&
        Object.keys(preChecksumGate.checks || {}).length === 10 &&
        Object.values(preChecksumGate.checks || {}).every(
          (value) => value === true,
        ),
      "UniClash pre-checksum live gate failed",
    );
    assert(
      preChecksumGateSelftest.status === "passed" &&
        Number(preChecksumGateSelftest.checks_passed) === 13 &&
        Number(preChecksumGateSelftest.checks_total) === 13 &&
        Object.keys(preChecksumGateSelftest.checks || {}).length === 13 &&
        Object.values(preChecksumGateSelftest.checks || {}).every(
          (value) => value === true,
        ),
      "UniClash pre-checksum destructive controls failed",
    );
    assert(
      livePartialMarkerRejection.status === "passed" &&
        livePartialMarkerRejection.formal_completion_evidence === false &&
        livePartialMarkerRejection.precondition?.passed === true &&
        livePartialMarkerRejection.result?.rejected === true &&
        livePartialMarkerRejection.result?.marker_created === false,
      "live partial official mirror was not rejected by the completion gate",
    );
    const downloadMarkerVerification = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_verify_droid_download_marker.py"),
        "--data-dir",
        join(args.jobRoot, "data", "droid"),
        "--manifest",
        join(resultRoot, "droid_object_manifest.json"),
        "--checksum-manifest",
        join(resultRoot, "droid_object_checksum_manifest.json"),
        "--checksum-ledger",
        join(resultRoot, "droid_object_checksum_ledger.json"),
        "--transport-status",
        join(resultRoot, "parallel_download_status.json"),
        "--marker",
        downloadMarkerPath,
        "--expected-bytes",
        "3700745265151",
      ],
      {encoding: "utf8"},
    );
    assert(
      downloadMarkerVerification.status === 0,
      `download completion marker verification failed: ${
        downloadMarkerVerification.stderr ||
        downloadMarkerVerification.stdout
      }`,
    );
    const checkpointManifest = await readJson(checkpointManifestPath);
    const engineeringPreflight = await readJson(
      join(resultRoot, "droid_preflight_training_smoke.json"),
    );
    const engineeringPreflightReport = await readJson(
      join(resultRoot, "droid_preflight_training_smoke_report.json"),
    );
    assert(
      JSON.stringify(rareCoverageArtifact) ===
        JSON.stringify(report.rare_instruction_fingerprint_coverage),
      "rare-fingerprint report and standalone artifact differ",
    );
    validateEngineeringPreflight({
      summary: engineeringPreflight,
      report: engineeringPreflightReport,
    });
    assert(protocolSelftest.status === "passed", "DROID protocol self-test failed");
    const protocolChecks = protocolSelftest.checks || {};
    assert(
      Object.keys(protocolChecks).length === 39 &&
        Object.values(protocolChecks).every((value) => value === true) &&
        protocolChecks.holdout_membership_uses_locked_official_relative_paths ===
          true &&
        protocolChecks.positive_control_supports_hypothesis === true &&
        protocolChecks.outcome_controls_cover_exact_three_states === true &&
        protocolChecks.minimum_effect_ci_boundaries_are_fail_closed === true &&
        protocolChecks.checkpoint_resume_matches_uninterrupted === true &&
        protocolChecks.optimizer_update_boundary_is_exact === true &&
        protocolChecks.mismatched_device_optimizer_checkpoint_rejected ===
          true &&
        protocolChecks.mismatched_environment_checkpoint_rejected === true &&
        protocolChecks.truncated_checkpoint_rejected === true &&
        protocolChecks.overstep_checkpoint_rejected === true &&
        protocolChecks.model_tensor_tamper_checkpoint_rejected === true &&
        protocolChecks.optimizer_moment_tamper_checkpoint_rejected === true &&
        protocolChecks.rare_fingerprint_coverage_exact_expectation === true &&
        protocolChecks.rare_fingerprint_qtail_positive_small_budget_control ===
          true &&
        protocolChecks.rare_fingerprint_rarity_fit_excludes_holdout === true &&
        protocolChecks.rare_fingerprint_claim_boundary_is_explicit === true &&
        protocolChecks.empty_rare_fingerprint_sets_are_explicit_auxiliary_status ===
          true &&
        protocolChecks.intermediate_checkpoint_manifest_exact_grid === true &&
        protocolChecks.same_runtime_different_formal_snapshot_checkpoint_rejected ===
          true &&
        protocolChecks.unexpected_intermediate_checkpoint_rejected === true &&
        protocolChecks.bounded_cli_cannot_publish_formal_completion_marker ===
          true,
      "DROID protocol positive/negative controls are invalid",
    );
    assert(cacheVerification.status === "verified", "feature-cache verification failed");
    assert(
      cacheVerification.all_official_tfrecords === true,
      "feature-cache verification is not full-scope",
    );
    assert(
      cacheVerification.full_official_record_count_match === true,
      "decoded records do not match official shardLengths",
    );
    assert(
      cacheVerification.feature_values_recomputed === true &&
        cacheVerification.all_feature_values_recomputed === true &&
        Number(cacheVerification.recomputed_feature_count) === 4_096 &&
        Number(cacheVerification.error_count) === 0,
      "cached feature values were not independently recomputed",
    );
    assert(cacheVerification.error_count === 0, "feature-cache verification contains errors");
    assert(
      cacheVerification.unreferenced_cache_excluded_from_training === true,
      "unreferenced feature caches are not explicitly excluded",
    );
    const checkpointEntries = Array.isArray(checkpointManifest.entries)
      ? checkpointManifest.entries
      : [];
    const expectedCheckpointPairs = new Set(
      [
        "evaluation_source",
        "evaluation_qtail",
        "deployment_source",
        "deployment_qtail",
      ].flatMap((label) =>
        [0, 5000, 10000, 15000, 20000].map(
          (step) => `${label}:${step}`,
        ),
      ),
    );
    const checkpointFeatureSignatures = Object.fromEntries(
      [
        "evaluation_source",
        "evaluation_qtail",
        "deployment_source",
        "deployment_qtail",
      ].map((label) => [
        label,
        new Set(
          checkpointEntries
            .filter((entry) => entry.model_stage === label)
            .map((entry) => entry.feature_sha256),
        ),
      ]),
    );
    const checkpointInitializationSignatures = new Set(
      checkpointEntries.map((entry) => entry.initialized_state_sha256),
    );
    assert(
      checkpointManifest.status === "complete" &&
        checkpointManifest.actual_checkpoint_count === 20 &&
        checkpointManifest.contract?.expected_checkpoint_count === 20 &&
        checkpointManifest.contract?.checkpoint_format_version === 6 &&
        checkpointManifest.contract?.environment_fingerprint ===
          report.compute_audit?.checkpoint_environment_fingerprint &&
        checkpointManifest.contract?.checkpoint_chain_version ===
          "sha256_parent_v1" &&
        checkpointManifest.contract?.checkpoint_content_hashes_recomputed ===
          true &&
        checkpointManifest.contract?.parent_checkpoint_hash_chains_verified ===
          true &&
        JSON.stringify(checkpointManifest.contract?.expected_steps) ===
          JSON.stringify([0, 5000, 10000, 15000, 20000]) &&
        checkpointManifest.contract?.paired_feature_signatures_equal === true &&
        checkpointManifest.contract?.initialized_state_signatures_equal === true &&
        checkpointEntries.length === 20 &&
        checkpointManifest.errors?.length === 0 &&
        new Set(
          checkpointEntries.map(
            (entry) => `${entry.model_stage}:${entry.step}`,
          ),
        ).size === 20 &&
        checkpointEntries.every((entry) =>
          expectedCheckpointPairs.has(
            `${entry.model_stage}:${entry.step}`,
          ) &&
          entry.optimizer_updates_completed === entry.step &&
          entry.checkpoint_format_version === 6 &&
          entry.environment_fingerprint ===
            report.compute_audit?.checkpoint_environment_fingerprint &&
          entry.checkpoint_chain_version === "sha256_parent_v1" &&
          entry.device === "mps" &&
          entry.optimizer === "AdamW(lr=0.002, weight_decay=0.0001)" &&
          String(entry.training_signature || "").length === 64 &&
          /^[a-f0-9]{64}$/.test(String(entry.feature_sha256 || "")) &&
          /^[a-f0-9]{64}$/.test(
            String(entry.initialized_state_sha256 || ""),
          ) &&
          /^[a-f0-9]{64}$/.test(
            String(entry.model_state_sha256 || ""),
          ) &&
          /^[a-f0-9]{64}$/.test(
            String(entry.optimizer_state_sha256 || ""),
          )
        ) &&
        checkpointFeatureSignatures.evaluation_source.size === 1 &&
        checkpointFeatureSignatures.evaluation_qtail.size === 1 &&
        [...checkpointFeatureSignatures.evaluation_source][0] ===
          [...checkpointFeatureSignatures.evaluation_qtail][0] &&
        checkpointFeatureSignatures.deployment_source.size === 1 &&
        checkpointFeatureSignatures.deployment_qtail.size === 1 &&
        [...checkpointFeatureSignatures.deployment_source][0] ===
          [...checkpointFeatureSignatures.deployment_qtail][0] &&
        checkpointInitializationSignatures.size === 1,
      "intermediate checkpoint manifest is not a complete 4x5 grid",
    );
    for (const label of [
      "evaluation_source",
      "evaluation_qtail",
      "deployment_source",
      "deployment_qtail",
    ]) {
      const entries = checkpointEntries
        .filter((entry) => entry.model_stage === label)
        .sort((left, right) => Number(left.step) - Number(right.step));
      for (let index = 0; index < entries.length; index += 1) {
        const entry = entries[index];
        const parent = index > 0 ? entries[index - 1] : null;
        assert(
          entry.parent_checkpoint_name ===
            (parent ? String(parent.path).split("/").pop() : null) &&
            entry.parent_checkpoint_step ===
              (parent ? parent.step : null) &&
            entry.parent_checkpoint_sha256 ===
              (parent ? parent.sha256 : null),
          `checkpoint parent hash chain mismatch: ${label}:${entry.step}`,
        );
      }
    }
    for (const entry of checkpointEntries) {
      const metadata = await stat(entry.path);
      assert(
        metadata.size === entry.bytes &&
          await sha256(entry.path) === entry.sha256,
        `intermediate checkpoint hash mismatch: ${entry.path}`,
      );
    }

    const stableProcessArtifacts = [
      join(resultRoot, "qtail_orchestration_snapshot", "SHA256SUMS"),
      join(resultRoot, "qtail_orchestration_snapshot_sync_audit.json"),
      join(resultRoot, "droid_source_probe.json"),
      join(resultRoot, "droid_object_manifest.json"),
      join(resultRoot, "droid_object_checksum_manifest.json"),
      releaseMetadataAuditPath,
      join(resultRoot, "droid_object_checksum_ledger.json"),
      join(resultRoot, "parallel_download_status.json"),
      join(resultRoot, "droid_transport_tuning_audit.json"),
      finalProgressSamplesPath,
      join(resultRoot, "download_verification.json"),
      join(resultRoot, "droid_transport_cleanup_audit.json"),
      join(resultRoot, "range_assembly_audit.json"),
      finalGuardSnapshotPath,
      join(resultRoot, "uniclash_transport_guard_adjudication.json"),
      join(
        resultRoot,
        "uniclash_transport_guard_v1_classifier_false_positive.json",
      ),
      join(
        resultRoot,
        "uniclash_transport_guard_v2_descendant_environment_false_positive.json",
      ),
      join(
        resultRoot,
        "uniclash_transport_guard_v3_encoded_path_underobservation.json",
      ),
      join(
        resultRoot,
        "uniclash_transport_guard_classifier_v6_selftest.json",
      ),
      join(
        resultRoot,
        "transport_epochs/uniclash_transport_guard_v4_core_restart_pause.json",
      ),
      join(
        resultRoot,
        "transport_epochs/uniclash_transport_guard_v5_interface_migration_pause.json",
      ),
      featureCachePartialVerificationPath,
      incrementalClosurePath,
      incrementalClosureSelftestPath,
      prewarmStatusContractSelftestPath,
      downloadMarkerSelftestPath,
      mirrorVerifierSelftestPath,
      downloaderSingleWriterSelftestPath,
      runtimeProcessContractSelftestPath,
      preChecksumGatePath,
      preChecksumGateSelftestPath,
      livePartialMarkerRejectionPath,
      releaseMilestoneStatusPath,
      ...releaseMilestonePaths,
      join(resultRoot, "droid_preflight_training_smoke.json"),
      join(resultRoot, "droid_preflight_training_smoke_report.json"),
      join(resultRoot, "droid_forecast_908_summary.json"),
      ...processLogSnapshot.artifacts,
    ].filter((path) => existsSync(path));
    await mergeArtifactManifest(artifactManifestPath, {
      repoRoot: args.repoRoot,
      additions: [
        ...stableProcessArtifacts,
        trainingArtifactManifestPath,
      ],
      exclusions: [
        liveGuardPath,
        liveProgressSamplesPath,
        liveTimelinePath,
      ],
    });
    const preMarkerRefresh = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_droid_full_progress.py"),
        "--job-root",
        args.jobRoot,
      ],
      {encoding: "utf8"},
    );
    assert(
      preMarkerRefresh.status === 0,
      `pre-marker status refresh failed: ${
        preMarkerRefresh.stderr || preMarkerRefresh.stdout
      }`,
    );
    const preMarkerAudit = await readJson(auditPath);
    const preMarkerLatest = await readJson(latestPath);
    assert(
      preMarkerAudit.status === "in_progress" &&
        preMarkerAudit.passed_requirements === 8 &&
        preMarkerAudit.total_requirements === 9,
      `completion audit did not reach the required pre-marker 8/9 state: `
        + `status=${preMarkerAudit.status} `
        + `passed=${preMarkerAudit.passed_requirements}/`
        + `${preMarkerAudit.total_requirements} `
        + `failed=${(preMarkerAudit.requirements || [])
          .filter((item) => item?.passed !== true)
          .map((item) => item?.id || "unknown")
          .join(",")}`,
    );
    validateFinalEvidence({
      report,
      audit: preMarkerAudit,
      latest: preMarkerLatest,
    });

    const criticalRelativeUrls = [
      "",
      "results/qtail_droid_full/latest.json",
      "results/qtail_droid_full/completion_audit.json",
      "results/qtail_droid_full/qtail_orchestration_snapshot/SHA256SUMS",
      "results/qtail_droid_full/qtail_orchestration_snapshot_sync_audit.json",
      "results/qtail_droid_full/droid_source_probe.json",
      "results/qtail_droid_full/droid_object_manifest.json",
      "results/qtail_droid_full/droid_object_checksum_manifest.json",
      "results/qtail_droid_full/droid_release_metadata_audit.json",
      "results/qtail_droid_full/droid_object_checksum_ledger.json",
      "results/qtail_droid_full/parallel_download_status.json",
      "results/qtail_droid_full/droid_transport_tuning_audit.json",
      "results/qtail_droid_full/download_progress_samples_final.json",
      "results/qtail_droid_full/droid_process_log_manifest.json",
      "results/qtail_droid_full/process_logs_final/droid_full_pipeline.log",
      "results/qtail_droid_full/process_logs_final/droid_feature_prewarm.log",
      "results/qtail_droid_full/process_logs_final/pipeline_watchdog.log",
      "results/qtail_droid_full/process_logs_final/progress_loop.log",
      "results/qtail_droid_full/process_logs_final/progress_refresh.log",
      "results/qtail_droid_full/process_logs_final/pipeline_generation_handoff.log",
      "results/qtail_droid_full/process_logs_final/manual_endpoint_generation_handoff.log",
      "results/qtail_droid_full/process_logs_final/qtail_web_services.log",
      "results/qtail_droid_full/process_logs_final/qtail_droid_terminal_launcher.log",
      "results/qtail_droid_full/process_logs_final/qtail_droid_launchd_stderr.log",
      "results/qtail_droid_full/process_logs_final/qtail_droid_launchd_stdout.log",
      "results/qtail_droid_full/process_logs_final/qtail_uniclash_guard_stderr.log",
      "results/qtail_droid_full/process_logs_final/qtail_uniclash_guard_stdout.log",
      "results/qtail_droid_full/process_logs_final/qtail_web_services_local.log",
      "results/qtail_droid_full/pipeline_timeline.json",
      "results/qtail_droid_full/pipeline_timeline_current_verification.json",
      "results/qtail_droid_full/download_verification.json",
      "results/qtail_droid_full/download_completion_marker.json",
      "results/qtail_droid_full/droid_download_marker_selftest.json",
      "results/qtail_droid_full/droid_mirror_verifier_selftest.json",
      "results/qtail_droid_full/droid_training_gate_order_selftest.json",
      "results/qtail_droid_full/droid_downloader_single_writer_selftest.json",
      "results/qtail_droid_full/droid_runtime_process_contract_selftest.json",
      "results/qtail_droid_full/uniclash_pre_checksum_gate.json",
      "results/qtail_droid_full/uniclash_pre_checksum_gate_selftest.json",
      "results/qtail_droid_full/droid_live_partial_marker_rejection.json",
      "results/qtail_droid_full/droid_transport_cleanup_audit.json",
      "results/qtail_droid_full/range_assembly_audit.json",
      "results/qtail_droid_full/uniclash_transport_guard.json",
      "results/qtail_droid_full/uniclash_transport_guard_final.json",
      "results/qtail_droid_full/uniclash_transport_guard_adjudication.json",
      "results/qtail_droid_full/uniclash_transport_guard_v1_classifier_false_positive.json",
      "results/qtail_droid_full/uniclash_transport_guard_v2_descendant_environment_false_positive.json",
      "results/qtail_droid_full/uniclash_transport_guard_v3_encoded_path_underobservation.json",
      "results/qtail_droid_full/uniclash_transport_guard_classifier_v6_selftest.json",
      "results/qtail_droid_full/transport_epochs/uniclash_transport_guard_v4_core_restart_pause.json",
      "results/qtail_droid_full/transport_epochs/uniclash_transport_guard_v5_interface_migration_pause.json",
      "results/qtail_droid_full/empirical_pt_source.csv",
      "results/qtail_droid_full/droid_protocol_selftest.json",
      "results/qtail_droid_full/droid_environment_manifest.json",
      "results/qtail_droid_full/droid_environment_contract_selftest.json",
      "docs/experiments/qtail_droid_full_protocol.md",
      "results/qtail_droid_full/droid_full_run_manifest.json",
      "results/qtail_droid_full/droid_feature_extraction_status.json",
      "results/qtail_droid_full/droid_shard_features.csv",
      "results/qtail_droid_full/droid_model_training_status.json",
      "results/qtail_droid_full/training_status.json",
      "results/qtail_droid_full/droid_training_curve.csv",
      "results/qtail_droid_full/droid_intermediate_checkpoint_manifest.json",
      "results/qtail_droid_full/droid_rare_instruction_fingerprint_coverage.json",
      "results/qtail_droid_full/droid_shard_training_rows.csv",
      "results/qtail_droid_full/droid_full_training_report.json",
      "results/qtail_droid_full/droid_training_artifact_manifest.json",
      "results/qtail_droid_full/droid_artifact_manifest.json",
      "results/qtail_droid_full/droid_feature_cache_manifest.json",
      "results/qtail_droid_full/droid_feature_cache_partial_verification.json",
      "results/qtail_droid_full/droid_incremental_closure_audit.json",
      "results/qtail_droid_full/droid_incremental_closure_selftest.json",
      "results/qtail_droid_full/droid_release_milestone_status.json",
      "results/qtail_droid_full/release_milestones/droid_release_1.0.0_complete.json",
      "results/qtail_droid_full/release_milestones/droid_release_1.0.1_complete.json",
      "results/qtail_droid_full/droid_feature_cache_verification.json",
      "results/qtail_droid_full/droid_preflight_training_smoke.json",
      "results/qtail_droid_full/droid_preflight_training_smoke_report.json",
      "results/qtail_droid_full/droid_forecast_908_summary.json",
      "results/qtail_droid_full/qtail_droid_allocation_head.pt",
    ];
    const pageRoot = new URL(".", args.pageUrl);
    const artifactUrls = await collectArtifactUrls(browser, args.pageUrl);
    qa.displayed_artifact_link_count = artifactUrls.length;
    const deferredArtifactUrls = new Set([
      new URL(
        "results/qtail_droid_full/final_page_qa.json",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/final_page_desktop.png",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/final_page_mobile.png",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/pipeline_timeline_final.json",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/pipeline_timeline_final_verification.json",
        pageRoot,
      ).toString(),
    ]);
    const postCommitArtifactUrls = new Set([
      new URL(
        "results/qtail_droid_full/latest_final.json",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/completion_audit_final.json",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/final_page_postcommit_qa.json",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/final_page_postcommit_desktop.png",
        pageRoot,
      ).toString(),
      new URL(
        "results/qtail_droid_full/final_page_postcommit_mobile.png",
        pageRoot,
      ).toString(),
    ]);
    const urlsToProbe = new Set(
      artifactUrls.filter(
        (url) =>
          !deferredArtifactUrls.has(url)
          && !postCommitArtifactUrls.has(url)
          && !(
            url.endsWith(
              "/results/qtail_forecast_908_20260728/run/droid_full_training_report.json",
            )
            && !existsSync(join(
              args.repoRoot,
              "results/qtail_forecast_908_20260728/run/droid_full_training_report.json",
            ))
          )
          && !(
            url.endsWith(
              "/results/qtail_forecast_908_20260728/run/droid_intermediate_checkpoint_manifest.json",
            )
            && !existsSync(join(
              args.repoRoot,
              "results/qtail_forecast_908_20260728/run/droid_intermediate_checkpoint_manifest.json",
            ))
          ),
      ),
    );
    for (const relative of criticalRelativeUrls) {
      urlsToProbe.add(
        relative ? new URL(relative, pageRoot).toString() : args.pageUrl,
      );
    }
    for (const url of [...urlsToProbe].sort()) {
      const probe = await probeUrl(url);
      qa.url_probes.push(probe);
      assert(probe.ok, `critical artifact URL did not return HTTP 200: ${url}`);
    }

    qa.pre_marker_views.push(await inspectViewport(browser, {
      pageUrl: args.pageUrl,
      viewport: {width: 1440, height: 1000},
      expectedCompletion: "8 / 9",
      requireIntermediate: true,
      requireResults: false,
    }));
    qa.pre_marker_views.push(await inspectViewport(browser, {
      pageUrl: args.pageUrl,
      viewport: {width: 390, height: 844},
      expectedCompletion: "8 / 9",
      requireIntermediate: true,
      requireResults: false,
    }));

    const previewPayload = {
      status: "preview_active",
      owner_pid: process.ppid,
      verifier_pid: process.pid,
      generated_at: new Date().toISOString(),
      expires_at: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
      contract: (
        "Effective only while the parent qtail_orico_full_pipeline.sh "
        + "process is alive and before expires_at."
      ),
    };
    await atomicWriteJson(previewMarker, previewPayload);
    previewCreated = true;
    const bootstrapCommit = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
        "--job-root",
        args.jobRoot,
        "--stage",
        "final",
        "--commit-bootstrap",
      ],
      {encoding: "utf8"},
    );
    assert(
      bootstrapCommit.status === 0,
      `final QA bootstrap commit failed: ${
        bootstrapCommit.stderr || bootstrapCommit.stdout
      }`,
    );
    markerCreated = true;
    const refresh = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_droid_full_progress.py"),
        "--job-root",
        args.jobRoot,
      ],
      {encoding: "utf8"},
    );
    assert(refresh.status === 0, `final status refresh failed: ${refresh.stderr || refresh.stdout}`);

    const finalLatest = await readJson(latestPath);
    const finalAudit = await readJson(auditPath);
    assert(finalLatest.status === "in_progress", "latest.json left in_progress before commit");
    assert(finalLatest.stage === "final_page_qa", "latest.json left the final QA stage before commit");
    assert(finalAudit.status === "in_progress", "completion audit completed before final commit");
    assert(finalAudit.passed_requirements === 8, "sealing completion audit is not 8/9");
    assert(
      finalAudit.requirements?.find((item) => item.id === "final_page_qa")?.passed === false,
      "final page QA requirement passed before committed marker",
    );
    await atomicCopy(finalTimelinePath, liveTimelinePath);
    const timelineVerification = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_verify_droid_timeline.py"),
        "--timeline",
        finalTimelinePath,
        "--out",
        finalTimelineVerificationPath,
        "--require-final",
      ],
      {encoding: "utf8"},
    );
    assert(
      timelineVerification.status === 0,
      `final timeline verification failed: ${
        timelineVerification.stderr || timelineVerification.stdout
      }`,
    );
    const finalTimelineVerification = await readJson(finalTimelineVerificationPath);
    const dataContinuity = finalTimelineVerification.data_continuity;
    assert(dataContinuity && typeof dataContinuity === "object", "data continuity audit is missing");
    assert(
      ["passed", "repair_events_observed"].includes(dataContinuity.status),
      `unexpected data continuity status: ${dataContinuity.status}`,
    );
    for (const field of [
      "completed_object_decrease_events",
      "verified_object_decrease_events",
      "checksum_error_samples",
      "legacy_physical_byte_decrease_events",
      "feature_pass_reset_events",
      "committed_feature_counter_decrease_events",
    ]) {
      assert(
        Number.isInteger(dataContinuity[field]) && dataContinuity[field] >= 0,
        `invalid data continuity count: ${field}`,
      );
    }
    assert(
      dataContinuity.committed_feature_counter_decrease_events === 0,
      "committed feature counter decreased in final timeline",
    );
    assert(
      typeof dataContinuity.claim_boundary === "string"
        && dataContinuity.claim_boundary.length > 0,
      "data continuity claim boundary is missing",
    );

    qa.final_views.push(await inspectViewport(browser, {
      pageUrl: args.pageUrl,
      viewport: {width: 1440, height: 1000},
      expectedCompletion: "8 / 9",
      expectedStatus: "终态证据封存中",
      requireIntermediate: true,
      requireResults: false,
      screenshotPath: desktopScreenshot,
    }));
    qa.final_views.push(await inspectViewport(browser, {
      pageUrl: args.pageUrl,
      viewport: {width: 390, height: 844},
      expectedCompletion: "8 / 9",
      expectedStatus: "终态证据封存中",
      requireIntermediate: true,
      requireResults: false,
      screenshotPath: mobileScreenshot,
    }));
    assert(
      qa.final_views.every((view) => view.status === "终态证据封存中"),
      "page status did not enter the honest sealing state",
    );

    qa.status = "complete";
    qa.completed_at = new Date().toISOString();
    qa.evidence = {
      formal_training_report: reportPath,
      release_metadata_audit: releaseMetadataAuditPath,
      completion_audit: auditPath,
      training_checkpoint: checkpointPath,
      intermediate_checkpoint_manifest: checkpointManifestPath,
      forecast_908_summary: join(
        resultRoot,
        "droid_forecast_908_summary.json",
      ),
      rare_instruction_fingerprint_coverage: rareCoveragePath,
      feature_cache_manifest: featureCacheManifestPath,
      feature_cache_partial_verification: featureCachePartialVerificationPath,
      incremental_closure: incrementalClosurePath,
      incremental_closure_selftest: incrementalClosureSelftestPath,
      download_completion_marker: downloadMarkerPath,
      download_marker_selftest: downloadMarkerSelftestPath,
      mirror_verifier_selftest: mirrorVerifierSelftestPath,
      training_gate_order_selftest: trainingGateOrderSelftestPath,
      downloader_single_writer_selftest: (
        downloaderSingleWriterSelftestPath
      ),
      runtime_process_contract_selftest: (
        runtimeProcessContractSelftestPath
      ),
      uniclash_pre_checksum_gate: preChecksumGatePath,
      uniclash_pre_checksum_gate_selftest: preChecksumGateSelftestPath,
      live_partial_marker_rejection: livePartialMarkerRejectionPath,
      release_milestone_status: releaseMilestoneStatusPath,
      release_milestones: releaseMilestonePaths,
      feature_cache_verification: featureCacheVerificationPath,
      uniclash_transport_guard: liveGuardPath,
      uniclash_transport_guard_final: finalGuardSnapshotPath,
      download_progress_samples_final: finalProgressSamplesPath,
      process_log_manifest: processLogManifestPath,
      pipeline_timeline_final: finalTimelinePath,
      pipeline_timeline_final_verification: finalTimelineVerificationPath,
      pipeline_timeline_current_verification: (
        currentTimelineVerificationPath
      ),
      uniclash_transport_guard_adjudication: join(
        resultRoot,
        "uniclash_transport_guard_adjudication.json",
      ),
      desktop_screenshot: desktopScreenshot,
      mobile_screenshot: mobileScreenshot,
      marker_commit_protocol: (
        "The lease-bound bootstrap remains a non-complete 8/9 sealing state. "
        + "The parent pipeline replaces it with a committed final marker only "
        + "after QA, screenshots, timeline, logs, and hashes pass."
      ),
    };
    await atomicWriteJson(qaPath, qa);
    for (const url of [...deferredArtifactUrls].sort()) {
      const probe = await probeUrl(url);
      qa.url_probes.push(probe);
      assert(
        probe.ok,
        `deferred final artifact URL did not return HTTP 200: ${url}`,
      );
    }
    await atomicWriteJson(qaPath, qa);

    await mergeArtifactManifest(artifactManifestPath, {
      repoRoot: args.repoRoot,
      additions: [
        qaPath,
        desktopScreenshot,
        mobileScreenshot,
        finalTimelinePath,
        finalTimelineVerificationPath,
      ],
      exclusions: [
        liveGuardPath,
        liveProgressSamplesPath,
        liveTimelinePath,
      ],
    });
    const finalRefresh = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_droid_full_progress.py"),
        "--job-root",
        args.jobRoot,
      ],
      {encoding: "utf8"},
    );
    assert(
      finalRefresh.status === 0,
      `post-manifest status refresh failed: ${finalRefresh.stderr || finalRefresh.stdout}`,
    );
    const finalMarkerContract = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_verify_droid_stage_markers.py"),
        "--job-root",
        args.jobRoot,
        "--stage",
        "final",
        "--print-paths",
      ],
      {encoding: "utf8"},
    );
    assert(
      finalMarkerContract.status === 0,
      `final marker path contract failed: ${
        finalMarkerContract.stderr || finalMarkerContract.stdout
      }`,
    );
    const finalMarkerArtifacts = JSON.parse(finalMarkerContract.stdout).paths;
    assert(
      Array.isArray(finalMarkerArtifacts) &&
        finalMarkerArtifacts.length >= 10 &&
        finalMarkerArtifacts.every((path) => typeof path === "string"),
      "final marker path contract returned an invalid artifact set",
    );
    await Promise.all(finalMarkerArtifacts.map(artifactEntry));
    const bootstrapMarker = await readJson(finalMarker);
    assert(
      bootstrapMarker.marker_version === "droid_final_page_qa_bootstrap_v1"
        && bootstrapMarker.status === "sealing",
      "final QA bootstrap marker was replaced before terminal log sealing",
    );
    const committedRefresh = spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_droid_full_progress.py"),
        "--job-root",
        args.jobRoot,
      ],
      {encoding: "utf8"},
    );
    assert(
      committedRefresh.status === 0,
      `committed status refresh failed: ${committedRefresh.stderr || committedRefresh.stdout}`,
    );
    const committedAudit = await readJson(auditPath);
    assert(
      committedAudit.status === "in_progress" &&
        committedAudit.passed_requirements === 8,
      "precommit completion audit did not remain at sealing 8/9",
    );
    process.stdout.write(`${JSON.stringify({
      status: "qa_complete_waiting_final_commit",
      marker: finalMarker,
      qa: qaPath,
      screenshots: [desktopScreenshot, mobileScreenshot],
    }, null, 2)}\n`);
  } catch (error) {
    if (previewCreated || existsSync(previewMarker)) {
      await unlink(previewMarker).catch(() => {});
    }
    if (markerCreated) {
      await unlink(finalMarker).catch(() => {});
    }
    qa.status = "failed";
    qa.failed_at = new Date().toISOString();
    qa.error = error instanceof Error ? error.message : String(error);
    await atomicWriteJson(qaPath, qa).catch(() => {});
    const recoveryManifestPath = join(
      resultRoot,
      "droid_artifact_manifest.json",
    );
    if (existsSync(recoveryManifestPath) && existsSync(qaPath)) {
      await mergeArtifactManifest(recoveryManifestPath, {
        repoRoot: args.repoRoot,
        additions: [qaPath],
        exclusions: [
          join(resultRoot, "uniclash_transport_guard.json"),
          join(resultRoot, "download_progress_samples.json"),
          join(resultRoot, "pipeline_timeline.json"),
        ],
      }).catch(() => {});
    }
    spawnSync(
      "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
      [
        join(args.repoRoot, "tools", "qtail_droid_full_progress.py"),
        "--job-root",
        args.jobRoot,
      ],
      {encoding: "utf8"},
    );
    throw error;
  } finally {
    await browser.close();
  }
}

export {
  snapshotProcessLogs,
  validateFinalEvidence,
  validatePrewarmStatusContractSelftest,
};

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  main().catch((error) => {
    process.stderr.write(`${error.stack || error.message || String(error)}\n`);
    process.exit(1);
  });
}
