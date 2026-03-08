// Segment textbook PDFs into labeled boxes using pdfjs-dist and a lightweight microcortex classifier.
// Outputs page PNGs with overlays and JSON metadata per page.

import fs from 'node:fs/promises';
import path from 'node:path';
import { createCanvas } from '@napi-rs/canvas';
import { pathToFileURL } from 'node:url';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import { buildPerceptron, softmax } from '../../packages/microcortex/src/index.js';

const TEXTBOOK_DIR = path.join(process.cwd(), 'textbooks');
const OUTPUT_DIR = path.join(process.cwd(), 'textbooks', 'output');
const MAX_PAGES = process.env.MAX_PAGES ? Number(process.env.MAX_PAGES) : null;
const BOOK_TIMEOUT_MS = process.env.BOOK_TIMEOUT_MS ? Number(process.env.BOOK_TIMEOUT_MS) : 30 * 60 * 1000; // 30 min per book default
const PAGE_TIMEOUT_MS = process.env.PAGE_TIMEOUT_MS ? Number(process.env.PAGE_TIMEOUT_MS) : 5 * 60 * 1000; // 5 min per page default
const MAX_BOOKS = process.env.MAX_BOOKS ? Number(process.env.MAX_BOOKS) : null;
const BOOK_FILTER = process.env.BOOK_FILTER ? String(process.env.BOOK_FILTER).toLowerCase() : null;
const FORCE_REPROCESS = process.env.FORCE_REPROCESS === 'true';
const WORKER_SRC = path.join(process.cwd(), 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.mjs');
const LOG_PATH = path.join(OUTPUT_DIR, 'segmentation-run.log');

pdfjsLib.GlobalWorkerOptions.workerSrc = pathToFileURL(WORKER_SRC).href;

const ensureDir = async (dir) => {
  await fs.mkdir(dir, { recursive: true });
};

const appendLog = async (message) => {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  try {
    await fs.appendFile(LOG_PATH, line, 'utf-8');
  } catch {
    // Best-effort logging only.
  }
  console.log(message);
};

const withTimeout = async (promise, ms, label) => {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`timeout: ${label}`)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
};

const createClassifier = () => {
  // Feature order: [fontSizeNorm, heightNorm, widthNorm, yNorm, isAllCaps, density, lineCountNorm]
  const inputSize = 7;
  const hiddenSize = 10;
  const outputSize = 7; // title, subtitle, heading, paragraph, list, callout, footer

  // Heuristic weights crafted for textbook-ish layouts.
  const hiddenWeights = [
    // fontSizeNorm, h, w, y, caps, density, lines
    1.8, 0.4, -0.6, -0.2, 0.4, -0.2, -0.3, // neuron 0
    1.3, 0.5, -0.4, -0.1, 0.8, -0.1, -0.2, // neuron 1
    -0.2, 0.3, 0.9, -0.3, -0.4, 0.6, 0.4,  // neuron 2
    -0.5, 0.2, 0.8, -0.5, -0.2, 0.9, 0.6,  // neuron 3
    0.6, 0.1, -0.2, 0.5, -0.6, -0.5, 0.7,  // neuron 4
    0.9, -0.3, -0.8, 0.6, 0.1, -0.3, -0.6, // neuron 5
    -0.3, 0.7, 0.6, -0.2, -0.5, 0.8, 0.9,  // neuron 6
    0.4, 0.2, 0.2, 0.8, -0.3, -0.2, -0.1,  // neuron 7
    -0.1, 0.1, 0.5, 0.4, -0.2, 0.2, 0.5,   // neuron 8
    0.2, -0.2, 0.2, -0.7, -0.4, -0.1, -0.3 // neuron 9
  ];

  const hiddenBiases = [0.4, 0.2, -0.1, -0.2, 0.1, 0.05, 0.0, 0.1, 0.0, -0.05];

  const outputWeights = [
    // title
    0.9, 1.1, -0.5, -0.6, 0.6, -0.2, -0.4, 0.3, -0.3, -0.2,
    // subtitle
    0.6, 0.7, -0.4, -0.5, 0.4, -0.1, -0.3, 0.2, -0.2, -0.1,
    // heading
    0.3, 0.5, -0.1, -0.2, 0.5, -0.2, -0.2, 0.3, -0.1, -0.1,
    // paragraph
    -0.4, -0.3, 0.9, 0.1, -0.4, 0.8, 0.6, -0.2, 0.3, 0.1,
    // list
    -0.5, -0.4, 0.8, 0.2, -0.3, 0.9, 0.7, -0.1, 0.4, 0.2,
    // callout
    0.2, 0.1, 0.5, -0.1, 0.0, 0.6, 0.4, 0.2, 0.2, 0.1,
    // footer
    -0.3, -0.2, 0.4, 0.8, -0.5, -0.2, -0.1, 0.1, 0.0, -0.2
  ];

  const outputBiases = [0.2, 0.1, 0.05, 0.0, 0.0, -0.05, -0.1];

  return buildPerceptron({
    inputSize,
    hiddenSize,
    outputSize,
    hiddenWeights,
    hiddenBiases,
    outputWeights,
    outputBiases
  });
};

const classifier = createClassifier();

const LABELS = ['title', 'subtitle', 'heading', 'paragraph', 'list', 'callout', 'footer'];
const COLORS = {
  title: '#f97316',
  subtitle: '#f59e0b',
  heading: '#22d3ee',
  paragraph: '#34d399',
  list: '#a855f7',
  callout: '#fb7185',
  footer: '#94a3b8',
  default: '#e2e8f0'
};

const buildFeatureVector = (box, pageHeight) => {
  const fontSizeNorm = Math.min(box.fontSize / 36, 1);
  const heightNorm = Math.min(box.height / 300, 1);
  const widthNorm = Math.min(box.width / 800, 1);
  const yNorm = Math.max(0, Math.min(1, 1 - box.y / pageHeight));
  const isAllCaps = box.text === box.text.toUpperCase() ? 1 : 0;
  const wordCount = Math.max(1, box.text.split(/\s+/).length);
  const density = Math.min(wordCount / Math.max(1, (box.width * box.height) / 10000), 1);
  const lineCountNorm = Math.min((box.lines ?? 1) / 12, 1);
  return new Float32Array([fontSizeNorm, heightNorm, widthNorm, yNorm, isAllCaps, density, lineCountNorm]);
};

const classifyBox = (box, pageHeight) => {
  const features = buildFeatureVector(box, pageHeight);
  const logits = classifier.forward(features);
  const probs = softmax(Array.from(logits));
  const maxIdx = probs.reduce((bestIdx, v, idx, arr) => (v > arr[bestIdx] ? idx : bestIdx), 0);
  return { label: LABELS[maxIdx], confidence: probs[maxIdx] };
};

const groupTextItems = (items) => {
  const lines = [];
  const sorted = [...items].sort((a, b) => a.y - b.y);
  const lineThreshold = 6;

  sorted.forEach((item) => {
    const line = lines.find((l) => Math.abs(l.y - item.y) < lineThreshold);
    if (line) {
      line.items.push(item);
      line.y = Math.min(line.y, item.y);
      line.maxY = Math.max(line.maxY, item.y + item.height);
    } else {
      lines.push({ y: item.y, maxY: item.y + item.height, items: [item] });
    }
  });

  const blocks = [];
  let current = null;
  lines
    .sort((a, b) => a.y - b.y)
    .forEach((line) => {
      const lineText = line.items.map((i) => i.text).join(' ');
      const minX = Math.min(...line.items.map((i) => i.x));
      const maxX = Math.max(...line.items.map((i) => i.x + i.width));
      if (!current) {
        current = { lines: [lineText], minX, maxX, minY: line.y, maxY: line.maxY, fontSizes: line.items.map((i) => i.fontSize) };
        return;
      }
      const verticalGap = line.y - current.maxY;
      const horizontalOverlap = Math.min(current.maxX, maxX) - Math.max(current.minX, minX);
      const minWidth = Math.min(current.maxX - current.minX, maxX - minX);
      const overlapRatio = minWidth > 0 ? Math.max(0, horizontalOverlap) / minWidth : 0;
      if (verticalGap < 16 && overlapRatio > 0.25) {
        current.lines.push(lineText);
        current.minX = Math.min(current.minX, minX);
        current.maxX = Math.max(current.maxX, maxX);
        current.maxY = Math.max(current.maxY, line.maxY);
        current.fontSizes.push(...line.items.map((i) => i.fontSize));
      } else {
        blocks.push(current);
        current = { lines: [lineText], minX, maxX, minY: line.y, maxY: line.maxY, fontSizes: line.items.map((i) => i.fontSize) };
      }
    });
  if (current) blocks.push(current);
  return blocks.map((b) => ({
    x: b.minX,
    y: b.minY,
    width: b.maxX - b.minX,
    height: b.maxY - b.minY,
    text: b.lines.join(' '),
    lines: b.lines.length,
    fontSize: median(b.fontSizes)
  }));
};

const median = (arr) => {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
};

const extractTextItems = (textContent, viewport) => {
  return textContent.items.map((item) => {
    const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
    const fontSize = Math.hypot(tx[2], tx[3]);
    const width = item.width * viewport.scale;
    const height = fontSize;
    const x = tx[4];
    const y = tx[5] - height; // move from baseline to top
    return {
      text: item.str,
      x,
      y,
      width,
      height,
      fontSize
    };
  });
};

const renderPage = async (page, scale = 2) => {
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(viewport.width, viewport.height);
  const context = canvas.getContext('2d');
  const renderContext = {
    canvasContext: context,
    viewport
  };
  await page.render(renderContext).promise;
  return { canvas, context, viewport };
};

const writePagesMeta = async (dir, meta) => {
  const pagesJsonPath = path.join(dir, 'pages.json');
  const tmpPath = `${pagesJsonPath}.tmp`;
  const payload = JSON.stringify(meta, null, 2);
  const attempt = async () => {
    await fs.writeFile(tmpPath, payload, 'utf-8');
    await fs.rename(tmpPath, pagesJsonPath);
  };
  try {
    await attempt();
  } catch {
    await new Promise((resolve) => setTimeout(resolve, 50));
    await attempt();
  }
};

const drawBoxes = (context, boxes) => {
  boxes.forEach((box) => {
    const color = COLORS[box.label] ?? COLORS.default;
    context.save();
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.fillStyle = 'rgba(0,0,0,0.45)';
    context.fillRect(box.x - 2, box.y - 2, box.width + 4, box.height + 4);
    context.strokeRect(box.x, box.y, box.width, box.height);
    context.fillStyle = color;
    context.font = '12px "IBM Plex Mono", monospace';
    context.fillText(`${box.label} ${(box.confidence * 100).toFixed(0)}%`, box.x + 4, box.y + 14);
    context.restore();
  });
};

const processPdf = async (pdfPath) => {
  const basename = path.basename(pdfPath, path.extname(pdfPath));
  const pdfData = new Uint8Array(await fs.readFile(pdfPath));
  const doc = await pdfjsLib.getDocument({ data: pdfData }).promise;
  const pdfPageCount = doc.numPages;
  const pdfOutputDir = path.join(OUTPUT_DIR, basename);
  await ensureDir(pdfOutputDir);

  // Skip if already processed (pages.json exists)
  const pagesJsonPath = path.join(pdfOutputDir, 'pages.json');
  try {
    const raw = await fs.readFile(pagesJsonPath, 'utf-8');
    const parsed = JSON.parse(raw);
    const status = parsed.status ?? 'completed';
    if (!FORCE_REPROCESS && status === 'completed' && parsed.totalPages === pdfPageCount && parsed.usablePages === pdfPageCount) {
      await appendLog(`Skipping already processed ${basename}`);
      return;
    }
    if (FORCE_REPROCESS) {
      await appendLog(`Reprocessing ${basename} (forcing overwrite)`);
    }
  } catch {
    // proceed when pages.json missing or unreadable
  }

  const totalPages = MAX_PAGES ? Math.min(MAX_PAGES, pdfPageCount) : pdfPageCount;
  const pad = Math.max(4, String(totalPages).length);
  const textByPage = [];
  let lastPage = 0;
  let nullPages = 0;
  let processedPages = 0;

  await appendLog(`Starting ${basename} (pages=${totalPages})`);
  await writePagesMeta(pdfOutputDir, {
    status: 'in-progress',
    totalPages: pdfPageCount,
    processedPages,
    nullPages,
    usablePages: processedPages - nullPages,
    lastPage,
    timestamp: new Date().toISOString()
  });

  try {
    for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
      await withTimeout(
        (async () => {
          const page = await doc.getPage(pageNum);
          const { canvas, context, viewport } = await renderPage(page, 2);
          const textContent = await page.getTextContent();
          const items = extractTextItems(textContent, viewport);
          const grouped = groupTextItems(items);
          const classified = grouped.map((block) => {
            const { label, confidence } = classifyBox(block, viewport.height);
            return { ...block, label, confidence };
          });
          drawBoxes(context, classified);

          const png = await canvas.encode('png');
          const suffix = String(pageNum).padStart(pad, '0');
          const pagePngPath = path.join(pdfOutputDir, `page-${suffix}.png`);
          const pageJsonPath = path.join(pdfOutputDir, `page-${suffix}.json`);
          await fs.writeFile(pagePngPath, png);
          await fs.writeFile(
            pageJsonPath,
            JSON.stringify({ page: pageNum, pageHeight: viewport.height, boxes: classified }, null, 2),
            'utf-8'
          );
          const isNullPage = classified.length === 0;
          if (isNullPage) nullPages += 1;
          processedPages += 1;
          lastPage = pageNum;
          textByPage.push({
            page: pageNum,
            blocks: grouped.map((b) => b.text)
          });
          await writePagesMeta(pdfOutputDir, {
            status: 'in-progress',
            totalPages: pdfPageCount,
            processedPages,
            nullPages,
            usablePages: processedPages - nullPages,
            lastPage,
            timestamp: new Date().toISOString()
          });
        })(),
        PAGE_TIMEOUT_MS,
        `${basename} page ${pageNum}`
      );

      if (pageNum === totalPages || pageNum % 25 === 0) {
        await appendLog(`Processed ${basename} page ${pageNum}/${totalPages}`);
      }
    }
  } catch (err) {
    const stuckAt = lastPage === 0 ? 'start' : `page ${lastPage}`;
    await appendLog(`Error while processing ${basename} at ${stuckAt}: ${err.message}`);
    throw err;
  }

  await writePagesMeta(pdfOutputDir, {
    status: 'completed',
    totalPages: pdfPageCount,
    processedPages,
    nullPages,
    usablePages: processedPages - nullPages,
    lastPage,
    timestamp: new Date().toISOString()
  });
  await fs.writeFile(path.join(pdfOutputDir, 'pages-text.json'), JSON.stringify(textByPage, null, 2), 'utf-8');
  await appendLog(`Completed ${basename} (${totalPages} pages)`);
};

const main = async () => {
  await ensureDir(OUTPUT_DIR);
  await appendLog(
    `Run started (MAX_PAGES=${MAX_PAGES ?? 'all'}, MAX_BOOKS=${MAX_BOOKS ?? 'all'}, BOOK_FILTER=${BOOK_FILTER ?? 'none'}, BOOK_TIMEOUT_MS=${BOOK_TIMEOUT_MS}, PAGE_TIMEOUT_MS=${PAGE_TIMEOUT_MS})`
  );
  const entries = await fs.readdir(TEXTBOOK_DIR);
  let pdfs = entries.filter((f) => f.toLowerCase().endsWith('.pdf'));
  if (BOOK_FILTER) {
    pdfs = pdfs.filter((f) => f.toLowerCase().includes(BOOK_FILTER));
  }
  if (MAX_BOOKS !== null) {
    pdfs = pdfs.slice(0, MAX_BOOKS);
  }
  pdfs = pdfs.map((f) => path.join(TEXTBOOK_DIR, f));
  if (!pdfs.length) {
    console.error('No PDFs found in textbooks/');
    process.exit(1);
  }
  const errors = [];
  for (const pdfPath of pdfs) {
    const basename = path.basename(pdfPath);
    try {
      await Promise.race([
        processPdf(pdfPath),
        new Promise((_, reject) => setTimeout(() => reject(new Error(`timeout: ${basename}`)), BOOK_TIMEOUT_MS))
      ]);
    } catch (err) {
      await appendLog(`Skipping ${basename}: ${err.message}`);
      errors.push({ book: basename, reason: err.message });
    }
  }
  if (errors.length) {
    await appendLog(`Completed with ${errors.length} skipped/failed books:`);
    for (const e of errors) {
      await appendLog(` - ${e.book}: ${e.reason}`);
    }
  } else {
    await appendLog('Completed all books without errors.');
  }
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
