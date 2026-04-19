// Crawl LibreTexts Bookshelves, extract print IDs, and download full PDFs.
// printId format: "<subdomain>-<data-page-id>"
// PDF URL: https://batch.libretexts.org/print/Letter/Finished/<printId>/Full.pdf

import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const textbooksDir = path.join(process.cwd(), 'textbooks');
await fs.mkdir(textbooksDir, { recursive: true });
const manifestPath = path.join(textbooksDir, 'manifest.jsonl');

const MAX_DOWNLOADS = process.env.MAX_DOWNLOADS ? Number(process.env.MAX_DOWNLOADS) : 200;
const MAX_PAGES = process.env.MAX_PAGES ? Number(process.env.MAX_PAGES) : 500;
const DELAY_MS = process.env.DELAY_MS ? Number(process.env.DELAY_MS) : 500;
const TARGET_PER_DOMAIN = process.env.TARGET_PER_DOMAIN ? Number(process.env.TARGET_PER_DOMAIN) : 20;
const SKIP_PRINT_IDS = process.env.SKIP_PRINT_IDS ? new Set(process.env.SKIP_PRINT_IDS.split(',').map((s) => s.trim())) : new Set();
const SKIP_TITLES = process.env.SKIP_TITLES ? new Set(process.env.SKIP_TITLES.split(',').map((s) => s.trim().toLowerCase())) : new Set();

// Known Bookshelves roots to seed the crawl.
// Broader set of subject libraries (from libretexts.org Libraries menu).
// Known working subject libraries (HEAD-tested):
// bio, biz, chem, eng, geo, human, k12, math, med, phys, socialsci, stats, workforce, espanol
const ROOTS = [
  'https://bio.libretexts.org/Bookshelves',
  'https://biz.libretexts.org/Bookshelves',
  'https://chem.libretexts.org/Bookshelves',
  'https://eng.libretexts.org/Bookshelves',
  'https://geo.libretexts.org/Bookshelves',
  'https://human.libretexts.org/Bookshelves',
  'https://k12.libretexts.org/Bookshelves',
  'https://math.libretexts.org/Bookshelves',
  'https://med.libretexts.org/Bookshelves',
  'https://phys.libretexts.org/Bookshelves',
  'https://socialsci.libretexts.org/Bookshelves',
  'https://stats.libretexts.org/Bookshelves',
  'https://workforce.libretexts.org/Bookshelves',
  'https://espanol.libretexts.org/Bookshelves'
];

const sanitizeName = (raw) => {
  const cleaned = raw.replace(/[^a-z0-9]+/gi, ' ').trim();
  if (!cleaned) return 'LibreText';
  return cleaned
    .split(' ')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join('');
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const extractFromListing = async (page, listingUrl) => {
  const subdomain = new URL(listingUrl).hostname.split('.')[0];
  return page.$$eval(
    'li.mt-sortable-listing',
    (lis, sub) => {
      const entries = [];
      const nextLinks = [];
      for (const li of lis) {
        const a = li.querySelector('a[href]');
        const href = a ? a.href : '';
        const title =
          (li.querySelector('.mt-sortable-listing-title')?.textContent || a?.textContent || '').trim();
        const id = li.getAttribute('data-page-id');
        if (id) entries.push({ printId: `${sub}-${id}`, href, title });
        if (href) nextLinks.push(href);
      }
      return { entries, nextLinks };
    },
    subdomain
  );
};

const filterNextLinks = (currentUrl, links) => {
  const current = new URL(currentUrl);
  return Array.from(new Set(links))
    .map((href) => {
      try {
        const u = new URL(href, current.origin);
        if (u.hostname !== current.hostname) return null;
        if (!u.pathname.startsWith('/Bookshelves')) return null;
        return u.toString();
      } catch {
        return null;
      }
    })
    .filter(Boolean);
};

const downloadById = async (printId, titleHint, sourceUrl = '') => {
  if (SKIP_PRINT_IDS.has(printId)) {
    console.log(`Skipping ${printId} (already in DB)`);
    return false;
  }
  const printUrl = `https://batch.libretexts.org/print/Letter/Finished/${printId}/Full.pdf`;
  const targetName = `${sanitizeName(titleHint || printId)}-full.pdf`;
  if (SKIP_TITLES.has((titleHint || '').toLowerCase())) {
    console.log(`Skipping ${titleHint} (already in DB)`);
    return false;
  }
  const filePath = path.join(textbooksDir, targetName);

  try {
    const existing = await fs.stat(filePath);
    if (existing.size > 0) {
      console.log(`Skipping existing ${filePath}`);
      return false;
    }
  } catch {
    // not present, continue
  }

  console.log(`Downloading ${printUrl} -> ${filePath}`);

  const res = await fetch(printUrl, {
    headers: {
      'User-Agent': 'StateOfLociFetcher/1.0'
    }
  });
  if (!res.ok) {
    console.warn(`Failed to fetch PDF ${printUrl} status ${res.status}`);
    return false;
  }
  const buf = Buffer.from(await res.arrayBuffer());
  await fs.writeFile(filePath, buf);
  const stats = await fs.stat(filePath);
  console.log(`Saved ${filePath} (${(stats.size / (1024 * 1024)).toFixed(2)} MB)`);
  const manifestEntry = {
    printId,
    title: titleHint || printId,
    filePath,
    sourceUrl,
    downloadUrl: printUrl,
    sizeBytes: stats.size,
    timestamp: new Date().toISOString()
  };
  await fs.appendFile(manifestPath, JSON.stringify(manifestEntry) + '\n');
  return true;
};

const crawlAndDownload = async () => {
  const browser = await chromium.launch({ headless: true });
  const queue = [...ROOTS];
  const visited = new Set();
  const seenPrintIds = new Set();
  let downloaded = 0;
  const perDomainCount = {};

  while (queue.length && visited.size < MAX_PAGES && downloaded < MAX_DOWNLOADS) {
    const url = queue.shift();
    if (visited.has(url)) continue;
    visited.add(url);

    const page = await browser.newPage();
    try {
      console.log(`Visiting ${url}`);
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
      // Try to give dynamic listings a moment to render.
      await page.waitForTimeout(500);

      const { entries, nextLinks } = await extractFromListing(page, url);
      console.log(`Found ${entries.length} entries on ${url}`);

      const isRootBookshelf =
        new URL(url).pathname.replace(/\/+$/, '') === '/Bookshelves';
      if (!isRootBookshelf) {
        const domain = new URL(url).hostname.split('.')[0];
        if (!perDomainCount[domain]) perDomainCount[domain] = 0;
        for (const entry of entries) {
          if (downloaded >= MAX_DOWNLOADS) break;
          if (seenPrintIds.has(entry.printId)) continue;
          if (perDomainCount[domain] >= TARGET_PER_DOMAIN) break;
          seenPrintIds.add(entry.printId);
          const ok = await downloadById(entry.printId, entry.title || entry.href, entry.href || '');
          if (ok) {
            downloaded += 1;
            perDomainCount[domain] += 1;
            if (perDomainCount[domain] >= TARGET_PER_DOMAIN) break;
          }
          if (DELAY_MS > 0) await sleep(DELAY_MS);
        }
      }

      const toQueue = filterNextLinks(url, nextLinks);
      for (const link of toQueue) {
        if (!visited.has(link)) queue.push(link);
      }
    } catch (err) {
      console.warn(`Error processing ${url}: ${err.message}`);
    } finally {
      await page.close();
    }
  }

  await browser.close();
  console.log(`Done. Downloaded ${downloaded} PDFs. Visited ${visited.size} pages.`);
};

crawlAndDownload().catch((err) => {
  console.error(err);
  process.exit(1);
});
