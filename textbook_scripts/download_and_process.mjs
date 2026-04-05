/**
 * OpenStax textbook downloader.
 *
 * Downloads freely available CC-licensed textbooks from openstax.org using
 * their public book catalog API, then hands each PDF off to the segmentation
 * pipeline.
 *
 * Usage (test mode — no downloads):
 *   node textbook_scripts/download_and_process.mjs
 *
 * Usage (full):
 *   TEXTBOOK_TEST_MODE=false node textbook_scripts/download_and_process.mjs
 *
 * Environment variables:
 *   TEXTBOOK_TEST_MODE   "false" to actually download/process (default: true)
 *   TEXTBOOK_OUTPUT_DIR  where to save PDFs (default: ./data/textbooks/raw)
 *   TEXTBOOK_MAX_BOOKS   limit number of books (default: all)
 *   TEXTBOOK_SUBJECTS    comma-separated subject filter (default: all)
 *                        e.g. "science,math"
 *   TEXTBOOK_USER_AGENT  custom User-Agent header
 */

import fs from 'fs';
import https from 'https';
import http from 'http';
import path from 'path';
import { execSync } from 'child_process';
import { URL } from 'url';

// ─── Configuration ────────────────────────────────────────────────────────────

const TEST_MODE = process.env.TEXTBOOK_TEST_MODE !== 'false';
const OUTPUT_DIR = process.env.TEXTBOOK_OUTPUT_DIR || path.join('data', 'textbooks', 'raw');
const MAX_BOOKS = process.env.TEXTBOOK_MAX_BOOKS ? Number(process.env.TEXTBOOK_MAX_BOOKS) : null;
const SUBJECT_FILTER = process.env.TEXTBOOK_SUBJECTS
  ? process.env.TEXTBOOK_SUBJECTS.toLowerCase().split(',').map(s => s.trim()).filter(Boolean)
  : [];
const USER_AGENT = process.env.TEXTBOOK_USER_AGENT || 'W1z4rDV1510n-TextbookDownloader/1.0 (research; educational use; https://openstax.org/license)';

// OpenStax public REST API — returns the book catalog as JSON.
// Endpoint documented at: https://openstax.org/api/v2/books/
const OPENSTAX_API = 'https://openstax.org/api/v2/books/?format=json&limit=200';

// Fallback curated list: known stable slugs that map directly to CDN PDFs.
// Format: { slug, title, subject }
// CDN URL: https://assets.openstax.org/oscms-prodcms/media/documents/<slug>.pdf
const FALLBACK_CATALOG = [
  // Sciences
  { slug: 'biology-2e',                  title: 'Biology 2e',                       subject: 'science' },
  { slug: 'microbiology',                title: 'Microbiology',                     subject: 'science' },
  { slug: 'anatomy-and-physiology-2e',   title: 'Anatomy and Physiology 2e',        subject: 'science' },
  { slug: 'chemistry-2e',                title: 'Chemistry 2e',                     subject: 'science' },
  { slug: 'chemistry-atoms-first-2e',    title: 'Chemistry: Atoms First 2e',        subject: 'science' },
  { slug: 'college-physics-2e',          title: 'College Physics 2e',               subject: 'science' },
  { slug: 'university-physics-volume-1', title: 'University Physics Vol. 1',        subject: 'science' },
  { slug: 'university-physics-volume-2', title: 'University Physics Vol. 2',        subject: 'science' },
  { slug: 'university-physics-volume-3', title: 'University Physics Vol. 3',        subject: 'science' },
  { slug: 'astronomy-2e',                title: 'Astronomy 2e',                     subject: 'science' },
  // Mathematics
  { slug: 'calculus-volume-1',           title: 'Calculus Vol. 1',                  subject: 'math'    },
  { slug: 'calculus-volume-2',           title: 'Calculus Vol. 2',                  subject: 'math'    },
  { slug: 'calculus-volume-3',           title: 'Calculus Vol. 3',                  subject: 'math'    },
  { slug: 'precalculus-2e',              title: 'Precalculus 2e',                   subject: 'math'    },
  { slug: 'algebra-and-trigonometry-2e', title: 'Algebra and Trigonometry 2e',      subject: 'math'    },
  { slug: 'statistics',                  title: 'Statistics',                       subject: 'math'    },
  { slug: 'introductory-statistics-2e',  title: 'Introductory Statistics 2e',       subject: 'math'    },
  // Social Sciences / Business
  { slug: 'principles-of-economics-3e',  title: 'Principles of Economics 3e',       subject: 'social'  },
  { slug: 'principles-of-microeconomics-3e', title: 'Principles of Microeconomics 3e', subject: 'social' },
  { slug: 'introduction-to-sociology-3e', title: 'Introduction to Sociology 3e',   subject: 'social'  },
  { slug: 'psychology-2e',               title: 'Psychology 2e',                    subject: 'social'  },
  { slug: 'us-history',                  title: 'U.S. History',                     subject: 'social'  },
  // Computer Science
  { slug: 'introduction-to-python-programming', title: 'Introduction to Python Programming', subject: 'cs' },
];

const CDN_BASE = 'https://assets.openstax.org/oscms-prodcms/media/documents';

// ─── Utilities ─────────────────────────────────────────────────────────────────

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

/**
 * HTTP/HTTPS GET with automatic redirect following (up to maxRedirects).
 * Returns a Promise<IncomingMessage> pointing at the final response.
 */
function fetchFollowRedirects(urlStr, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    function doRequest(currentUrl, redirectsLeft) {
      const parsed = new URL(currentUrl);
      const client = parsed.protocol === 'https:' ? https : http;
      const options = {
        hostname: parsed.hostname,
        port: parsed.port || (parsed.protocol === 'https:' ? 443 : 80),
        path: parsed.pathname + parsed.search,
        method: 'GET',
        headers: {
          'User-Agent': USER_AGENT,
          'Accept': 'application/pdf,application/octet-stream,*/*',
        },
        timeout: 60_000,
      };
      const req = client.request(options, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          if (redirectsLeft <= 0) {
            reject(new Error(`Too many redirects for ${currentUrl}`));
            return;
          }
          const nextUrl = new URL(res.headers.location, currentUrl).toString();
          res.resume(); // drain body
          doRequest(nextUrl, redirectsLeft - 1);
        } else {
          resolve(res);
        }
      });
      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error(`Request timed out: ${currentUrl}`));
      });
      req.end();
    }
    doRequest(urlStr, maxRedirects);
  });
}

/**
 * Fetch JSON from a URL.
 * @param {string} urlStr
 * @returns {Promise<any>}
 */
async function fetchJson(urlStr) {
  const res = await fetchFollowRedirects(urlStr);
  if (res.statusCode !== 200) {
    throw new Error(`HTTP ${res.statusCode} fetching ${urlStr}`);
  }
  return new Promise((resolve, reject) => {
    let body = '';
    res.setEncoding('utf-8');
    res.on('data', (chunk) => { body += chunk; });
    res.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch (e) { reject(new Error(`JSON parse error: ${e.message}`)); }
    });
    res.on('error', reject);
  });
}

/**
 * Download a file to disk with progress logging, following redirects.
 * Skips if the file already exists.
 * @param {string} urlStr
 * @param {string} destPath
 * @returns {Promise<void>}
 */
async function downloadFile(urlStr, destPath) {
  if (fs.existsSync(destPath)) {
    const { size } = fs.statSync(destPath);
    if (size > 4096) {
      log(`  Skipping (already exists, ${(size / 1024 / 1024).toFixed(1)} MB): ${path.basename(destPath)}`);
      return;
    }
    // File too small — likely a failed previous download; retry.
    fs.unlinkSync(destPath);
  }

  const res = await fetchFollowRedirects(urlStr);
  if (res.statusCode !== 200) {
    throw new Error(`HTTP ${res.statusCode} downloading ${urlStr}`);
  }

  const tmpPath = destPath + '.download';
  await new Promise((resolve, reject) => {
    const out = fs.createWriteStream(tmpPath);
    let bytesWritten = 0;
    res.on('data', (chunk) => { bytesWritten += chunk.length; });
    res.pipe(out);
    out.on('finish', () => {
      out.close(() => resolve());
    });
    out.on('error', (err) => {
      fs.unlink(tmpPath, () => {});
      reject(err);
    });
    res.on('error', (err) => {
      fs.unlink(tmpPath, () => {});
      reject(err);
    });
    // Log final size after piping completes.
    out.on('close', () => {
      log(`  Downloaded ${path.basename(destPath)} (${(bytesWritten / 1024 / 1024).toFixed(1)} MB)`);
    });
  });

  // Atomic rename: only replace destination once the download is complete.
  fs.renameSync(tmpPath, destPath);
}

// ─── Catalog fetching ─────────────────────────────────────────────────────────

/**
 * Try to fetch the live book catalog from openstax.org.
 * Falls back to FALLBACK_CATALOG on any error.
 *
 * @returns {Promise<Array<{slug: string, title: string, subject: string, pdfUrl?: string}>>}
 */
async function fetchCatalog() {
  try {
    log('Fetching OpenStax book catalog from API…');
    const data = await fetchJson(OPENSTAX_API);

    // The Wagtail API returns { items: [...] } or similar.  Be defensive.
    const items = Array.isArray(data) ? data
      : Array.isArray(data?.items) ? data.items
      : Array.isArray(data?.results) ? data.results
      : [];

    if (items.length === 0) {
      throw new Error('Empty catalog response');
    }

    const books = items.map((book) => ({
      slug: book.slug || book.book_slug || '',
      title: book.title || book.book_title || book.slug || '',
      subject: (book.subject_name || book.subjects?.[0]?.name || '').toLowerCase(),
      // Some catalog entries include a direct PDF URL field.
      pdfUrl: book.high_resolution_pdf_url
        || book.pdf_url
        || book.download_url
        || null,
    })).filter(b => b.slug);

    log(`Catalog loaded: ${books.length} books from API`);
    return books;
  } catch (err) {
    log(`API catalog unavailable (${err.message}), using curated fallback list`);
    return FALLBACK_CATALOG;
  }
}

/**
 * Construct the CDN PDF URL for a book slug.
 * Tries the primary CDN pattern; the download function will follow any redirects.
 */
function pdfUrlForSlug(slug, overridePdfUrl) {
  if (overridePdfUrl) return overridePdfUrl;
  return `${CDN_BASE}/${slug}.pdf`;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  log('=== OpenStax Textbook Downloader ===');
  log(`Test mode: ${TEST_MODE}`);
  log(`Output dir: ${OUTPUT_DIR}`);

  if (TEST_MODE) {
    log('[TEST MODE] Fetching catalog without downloading PDFs…');
    const catalog = await fetchCatalog();
    let visible = catalog;
    if (SUBJECT_FILTER.length > 0) {
      visible = visible.filter(b => SUBJECT_FILTER.some(s => b.subject.includes(s)));
    }
    if (MAX_BOOKS !== null) visible = visible.slice(0, MAX_BOOKS);
    log(`[TEST MODE] Would download ${visible.length} book(s):`);
    visible.forEach((b, i) => {
      const url = pdfUrlForSlug(b.slug, b.pdfUrl);
      console.log(`  ${i + 1}. ${b.title} (${b.slug})`);
      console.log(`     → ${url}`);
    });
    log('[TEST MODE] Set TEXTBOOK_TEST_MODE=false to perform real downloads.');
    return;
  }

  // Full mode — actually download.
  ensureDir(OUTPUT_DIR);

  let catalog = await fetchCatalog();

  if (SUBJECT_FILTER.length > 0) {
    const before = catalog.length;
    catalog = catalog.filter(b => SUBJECT_FILTER.some(s => b.subject.includes(s)));
    log(`Subject filter "${SUBJECT_FILTER.join(',')}" reduced catalog from ${before} → ${catalog.length} books`);
  }

  if (MAX_BOOKS !== null) {
    catalog = catalog.slice(0, MAX_BOOKS);
    log(`Limited to first ${MAX_BOOKS} book(s)`);
  }

  log(`Downloading ${catalog.length} book(s)…`);

  const errors = [];
  for (const book of catalog) {
    const pdfUrl = pdfUrlForSlug(book.slug, book.pdfUrl);
    const filename = `${book.slug}.pdf`;
    const destPath = path.join(OUTPUT_DIR, filename);
    try {
      log(`Downloading: ${book.title}`);
      log(`  URL: ${pdfUrl}`);
      await downloadFile(pdfUrl, destPath);
    } catch (err) {
      log(`  ERROR downloading ${book.slug}: ${err.message}`);
      errors.push({ slug: book.slug, error: err.message });
    }
  }

  // Wire into segment-textbook.mjs if Node.js segmentation is requested.
  const segmentScript = path.join('textbook_scripts', 'segment-textbook.mjs');
  if (fs.existsSync(segmentScript)) {
    log('Running textbook segmentation pipeline…');
    try {
      execSync(`node ${segmentScript}`, {
        stdio: 'inherit',
        env: {
          ...process.env,
          TEXTBOOK_DIR: OUTPUT_DIR,
        },
      });
    } catch (err) {
      log(`Segmentation pipeline error: ${err.message}`);
    }
  }

  if (errors.length > 0) {
    log(`\nCompleted with ${errors.length} error(s):`);
    errors.forEach(e => log(`  - ${e.slug}: ${e.error}`));
  } else {
    log('\nAll downloads completed successfully.');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
