/* Orbit Service Worker
 * ---------------------
 * Full offline-capable PWA. Four layered strategies:
 *
 *   1. App shell (cache-first, precached)
 *      index.html, manifest, icons, fonts, Leaflet CSS/JS.
 *
 *   2. GET API data (stale-while-revalidate)
 *      /quiz, /polygons, /pois, /suburb/*  — read heavy, safe to serve
 *      a slightly stale copy while revalidating in the background.
 *
 *   3. Map tiles (network-first with LRU cache, bounded)
 *      OSM CartoDB light/dark tiles. Offline = last-seen tiles.
 *
 *   4. POST /match, POST /profile (network-only with sentinel fallback)
 *      Mutating requests — don't cache. When offline, return a JSON
 *      payload the frontend recognizes as "offline: try again later".
 *
 * Versioning: bump SW_VERSION to invalidate old caches. All old cache
 * keys for THIS origin are deleted on activate, so no clutter across
 * deploys.
 */

const SW_VERSION = 'orbit-v1';
const SHELL_CACHE  = `${SW_VERSION}-shell`;
const API_CACHE    = `${SW_VERSION}-api`;
const TILES_CACHE  = `${SW_VERSION}-tiles`;

// ---------------------------------------------------------------------------
// Assets that must be in cache for the app to boot offline.
// ---------------------------------------------------------------------------
const SHELL_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/icon-maskable-512.png',
  '/icons/apple-touch-icon.png',
  '/icons/favicon-32.png',
  // Third-party — Leaflet CSS/JS + fonts. Cached opaque responses are fine
  // for these: they're public CDNs and the frontend doesn't introspect them.
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap',
];

// Tiles cache max entries — bounded to keep storage modest on phones.
const MAX_TILES = 400;

// ---------------------------------------------------------------------------
// Install: precache the shell. If any asset fails, we log and continue —
// better a degraded-but-alive SW than no SW at all.
// ---------------------------------------------------------------------------
self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    await Promise.all(SHELL_ASSETS.map(async (url) => {
      try {
        // `reload` bypasses HTTP cache for the initial fetch so we don't
        // precache a stale copy the browser already had.
        const req = new Request(url, { cache: 'reload' });
        const res = await fetch(req);
        if (res.ok || res.type === 'opaque') {
          await cache.put(url, res);
        }
      } catch (err) {
        console.warn('[sw] precache miss', url, err);
      }
    }));
    // Activate new SW without waiting for old tabs to close — safe here
    // because we have version-scoped cache names.
    self.skipWaiting();
  })());
});

// ---------------------------------------------------------------------------
// Activate: prune old caches, claim clients so the SW starts serving
// immediately without a reload.
// ---------------------------------------------------------------------------
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys
      .filter((k) => !k.startsWith(SW_VERSION))
      .map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Only handle GET with http(s). POST/PUT bypass the cache entirely,
  // *except* for /match and /profile which get a friendly offline fallback.
  if (req.method !== 'GET') {
    if (isMutatingApi(req)) {
      event.respondWith(postApiOrOffline(req));
    }
    return; // let POSTs pass through normally when online
  }

  if (!req.url.startsWith('http')) return;

  const url = new URL(req.url);

  // 1. Navigation requests → serve cached index.html offline.
  if (req.mode === 'navigate') {
    event.respondWith(navigationHandler(req));
    return;
  }

  // 2. Map tile providers — network-first with LRU cache.
  if (isMapTile(url)) {
    event.respondWith(tilesHandler(req));
    return;
  }

  // 3. Same-origin API GETs — stale-while-revalidate.
  if (url.origin === self.location.origin && isApiGet(url.pathname)) {
    event.respondWith(staleWhileRevalidate(req, API_CACHE));
    return;
  }

  // 4. Everything else (shell assets, fonts, Leaflet) — cache-first with
  //    network fallback + opportunistic refill.
  event.respondWith(cacheFirst(req, SHELL_CACHE));
});

// ---------------------------------------------------------------------------
// Strategy helpers
// ---------------------------------------------------------------------------

async function navigationHandler(req) {
  try {
    const net = await fetch(req);
    // Cache the latest index.html copy so offline loads stay fresh.
    const cache = await caches.open(SHELL_CACHE);
    cache.put('/', net.clone());
    return net;
  } catch {
    const cache = await caches.open(SHELL_CACHE);
    const cached = await cache.match('/') || await cache.match('/index.html');
    if (cached) return cached;
    return offlineHtml();
  }
}

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(req);
  if (cached) {
    // Background refresh — silent failure is fine.
    fetch(req).then((res) => {
      if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone());
    }).catch(() => {});
    return cached;
  }
  try {
    const net = await fetch(req);
    if (net && (net.ok || net.type === 'opaque')) cache.put(req, net.clone());
    return net;
  } catch {
    // Last-ditch: any cache has this exact URL?
    const any = await caches.match(req);
    if (any) return any;
    return new Response('', { status: 504, statusText: 'Offline' });
  }
}

async function staleWhileRevalidate(req, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(req);
  const networkPromise = fetch(req).then((res) => {
    if (res && res.ok) cache.put(req, res.clone());
    return res;
  }).catch(() => null);
  // Serve cache immediately if we have it; otherwise await network.
  if (cached) {
    // Fire-and-forget revalidation; swallow errors so rejected promises
    // don't bubble up and spam the console.
    networkPromise.catch(() => {});
    return cached;
  }
  const net = await networkPromise;
  if (net) return net;
  return new Response(JSON.stringify({ offline: true, error: 'offline' }), {
    status: 503,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function tilesHandler(req) {
  const cache = await caches.open(TILES_CACHE);
  try {
    const net = await fetch(req);
    if (net && (net.ok || net.type === 'opaque')) {
      cache.put(req, net.clone());
      trimCache(TILES_CACHE, MAX_TILES);
    }
    return net;
  } catch {
    const cached = await cache.match(req);
    if (cached) return cached;
    // Return a 1x1 transparent PNG so Leaflet doesn't show broken-image tiles.
    return transparentTile();
  }
}

async function postApiOrOffline(req) {
  try {
    return await fetch(req);
  } catch {
    return new Response(JSON.stringify({
      offline: true,
      error: 'You are offline. Results will load when you reconnect.',
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function isApiGet(pathname) {
  // List kept explicit — don't want to catch the SPA HTML route or icons.
  if (pathname === '/quiz')           return true;
  if (pathname === '/polygons')       return true;
  if (pathname === '/pois')           return true;
  if (pathname === '/health')         return true;
  if (pathname === '/api')            return true;
  if (pathname.startsWith('/suburb/')) return true;
  return false;
}

function isMutatingApi(req) {
  try {
    const u = new URL(req.url);
    if (u.origin !== self.location.origin) return false;
    return u.pathname === '/match' || u.pathname === '/profile';
  } catch {
    return false;
  }
}

function isMapTile(url) {
  const h = url.hostname;
  // OSM / CartoDB raster tile CDNs used by Leaflet in Orbit.
  if (h.endsWith('tile.openstreetmap.org')) return true;
  if (h.endsWith('basemaps.cartocdn.com'))  return true;
  if (h.includes('cartodb'))                return true;
  if (h.includes('openstreetmap'))          return true;
  return false;
}

/** Cap cache size — delete oldest entries (FIFO, which is close enough to LRU
 * for tiles where replacement order tracks visit order). */
async function trimCache(cacheName, maxEntries) {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();
  if (keys.length <= maxEntries) return;
  const overflow = keys.length - maxEntries;
  for (let i = 0; i < overflow; i++) {
    await cache.delete(keys[i]);
  }
}

/** 1x1 transparent PNG so Leaflet's tile grid stays clean when offline and
 * a tile isn't cached. Base64 decoded to a binary blob. */
function transparentTile() {
  const b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Response(bytes, { headers: { 'Content-Type': 'image/png' } });
}

function offlineHtml() {
  return new Response(
    `<!doctype html><meta charset="utf-8"><title>Orbit — offline</title>
     <style>html,body{height:100%;margin:0;font:16px/1.5 system-ui;background:#fafaf9;color:#0a0a0a;display:flex;align-items:center;justify-content:center}div{max-width:380px;text-align:center;padding:24px}</style>
     <div><h1>You're offline</h1><p>Open Orbit again when you're back on a network, or keep this tab open to auto-retry.</p></div>`,
    { headers: { 'Content-Type': 'text/html; charset=utf-8' }, status: 200 },
  );
}

// ---------------------------------------------------------------------------
// Message channel — the page can ask the SW for its version or to skip
// waiting. Keep it minimal.
// ---------------------------------------------------------------------------
self.addEventListener('message', (event) => {
  if (!event.data || typeof event.data !== 'object') return;
  if (event.data.type === 'SKIP_WAITING') self.skipWaiting();
  if (event.data.type === 'VERSION') {
    event.ports[0]?.postMessage({ version: SW_VERSION });
  }
});
