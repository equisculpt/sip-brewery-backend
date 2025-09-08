/*
 * JTI Replay Guard Middleware
 * - Blocks replay of access token jti for a short TTL
 * - Uses Redis when available (REDIS_URL), falls back to in-memory TTL map
 */

const crypto = require('crypto');
const logger = require('../utils/logger');

let redisClient = null;
let redisReady = false;

(function initRedis() {
  const url = process.env.REDIS_URL || process.env.REDIS_CONNECTION_URL;
  if (!url) return;
  try {
    // Try ioredis first
    // eslint-disable-next-line global-require
    const IORedis = require('ioredis');
    redisClient = new IORedis(url, {
      maxRetriesPerRequest: 1,
      enableOfflineQueue: false
    });
    redisClient.on('ready', () => { redisReady = true; });
    redisClient.on('error', () => { redisReady = false; });
  } catch (e) {
    try {
      // Try node-redis v4
      // eslint-disable-next-line global-require
      const { createClient } = require('redis');
      redisClient = createClient({ url });
      redisClient.on('ready', () => { redisReady = true; });
      redisClient.on('error', () => { redisReady = false; });
      redisClient.connect().catch(() => {});
    } catch (e2) {
      redisClient = null;
      redisReady = false;
    }
  }
}());

// In-memory fallback with TTL
const memStore = new Map(); // key -> expiresAt(ms)
function memSetIfAbsent(key, ttlSec) {
  const now = Date.now();
  const existing = memStore.get(key);
  if (existing && existing > now) return false;
  const exp = now + ttlSec * 1000;
  memStore.set(key, exp);
  return true;
}

function sweepExpired() {
  const now = Date.now();
  for (const [k, exp] of memStore.entries()) {
    if (exp <= now) memStore.delete(k);
  }
}
setInterval(sweepExpired, 60 * 1000).unref?.();

function jtiReplayGuard(options = {}) {
  const ttlSeconds = Number(options.ttlSeconds || 120); // default 2 min
  const prefix = String(options.prefix || 'jti:access:');

  return async function jtiGuard(req, res, next) {
    try {
      const claims = req.tokenClaims || {};
      const jti = claims.jti;
      const exp = claims.exp; // seconds since epoch
      if (!jti) return res.status(401).json({ success: false, message: 'Invalid token (missing jti)' });

      // Compute effective TTL so we don't hold beyond token expiry
      const nowSec = Math.floor(Date.now() / 1000);
      const remaining = exp ? Math.max(0, exp - nowSec) : ttlSeconds;
      const ttl = Math.max(1, Math.min(ttlSeconds, remaining));

      const key = prefix + jti;

      if (redisClient && redisReady) {
        try {
          // SET key 1 NX EX <ttl>
          const resSet = await redisClient.set(key, '1', 'EX', ttl, 'NX');
          if (resSet !== 'OK') {
            try { logger.warn('Replay detected (redis)', { jti, ip: req.ip, path: req.originalUrl }); } catch (_) {}
            return res.status(409).json({ success: false, message: 'Replay detected' });
          }
          return next();
        } catch (e) {
          // fall through to memory on any redis error
          try { logger.warn('Replay guard redis error, falling back to memory', { error: e?.message }); } catch (_) {}
        }
      }

      // Fallback in-memory
      const ok = memSetIfAbsent(key, ttl);
      if (!ok) {
        try { logger.warn('Replay detected (memory)', { jti, ip: req.ip, path: req.originalUrl }); } catch (_) {}
        return res.status(409).json({ success: false, message: 'Replay detected' });
      }
      return next();
    } catch (err) {
      try { logger.error('Replay guard failure', { error: err?.message }); } catch (_) {}
      return res.status(500).json({ success: false, message: 'Replay guard failure' });
    }
  };
}

module.exports = { jtiReplayGuard };
