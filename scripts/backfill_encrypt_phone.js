#!/usr/bin/env node
/*
 * Backfill: Encrypt plaintext phone numbers and compute phone_hash
 * - Uses AES-256-GCM encryptPII() for users.phone
 * - Computes SHA-256 of normalized phone (+91XXXXXXXXXX) into users.phone_hash
 * - Processes in batches with transactions, supports dry-run
 *
 * Usage:
 *   node scripts/backfill_encrypt_phone.js --batch=500 --dry-run
 *   node scripts/backfill_encrypt_phone.js --batch=1000
 */

const { Pool } = require('pg');
const crypto = require('crypto');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { encryptPII } = require('../src/utils/pii');
const logger = require('../src/utils/logger');

const argv = yargs(hideBin(process.argv))
  .option('batch', { type: 'number', default: 500 })
  .option('dry-run', { type: 'boolean', default: false })
  .option('max', { type: 'number', description: 'Max rows to process (for testing)', default: 0 })
  .help()
  .argv;

function normalizePhone(phone) {
  if (!phone) return phone;
  const digits = String(phone).replace(/\D/g, '');
  if (digits.startsWith('91') && digits.length === 12) return '+' + digits;
  if (digits.length === 10) return '+91' + digits;
  return phone;
}

function computePhoneHash(phone) {
  const normalized = normalizePhone(phone);
  return crypto.createHash('sha256').update(normalized).digest('hex');
}

async function columnExists(pool, table, column) {
  const q = `SELECT 1 FROM information_schema.columns WHERE table_name=$1 AND column_name=$2`;
  const r = await pool.query(q, [table, column]);
  return r.rows.length > 0;
}

async function main() {
  const dryRun = !!argv['dry-run'];
  const batchSize = Math.max(1, argv.batch || 500);
  const maxRows = Math.max(0, argv.max || 0);

  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  });

  try {
    // Validate prerequisites
    const hasPhone = await columnExists(pool, 'users', 'phone');
    const hasPhoneHash = await columnExists(pool, 'users', 'phone_hash');
    if (!hasPhone) throw new Error("users.phone column not found");
    if (!hasPhoneHash) throw new Error("users.phone_hash column not found");

    logger.info(`Starting phone backfill (batch=${batchSize}, dryRun=${dryRun}, max=${maxRows || '∞'})`);

    let totalScanned = 0;
    let totalUpdated = 0;
    let done = false;

    while (!done) {
      // Fetch a batch of rows needing backfill
      const selectQuery = `
        SELECT id, phone
        FROM users
        WHERE (phone IS NOT NULL AND phone NOT LIKE 'enc:%')
           OR phone_hash IS NULL
        ORDER BY id
        LIMIT $1
      `;
      const { rows } = await pool.query(selectQuery, [batchSize]);
      if (rows.length === 0) break;

      totalScanned += rows.length;

      // Build updates
      const updates = [];
      for (const row of rows) {
        const { id, phone } = row;
        if (!phone) {
          // No phone but missing hash -> nothing to encrypt; skip (hash remains NULL)
          continue;
        }
        const normalized = normalizePhone(phone);
        const hash = computePhoneHash(normalized);
        const encrypted = phone.startsWith('enc:') ? phone : encryptPII(normalized);
        updates.push({ id, encrypted, hash });
      }

      if (updates.length === 0) {
        done = rows.length < batchSize;
        continue;
      }

      if (dryRun) {
        totalUpdated += updates.length;
        logger.info(`[dry-run] Would update ${updates.length} users (example):`, updates.slice(0, 3));
        done = rows.length < batchSize || (maxRows && (totalScanned >= maxRows));
        continue;
      }

      const client = await pool.connect();
      try {
        await client.query('BEGIN');

        for (const u of updates) {
          await client.query(
            `UPDATE users SET phone = $1, phone_hash = $2, updated_at = NOW() WHERE id = $3`,
            [u.encrypted, u.hash, u.id]
          );
        }

        await client.query('COMMIT');
        totalUpdated += updates.length;
        logger.info(`Updated ${updates.length} users in this batch`);
      } catch (e) {
        await client.query('ROLLBACK');
        logger.error('Batch failed, rolled back', { error: e.message });
        throw e;
      } finally {
        client.release();
      }

      if (maxRows && totalScanned >= maxRows) break;
      if (rows.length < batchSize) done = true;
    }

    logger.info(`Backfill complete. Scanned=${totalScanned}, Updated=${totalUpdated}, DryRun=${dryRun}`);
  } catch (err) {
    logger.error('Backfill aborted due to error', { error: err.message });
    process.exitCode = 1;
  } finally {
    await pool.end().catch(() => {});
  }
}

if (require.main === module) {
  main();
}
