#!/usr/bin/env node
/*
 * Create index on users.phone_hash if not present (concurrently)
 * Note: CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
 */
const { Pool } = require('pg');

async function main() {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  });
  try {
    console.log('Ensuring index idx_users_phone_hash on users(phone_hash)...');
    const checkSql = `
      SELECT 1 FROM pg_indexes 
      WHERE schemaname = 'public' AND indexname = 'idx_users_phone_hash'
    `;
    const r = await pool.query(checkSql);
    if (r.rows.length > 0) {
      console.log('Index already exists.');
      return;
    }
    console.log('Index missing. Creating with CONCURRENTLY...');
    await pool.query('CREATE INDEX CONCURRENTLY idx_users_phone_hash ON users (phone_hash)');
    console.log('Index created.');
  } catch (e) {
    console.error('Failed to create index:', e && (e.stack || e.message || e));
    process.exitCode = 1;
  } finally {
    await pool.end().catch(() => {});
  }
}

if (require.main === module) main();
