const crypto = require('crypto');

// AES-256-GCM field encryption helpers
// Requires DATA_ENC_KEY as 32-byte base64 or hex key
const RAW_KEY = process.env.DATA_ENC_KEY || '';

function getKey() {
  if (!RAW_KEY) throw new Error('DATA_ENC_KEY not configured');
  let key;
  try {
    key = Buffer.from(RAW_KEY, 'base64');
  } catch {
    key = Buffer.from(RAW_KEY, 'hex');
  }
  if (key.length !== 32) {
    throw new Error('DATA_ENC_KEY must be 32 bytes (AES-256)');
  }
  return key;
}

function encryptPII(plaintext) {
  if (plaintext == null) return null;
  const key = getKey();
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const ciphertext = Buffer.concat([cipher.update(String(plaintext), 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  const payload = Buffer.concat([iv, tag, ciphertext]).toString('base64');
  return `enc:${payload}`;
}

function decryptPII(value) {
  if (!value || typeof value !== 'string' || !value.startsWith('enc:')) return value;
  const key = getKey();
  const buf = Buffer.from(value.slice(4), 'base64');
  const iv = buf.subarray(0, 12);
  const tag = buf.subarray(12, 28);
  const ciphertext = buf.subarray(28);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(tag);
  const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString('utf8');
  return plaintext;
}

module.exports = { encryptPII, decryptPII };
