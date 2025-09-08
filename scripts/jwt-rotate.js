#!/usr/bin/env node
/*
 * JWT Key Rotation Helper
 * - Generates a new RSA keypair (RS256)
 * - Computes kid from public key (SPKI DER sha256 base64url)
 * - Writes PEMs to output directory and prints environment export guidance
 *
 * Usage:
 *   node scripts/jwt-rotate.js ./keys
 *   node scripts/jwt-rotate.js ./keys mykeyprefix
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function deriveKid(pem) {
  const keyObject = crypto.createPublicKey(pem);
  const der = keyObject.export({ type: 'spki', format: 'der' });
  const hash = crypto.createHash('sha256').update(der).digest('base64');
  return hash.replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
}

function main() {
  const outDir = process.argv[2] || './keys';
  const prefix = process.argv[3] || 'jwt';

  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
  });

  const kid = deriveKid(publicKey);
  const pubFile = path.join(outDir, `${prefix}.${kid}.public.pem`);
  const privFile = path.join(outDir, `${prefix}.${kid}.private.pem`);

  fs.writeFileSync(pubFile, publicKey, 'utf8');
  fs.writeFileSync(privFile, privateKey, 'utf8');

  console.log('Generated new RSA keypair');
  console.log('kid:', kid);
  console.log('Public Key:', pubFile);
  console.log('Private Key:', privFile);
  console.log('');
  console.log('Next steps:');
  console.log('1) Set environment variables to use the new key:');
  console.log(`   export JWT_PRIVATE_KEY="$(cat ${privFile} | sed 's/\n/\\n/g')"`);
  console.log(`   export JWT_PUBLIC_KEY="$(cat ${pubFile} | sed 's/\n/\\n/g')"`);
  console.log(`   export JWT_KID=${kid}`);
  console.log('');
  console.log('2) For seamless rotation, publish both old and new public keys via JWKS:');
  console.log('   - Place all active public keys in a directory and set JWKS_DIR to that directory');
  console.log('   - Or keep JWT_PUBLIC_KEY set to the current key while JWKS_DIR contains both old and new');
  console.log('');
  console.log('3) Roll restart the service after updating env vars. Keep the old public key in JWKS until old tokens expire.');
}

main();
