const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Build a JWK from the PUBLIC_KEY PEM and optional KID
function getJwks() {
  const dir = process.env.JWKS_DIR;
  const keys = [];

  if (dir && fs.existsSync(dir)) {
    const files = fs.readdirSync(dir).filter(f => /\.pem$/i.test(f));
    for (const file of files) {
      try {
        const pem = fs.readFileSync(path.join(dir, file), 'utf8');
        const kid = deriveKid(pem);
        const keyObject = crypto.createPublicKey(pem);
        const jwk = keyObject.export({ format: 'jwk' });
        jwk.kid = kid;
        jwk.use = 'sig';
        jwk.alg = 'RS256';
        jwk.kty = jwk.kty || 'RSA';
        keys.push(jwk);
      } catch (_) {
        // skip invalid file
      }
    }
  }

  if (keys.length === 0) {
    const PUBLIC_KEY = process.env.JWT_PUBLIC_KEY;
    if (!PUBLIC_KEY) {
      throw new Error('JWT_PUBLIC_KEY not configured');
    }
    const kid = process.env.JWT_KID || deriveKid(PUBLIC_KEY);
    const keyObject = crypto.createPublicKey(PUBLIC_KEY);
    const jwk = keyObject.export({ format: 'jwk' });
    jwk.kid = kid;
    jwk.use = 'sig';
    jwk.alg = 'RS256';
    jwk.kty = jwk.kty || 'RSA';
    keys.push(jwk);
  }

  return { keys };
}

function deriveKid(pem) {
  // Hash DER of public key to get a deterministic KID
  const keyObject = crypto.createPublicKey(pem);
  const der = keyObject.export({ type: 'spki', format: 'der' });
  const hash = crypto.createHash('sha256').update(der).digest('base64');
  // base64url
  return hash.replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
}

module.exports = { getJwks, deriveKid };
