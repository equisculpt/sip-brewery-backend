# Frontend Integration Guide

## 1. API Base URL
- **All frontend API calls must use:**
  
  `https://api.sipbrewery.com/api`

---

## 2. CORS Policy
- Only requests from `https://sipbrewery.com` are allowed.
- Allowed HTTP methods: `GET`, `POST`, `PUT`, `DELETE`
- Credentials (cookies, auth headers) are supported.

---

## 3. Authentication
- **JWT (JSON Web Token)** is required for all secure endpoints.
- Secure API routes are prefixed with `/secure` (e.g., `https://api.sipbrewery.com/api/secure/...`).
- Include the JWT token in the `Authorization` header for all secure requests:
  ```
  Authorization: <token>
  ```
- If the token is missing or invalid, the backend will respond with `403 Forbidden` or `401 Unauthorized`.

---

## 4. Rate Limiting
- All `/api` routes are rate-limited to **100 requests per 15 minutes per IP**.
- Exceeding this limit will result in a `429 Too Many Requests` error.

---

## 5. Health Check
- Endpoint: `GET https://api.sipbrewery.com/api/health`
- Returns: `{ "status": "ok" }`
- Use this to verify if the backend is up.

---

## 6. Error Handling
- `403 Forbidden`: No token provided.
- `401 Unauthorized`: Invalid or expired token.
- `429 Too Many Requests`: Rate limit exceeded.

---

## 7. Security
- Backend uses Helmet for HTTP security headers.
- Express hides the `x-powered-by` header.

---

## 8. Contact
- For any issues with API access, authentication, or CORS, contact the backend team. 