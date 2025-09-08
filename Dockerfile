# 🚀 SIP BREWERY BACKEND - PRODUCTION DOCKERFILE
# Multi-stage build for optimized production deployment

# ===== BUILD STAGE =====
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++ git

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Remove development files
RUN rm -rf __tests__ tests *.test.js *.spec.js coverage docs

# ===== PYTHON DEPENDENCIES STAGE =====
FROM python:3.11-alpine AS python-builder

# Install Python dependencies for ASI system
WORKDIR /python-app

# Copy Python requirements
COPY requirements.txt ./
COPY asi/ ./asi/
COPY financial-asi/ ./financial-asi/
COPY src/unified-asi/python-asi/ ./unified-asi/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ===== PRODUCTION STAGE =====
FROM node:18-alpine AS production

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    dumb-init \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S sipbrewery -u 1001

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=sipbrewery:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=sipbrewery:nodejs /app/src ./src
COPY --from=builder --chown=sipbrewery:nodejs /app/package*.json ./
COPY --from=builder --chown=sipbrewery:nodejs /app/ecosystem.config.js ./

# Copy Python ASI components
COPY --from=python-builder --chown=sipbrewery:nodejs /python-app ./python-asi

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache && \
    chown -R sipbrewery:nodejs /app

# Copy production configuration
COPY --chown=sipbrewery:nodejs docker/production.env .env

# Install PM2 globally
RUN npm install -g pm2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Switch to non-root user
USER sipbrewery

# Expose ports
EXPOSE 3000 8001

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start application with PM2
CMD ["pm2-runtime", "start", "ecosystem.config.js", "--env", "production"]
