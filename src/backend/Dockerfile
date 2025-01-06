# Builder stage
FROM node:18-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++ curl

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src/ ./src/

# Build application
RUN npm run build

# Run security audit
RUN npm audit

# Prune dev dependencies
RUN npm prune --production

# Production stage
FROM node:18-alpine

# Install runtime dependencies
RUN apk add --no-cache curl tini

# Create non-root user
RUN addgroup -S appgroup && \
    adduser -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy production dependencies and built application
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

# Set secure permissions
RUN chown -R appuser:appgroup /app && \
    chmod -R 755 /app

# Configure security settings
ENV NODE_ENV=production
ENV PORT=3000
ENV NODE_OPTIONS="--max-old-space-size=4096 --optimize-for-size"
ENV UV_THREADPOOL_SIZE=64

# Configure resource limits
RUN ulimit -n 65536 && \
    ulimit -u 512

# Expose port
EXPOSE 3000

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Create volumes
VOLUME ["/app/node_modules", "/app/dist"]
RUN mkdir /tmp/tald && chown appuser:appgroup /tmp/tald
VOLUME ["/tmp/tald"]

# Switch to non-root user
USER appuser

# Use tini as init
ENTRYPOINT ["/sbin/tini", "--"]

# Start application with clustering
CMD ["node", "--require", "./dist/instrumentation.js", "./dist/index.js"]

# Security labels
LABEL org.opencontainers.image.vendor="TALD UNIA Team" \
      org.opencontainers.image.title="TALD UNIA Backend" \
      org.opencontainers.image.description="Backend server for TALD UNIA gaming platform" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT" \
      security.capabilities="cap_net_bind_service=+ep" \
      seccomp="unconfined"

# Security capabilities
RUN setcap 'cap_net_bind_service=+ep' /usr/local/bin/node

# Configure Node.js production optimizations
ENV NODE_ENV=production \
    NPM_CONFIG_LOGLEVEL=warn \
    NPM_CONFIG_PRODUCTION=true \
    NODE_OPTIONS="--max-old-space-size=4096 --optimize-for-size" \
    UV_THREADPOOL_SIZE=64

# Resource limits
ENV MEMORY_LIMIT=4G \
    CPU_SHARES=2048 \
    PIDS_LIMIT=512

# Performance optimizations
ENV NODE_CLUSTER_WORKERS=auto \
    NODE_COMPRESSION=true \
    NODE_KEEP_ALIVE_TIMEOUT=120000

# Security headers
ENV SECURITY_HEADERS_ENABLED=true \
    SECURITY_HEADERS_HSTS_ENABLED=true \
    SECURITY_HEADERS_CSP_ENABLED=true \
    SECURITY_HEADERS_REFERRER_POLICY="strict-origin-when-cross-origin"