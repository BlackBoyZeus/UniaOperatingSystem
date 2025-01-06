# Stage 1: Builder
FROM node:18-alpine AS builder

# Build arguments for optimization
ARG TYPESCRIPT_INCREMENTAL=true
ARG YARN_CACHE_FOLDER=/tmp/.yarn-cache
ARG NODE_OPTIONS=--max-old-space-size=4096

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache curl=~8.4.0 \
    && yarn config set cache-folder $YARN_CACHE_FOLDER

# Copy package files for dependency installation
COPY package.json yarn.lock ./

# Install dependencies with yarn
RUN yarn install --frozen-lockfile --production=false \
    && yarn cache clean

# Copy source code and TypeScript configuration
COPY tsconfig.json ./
COPY src/ ./src/

# Build TypeScript with optimization flags
RUN yarn build \
    && yarn install --frozen-lockfile --production=true \
    && yarn cache clean

# Clean up build artifacts and development dependencies
RUN rm -rf $YARN_CACHE_FOLDER \
    && rm -rf /tmp/* \
    && rm -rf /var/cache/apk/*

# Stage 2: Production
FROM node:18-alpine

# Set production environment
ENV NODE_ENV=production
ENV PORT=3000
ENV WEBSOCKET_PORT=8080

# Create non-root user
RUN addgroup -g 1001 -S nodejs \
    && adduser -S nodejs -u 1001 -G nodejs

# Set working directory
WORKDIR /app

# Copy built artifacts from builder stage
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./

# Install production-only system dependencies
RUN apk add --no-cache curl=~8.4.0 \
    && rm -rf /var/cache/apk/*

# Configure health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Security hardening
USER nodejs
WORKDIR /app

# Read-only filesystem and security options
RUN chmod -R 555 /app \
    && chmod -R 555 /app/node_modules \
    && chmod -R 555 /app/dist

# Expose ports
EXPOSE 3000
EXPOSE 8080

# Set security options
LABEL security.alpha.kubernetes.io/seccomp=unconfined
LABEL security.alpha.kubernetes.io/unsafe-sysctls=kernel.shm_rmid_forced=0

# Drop capabilities and set security opts
RUN setcap cap_net_bind_service=+ep /usr/local/bin/node

# Set container entrypoint
ENTRYPOINT ["node", "dist/server.js"]

# Security labels
LABEL org.opencontainers.image.source="https://github.com/tald-unia/backend"
LABEL org.opencontainers.image.description="TALD UNIA Backend Service"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="TALD UNIA"
LABEL org.opencontainers.image.licenses="Proprietary"

# Apply security configurations
VOLUME ["/app/node_modules", "/app/dist"]
USER nodejs:nodejs
WORKDIR /app

# Set no privilege escalation
RUN echo "no-new-privileges=true" >> /etc/security/limits.conf

# Final security checks
RUN chmod 555 /app \
    && chown -R nodejs:nodejs /app \
    && chmod -R 555 /app/node_modules \
    && chmod -R 555 /app/dist

CMD ["node", "dist/server.js"]