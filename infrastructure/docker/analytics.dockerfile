# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /usr/src/app

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    && yarn global add typescript@4.9.5

# Copy package files
COPY package.json yarn.lock ./

# Install dependencies with yarn
RUN yarn install --frozen-lockfile \
    && yarn cache clean

# Copy source code and configs
COPY tsconfig.json ./
COPY src ./src

# Build TypeScript code
RUN yarn build \
    && yarn audit \
    && yarn install --production --ignore-scripts --prefer-offline \
    && yarn cache clean

# Production stage
FROM node:18-alpine

# Install production dependencies
RUN apk add --no-cache \
    tini \
    curl \
    && addgroup -S tald \
    && adduser -S -G tald tald

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV NODE_ENV=production \
    METRICS_NAMESPACE=TALD/Metrics \
    ANALYTICS_NAMESPACE=TALD/Analytics \
    NODE_OPTIONS="--max-old-space-size=4096" \
    PROMETHEUS_PORT=9090 \
    HEALTH_CHECK_INTERVAL=30

# Copy built artifacts from builder
COPY --from=builder --chown=tald:tald /usr/src/app/dist ./dist
COPY --from=builder --chown=tald:tald /usr/src/app/node_modules ./node_modules
COPY --chown=tald:tald infrastructure/scripts/health-check.sh ./health-check.sh

# Make health check script executable
RUN chmod +x ./health-check.sh

# Set up security measures
RUN mkdir -p /usr/src/app/logs \
    && chown -R tald:tald /usr/src/app \
    && chmod -R 755 /usr/src/app

# Switch to non-root user
USER tald

# Expose ports
EXPOSE 9090

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ./health-check.sh || exit 1

# Use tini as init
ENTRYPOINT ["/sbin/tini", "--"]

# Start analytics service
CMD ["node", "--enable-source-maps", "dist/services/analytics/index.js"]

# Labels
LABEL maintainer="TALD UNIA Development Team" \
    version="1.0.0" \
    description="TALD UNIA Analytics Service" \
    org.opencontainers.image.source="https://github.com/tald-unia/analytics-service" \
    org.opencontainers.image.vendor="TALD UNIA" \
    org.opencontainers.image.title="Analytics Service" \
    org.opencontainers.image.description="Analytics service for processing metrics, LiDAR data, and game sessions" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.created="2023-09-14"