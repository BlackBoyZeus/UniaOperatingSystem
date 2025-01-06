# Stage 1: Builder
FROM node:18-alpine AS builder

# Security hardening
RUN apk add --no-cache --update \
    python3 \
    make \
    g++ \
    curl \
    && addgroup -S appgroup && adduser -S appuser -G appgroup \
    && npm install -g npm@9.x \
    && npm set audit-level=high \
    && npm set progress=false \
    && npm config set fund false

# Set build arguments and environment
ARG API_BASE_URL
ARG WEBRTC_SIGNALING_URL
ARG BUILD_VERSION
ENV NODE_ENV=production
ENV VITE_API_BASE_URL=${API_BASE_URL}
ENV VITE_WEBRTC_SIGNALING_URL=${WEBRTC_SIGNALING_URL}
ENV NPM_CONFIG_LOGLEVEL=warn

# Set working directory and user
WORKDIR /app
USER appuser

# Copy package files with layer caching
COPY --chown=appuser:appgroup package*.json ./
COPY --chown=appuser:appgroup tsconfig*.json ./
COPY --chown=appuser:appgroup vite.config.ts ./

# Install dependencies with security checks
RUN npm ci --production --no-optional \
    && npm audit \
    && npm cache clean --force

# Copy source files
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup .env.production ./

# Build application with optimizations
RUN npm run build \
    && npm prune --production

# Stage 2: Production
FROM nginx:1.24-alpine

# Security hardening
RUN apk add --no-cache --update curl \
    && addgroup -S nginx \
    && adduser -S nginx -G nginx \
    && rm -rf /var/cache/apk/* \
    && rm -rf /usr/share/nginx/html/* \
    && touch /var/run/nginx.pid \
    && chown -R nginx:nginx /var/run/nginx.pid \
    && chown -R nginx:nginx /var/cache/nginx

# Copy nginx configuration
COPY infrastructure/docker/nginx.conf /etc/nginx/nginx.conf
COPY infrastructure/docker/security-headers.conf /etc/nginx/conf.d/security-headers.conf

# Copy built assets from builder
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html

# Configure security headers
RUN echo "add_header Content-Security-Policy \"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' ${API_BASE_URL} ${WEBRTC_SIGNALING_URL}\";" > /etc/nginx/conf.d/csp.conf \
    && echo "add_header X-Frame-Options \"DENY\";" >> /etc/nginx/conf.d/security-headers.conf \
    && echo "add_header X-Content-Type-Options \"nosniff\";" >> /etc/nginx/conf.d/security-headers.conf \
    && echo "add_header Referrer-Policy \"strict-origin-when-cross-origin\";" >> /etc/nginx/conf.d/security-headers.conf

# Set resource limits
ENV NGINX_WORKER_PROCESSES=auto
ENV NGINX_WORKER_CONNECTIONS=1024

# Configure volumes
VOLUME ["/var/cache/nginx", "/var/run"]

# Set user
USER nginx

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:80/health || exit 1

# Expose port
EXPOSE 80

# Set resource limits
LABEL org.opencontainers.image.memory="512M"
LABEL org.opencontainers.image.cpu="1.0"
LABEL org.opencontainers.image.pids="50"
LABEL org.opencontainers.image.nofile="1024"

# Start nginx
CMD ["nginx", "-g", "daemon off;"]