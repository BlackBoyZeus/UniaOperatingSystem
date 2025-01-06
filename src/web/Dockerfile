# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies and security tools
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    && npm install -g npm@9.x

# Copy package files with version lock
COPY package.json package-lock.json ./

# Install production dependencies with exact versions
RUN npm ci --production=false \
    && npm cache clean --force

# Run security audit
RUN npm audit \
    && npm run security-audit || true

# Copy source code and configuration files
COPY . .

# Build production bundle with optimization
ENV NODE_ENV=production
ENV VITE_API_URL=${API_URL}
RUN npm run build \
    && npm run analyze

# Clean development dependencies
RUN npm prune --production

# Production stage
FROM nginx:1.25-alpine

# Create non-root user
RUN adduser -D -H -u 101 -s /sbin/nologin nginx-user

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Security headers configuration
RUN echo 'add_header X-Frame-Options "SAMEORIGIN";' >> /etc/nginx/conf.d/security.conf \
    && echo 'add_header X-Content-Type-Options "nosniff";' >> /etc/nginx/conf.d/security.conf \
    && echo 'add_header X-XSS-Protection "1; mode=block";' >> /etc/nginx/conf.d/security.conf \
    && echo 'add_header Content-Security-Policy "default-src '\''self'\''; script-src '\''self'\'' '\''unsafe-inline'\'' '\''unsafe-eval'\''; style-src '\''self'\'' '\''unsafe-inline'\''; img-src '\''self'\'' data: blob:; font-src '\''self'\'' data:; connect-src '\''self'\'' ws: wss:;";' >> /etc/nginx/conf.d/security.conf

# Configure static file compression
RUN echo 'gzip on;' >> /etc/nginx/conf.d/gzip.conf \
    && echo 'gzip_vary on;' >> /etc/nginx/conf.d/gzip.conf \
    && echo 'gzip_min_length 10240;' >> /etc/nginx/conf.d/gzip.conf \
    && echo 'gzip_proxied expired no-cache no-store private auth;' >> /etc/nginx/conf.d/gzip.conf \
    && echo 'gzip_types text/plain text/css text/xml text/javascript application/javascript application/x-javascript application/xml application/json;' >> /etc/nginx/conf.d/gzip.conf

# Configure WebSocket support
RUN echo 'map $http_upgrade $connection_upgrade {' >> /etc/nginx/conf.d/websocket.conf \
    && echo '    default upgrade;' >> /etc/nginx/conf.d/websocket.conf \
    && echo '    '\'''' close;' >> /etc/nginx/conf.d/websocket.conf \
    && echo '}' >> /etc/nginx/conf.d/websocket.conf

# Set proper file permissions
RUN chown -R nginx-user:nginx-user /usr/share/nginx/html \
    && chmod -R 755 /usr/share/nginx/html \
    && chown -R nginx-user:nginx-user /var/cache/nginx \
    && chown -R nginx-user:nginx-user /var/log/nginx \
    && chown -R nginx-user:nginx-user /etc/nginx/conf.d \
    && touch /var/run/nginx.pid \
    && chown -R nginx-user:nginx-user /var/run/nginx.pid

# Configure Nginx worker settings
ENV NGINX_WORKER_PROCESSES=auto
ENV NGINX_WORKER_CONNECTIONS=1024

# Expose port
EXPOSE 80

# Switch to non-root user
USER nginx-user

# Configure healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]