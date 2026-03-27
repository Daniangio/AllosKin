#
# Dockerfile for the PHASE Frontend (React + Tailwind)
# PRODUCTION build
#

FROM node:18-alpine AS builder
WORKDIR /app

COPY package.json ./
COPY package-lock.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:1.25-alpine
ENV NGINX_ENTRYPOINT_WORKER_PROCESSES_AUTOTUNE=0

COPY --from=builder /app/build /usr/share/nginx/html

RUN printf '%s\n' \
    'worker_processes 2;' \
    'pid /var/run/nginx.pid;' \
    'events { worker_connections 1024; }' \
    'http {' \
    '  include /etc/nginx/mime.types;' \
    '  default_type application/octet-stream;' \
    '  sendfile on;' \
    '  keepalive_timeout 65;' \
    '  include /etc/nginx/conf.d/*.conf;' \
    '}' > /etc/nginx/nginx.conf

RUN echo "server { \
      listen 3000; \
      location /api/ { \
        proxy_pass http://backend:8000; \
        proxy_http_version 1.1; \
        proxy_set_header Host \$host; \
        proxy_set_header X-Real-IP \$remote_addr; \
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto \$scheme; \
      } \
      location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files \$uri \$uri/ /index.html; \
      } \
    }" > /etc/nginx/conf.d/default.conf

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]