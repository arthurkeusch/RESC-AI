FROM node:23-alpine

WORKDIR /app

RUN apk add --no-cache python3 make g++ zstd pigz

COPY package*.json ./

RUN npm ci --only=production

COPY . .

RUN mkdir -p /app/models

EXPOSE 3000

CMD ["node", "index.js"]
