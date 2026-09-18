# Multi-stage build for the Next.js frontend.
#
# NEXT_PUBLIC_* env vars are inlined into the client bundle at build time,
# so they must be passed as build args (not just runtime env) for a
# production image -- see NEXT_PUBLIC_API_URL/NEXT_PUBLIC_WS_URL below.
# docker build --build-arg NEXT_PUBLIC_API_URL=http://api.example.com \
#              --build-arg NEXT_PUBLIC_WS_URL=ws://api.example.com .

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_WS_URL=ws://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["npm", "run", "start"]
