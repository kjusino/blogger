# Stage 1: build the React app
FROM node:lts-alpine AS build
WORKDIR /usr/src/app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: runtime — Express serves the build and the /api/personal routes
FROM node:lts-alpine
ENV NODE_ENV=production
ENV PORT=3000
WORKDIR /usr/src/app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev && npm i --no-save tsx
COPY --from=build /usr/src/app/build ./build
COPY --from=build /usr/src/app/server ./server
EXPOSE 3000
RUN chown -R node /usr/src/app
USER node
CMD ["npx", "tsx", "server/index.ts"]
