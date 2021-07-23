FROM node:14.5.0
COPY . .
RUN npm config set //registry.npmjs.org/:_authToken 6fa99ac4-2d1c-44a0-8ea9-0720b90d416e
RUN npm install