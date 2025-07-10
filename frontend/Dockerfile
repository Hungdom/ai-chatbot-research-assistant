FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Create public directory if it doesn't exist
RUN mkdir -p public

# Install additional dependencies
RUN npm install react-markdown remark-gfm react-syntax-highlighter chart.js react-chartjs-2

EXPOSE 3000

CMD ["npm", "start"] 