const express = require('express');
const cors = require('cors');
const chatRoutes = require('./routes/chat');
const searchRoutes = require('./routes/search');

const app = express();

app.use(cors());
app.use(express.json());

// Routes
app.use('/api/chat', chatRoutes);
app.use('/api/search', searchRoutes);

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
}); 