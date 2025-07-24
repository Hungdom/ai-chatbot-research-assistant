import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Chat from './pages/Chat';
import Home from './pages/Home';
import Search from './pages/Search';
import DatasetInsights from './pages/DatasetInsights';
import ChatbotMetrics from './pages/ChatbotMetrics';
import { ThemeProvider } from './context/ThemeContext';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/chat/:sessionId" element={<Chat />} />
            <Route path="/search" element={<Search />} />
            <Route path="/dataset" element={<DatasetInsights />} />
            <Route path="/metrics" element={<ChatbotMetrics />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App; 