import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Chat from './pages/Chat';
import Home from './pages/Home';
import Search from './pages/Search';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/chat/:sessionId" element={<Chat />} />
          <Route path="/search" element={<Search />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App; 