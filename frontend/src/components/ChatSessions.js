import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { TrashIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../context/ThemeContext';

const ChatSessions = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();
  const { isDarkMode } = useTheme();

  const loadSessions = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/api/sessions`);
      // Sort sessions by updated_at in descending order
      const sortedSessions = response.data.sort((a, b) => 
        new Date(b.updated_at) - new Date(a.updated_at)
      );
      setSessions(sortedSessions);
    } catch (error) {
      console.error('Error loading sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load sessions initially and when location changes
  useEffect(() => {
    loadSessions();
  }, [location.pathname]);

  const handleNewChat = () => {
    // Navigate to the root chat path to start a new chat
    navigate('/chat');
  };

  const handleDeleteSession = async (e, sessionId) => {
    e.preventDefault();
    e.stopPropagation();
    
    try {
      await axios.delete(`${process.env.REACT_APP_API_URL}/api/sessions/${sessionId}`);
      // If we're currently viewing the deleted session, redirect to new chat
      if (location.pathname === `/chat/${sessionId}`) {
        navigate('/chat');
      }
      // Reload sessions list
      loadSessions();
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  const getSessionPreview = (session) => {
    if (!session.messages || session.messages.length === 0) {
      return 'New conversation';
    }
    // Get the last user message or assistant message
    const lastMessage = [...session.messages].reverse().find(msg => 
      msg.role === 'user' || msg.role === 'assistant'
    );
    if (!lastMessage) return 'New conversation';
    
    // For assistant messages, show the content
    // For user messages, show "You: " followed by the content
    const prefix = lastMessage.role === 'user' ? 'You: ' : '';
    return prefix + lastMessage.content.substring(0, 50) + 
      (lastMessage.content.length > 50 ? '...' : '');
  };

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className={`w-64 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-r h-full flex flex-col`}>
      <div className={`p-4 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <button
          onClick={handleNewChat}
          className="w-full bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          New Chat
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className={`p-4 text-center ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div className={`p-4 text-center ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No chat sessions yet</div>
        ) : (
          <div className={`divide-y ${isDarkMode ? 'divide-gray-700' : 'divide-gray-200'}`}>
            {sessions.map((session) => (
              <Link
                key={session.session_id}
                to={`/chat/${session.session_id}`}
                className={`block p-4 ${
                  location.pathname === `/chat/${session.session_id}`
                    ? isDarkMode
                      ? 'bg-gray-700'
                      : 'bg-gray-50'
                    : isDarkMode
                      ? 'hover:bg-gray-700'
                      : 'hover:bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium ${isDarkMode ? 'text-gray-100' : 'text-gray-900'} truncate`}>
                      {getSessionPreview(session)}
                    </p>
                    <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'} mt-1`}>
                      {formatDate(session.updated_at)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(e, session.session_id)}
                    className={`ml-2 ${isDarkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-400 hover:text-gray-500'}`}
                  >
                    <TrashIcon className="h-5 w-5" />
                  </button>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatSessions; 