import React, { useState } from 'react';
import axios from 'axios';
import ArxivCard from '../components/ArxivCard';
import { useTheme } from '../context/ThemeContext';

const Search = () => {
  const { isDarkMode } = useTheme();
  const [yearFilter, setYearFilter] = useState('');
  const [keywords, setKeywords] = useState('');
  const [arxiv, setArxiv] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSummary('');
    setArxiv([]);

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/search`, {
        year: yearFilter ? parseInt(yearFilter) : null,
        keywords: keywords ? keywords.split(',').map(k => k.trim()) : []
      });

      if (response.data.papers) {
        setArxiv(response.data.papers);
        setSummary(response.data.summary || '');
      } else {
        setError('No papers found matching your criteria.');
      }
    } catch (err) {
      console.error('Search error:', err);
      if (err.response) {
        setError(err.response.data.message || 'Failed to fetch arxiv papers. Please try again.');
      } else if (err.request) {
        setError('Unable to connect to the server. Please check your connection.');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`container mx-auto px-4 py-8 ${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <div className="max-w-3xl mx-auto space-y-6">
        <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow sm:rounded-lg`}>
          <div className="px-4 py-5 sm:p-6">
            <h3 className={`text-lg font-medium leading-6 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Search ArXiv Papers</h3>
            <div className="mt-2 max-w-xl text-sm text-gray-500">
              <p>Use filters to find relevant academic papers from ArXiv.</p>
            </div>
            <form onSubmit={handleSearch} className="mt-5 space-y-4">
              <div>
                <label htmlFor="year" className={`block text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Year
                </label>
                <input
                  type="number"
                  name="year"
                  id="year"
                  value={yearFilter}
                  onChange={(e) => setYearFilter(e.target.value)}
                  className={`mt-1 block w-full rounded-md ${isDarkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300'} shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm`}
                  placeholder="e.g., 2023"
                  min="1991"
                  max={new Date().getFullYear()}
                />
              </div>
              <div>
                <label htmlFor="keywords" className={`block text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Keywords (comma-separated)
                </label>
                <input
                  type="text"
                  name="keywords"
                  id="keywords"
                  value={keywords}
                  onChange={(e) => setKeywords(e.target.value)}
                  className={`mt-1 block w-full rounded-md ${isDarkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300'} shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm`}
                  placeholder="e.g., AI, machine learning, deep learning"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Searching...
                  </>
                ) : (
                  'Search'
                )}
              </button>
            </form>
          </div>
        </div>

        {error && (
          <div className="rounded-md bg-red-50 p-4">
            <div className="flex">
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">{error}</h3>
              </div>
            </div>
          </div>
        )}

        {summary && (
          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow sm:rounded-lg p-4`}>
            <h3 className={`text-lg font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'} mb-2`}>Summary</h3>
            <p className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>{summary}</p>
          </div>
        )}

        <div className="space-y-4">
          {arxiv.map((item, index) => (
            <ArxivCard key={index} arxiv={item} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Search; 