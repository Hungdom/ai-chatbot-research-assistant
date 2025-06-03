import React, { useState } from 'react';
import axios from 'axios';
import ArxivCard from '../components/ArxivCard';

const Search = () => {
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

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/chat`, {
        query: 'search arxiv',
        year_filter: yearFilter ? parseInt(yearFilter) : null,
        keywords: keywords ? keywords.split(',').map(k => k.trim()) : null
      });

      setArxiv(response.data.arxiv || []);
      setSummary(response.data.response || '');
    } catch (err) {
      setError('Failed to fetch arxiv papers. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">Search ArXiv Papers</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>Use filters to find relevant academic papers from ArXiv.</p>
          </div>
          <form onSubmit={handleSearch} className="mt-5 space-y-4">
            <div>
              <label htmlFor="year" className="block text-sm font-medium text-gray-700">
                Year
              </label>
              <input
                type="number"
                name="year"
                id="year"
                value={yearFilter}
                onChange={(e) => setYearFilter(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                placeholder="e.g., 2023"
              />
            </div>
            <div>
              <label htmlFor="keywords" className="block text-sm font-medium text-gray-700">
                Keywords (comma-separated)
              </label>
              <input
                type="text"
                name="keywords"
                id="keywords"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                placeholder="e.g., AI, machine learning, deep learning"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              {loading ? 'Searching...' : 'Search'}
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
        <div className="bg-white shadow sm:rounded-lg p-4">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Summary</h3>
          <p className="text-gray-700">{summary}</p>
        </div>
      )}

      <div className="space-y-4">
        {arxiv.map((item, index) => (
          <ArxivCard key={index} arxiv={item} />
        ))}
      </div>
    </div>
  );
};

export default Search; 