import React from 'react';

const ArxivCard = ({ arxiv }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-4 mb-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{arxiv.title}</h3>
      <div className="text-sm text-gray-600 mb-2">
        <p>Authors: {Array.isArray(arxiv.authors) ? arxiv.authors.join(', ') : arxiv.authors}</p>
        <p>Update Date: {new Date(arxiv.update_date).toLocaleDateString()}</p>
      </div>
      {arxiv.categories && arxiv.categories.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {arxiv.categories.map((category, index) => (
            <span
              key={index}
              className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
            >
              {category}
            </span>
          ))}
        </div>
      )}
      <p className="text-gray-700 text-sm">{arxiv.abstract}</p>
    </div>
  );
};

export default ArxivCard; 