import React from 'react';

const PaperCard = ({ paper }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-4 mb-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{paper.title}</h3>
      <div className="text-sm text-gray-600 mb-2">
        <p>Authors: {paper.authors.join(', ')}</p>
        <p>Year: {paper.year}</p>
      </div>
      <div className="flex flex-wrap gap-2 mb-3">
        {paper.keywords.map((keyword, index) => (
          <span
            key={index}
            className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
          >
            {keyword}
          </span>
        ))}
      </div>
      <p className="text-gray-700 text-sm">{paper.abstract}</p>
    </div>
  );
};

export default PaperCard; 