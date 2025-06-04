import React from 'react';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Static data
const staticData = {
  totalPapers: 1440000,
  totalAuthors: 1238056,
  papersByYear: [
    { year: 2018, count: 150000 },
    { year: 2019, count: 180000 },
    { year: 2020, count: 220000 },
    { year: 2021, count: 250000 },
    { year: 2022, count: 280000 },
    { year: 2023, count: 300000 }
  ],
  topCategories: [
    { name: 'Computer Science', count: 800000 },
    { name: 'Physics', count: 600000 },
    { name: 'Mathematics', count: 400000 },
    { name: 'Statistics', count: 300000 },
    { name: 'Quantitative Biology', count: 200000 }
  ],
  topAuthors: [
    { name: 'John Smith', paperCount: 150 },
    { name: 'Maria Garcia', paperCount: 120 },
    { name: 'David Chen', paperCount: 100 },
    { name: 'Sarah Johnson', paperCount: 95 },
    { name: 'Michael Brown', paperCount: 90 }
  ]
};

function DatasetInsights() {
  // Prepare chart data
  const papersByYearData = {
    labels: staticData.papersByYear.map(item => item.year),
    datasets: [
      {
        label: 'Papers Published',
        data: staticData.papersByYear.map(item => item.count),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const topCategoriesData = {
    labels: staticData.topCategories.map(cat => cat.name),
    datasets: [
      {
        label: 'Number of Papers',
        data: staticData.topCategories.map(cat => cat.count),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      },
    ],
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">arXiv Dataset Insights</h1>
      
      {/* Dataset Overview */}
      <div className="bg-white rounded-lg shadow p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Dataset Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-gray-500 text-sm font-medium">Total Papers</h3>
            <p className="text-3xl font-bold">{staticData.totalPapers.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-gray-500 text-sm font-medium">Total Authors</h3>
            <p className="text-3xl font-bold">{staticData.totalAuthors.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-gray-500 text-sm font-medium">Categories</h3>
            <p className="text-3xl font-bold">{staticData.topCategories.length}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-gray-500 text-sm font-medium">Years Covered</h3>
            <p className="text-3xl font-bold">
              {staticData.papersByYear[0].year} - {staticData.papersByYear[staticData.papersByYear.length - 1].year}
            </p>
          </div>
        </div>
      </div>

      {/* Papers by Year */}
      <div className="bg-white rounded-lg shadow p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Publication Trends</h2>
        <div className="h-96">
          <Line 
            data={papersByYearData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                title: {
                  display: true,
                  text: 'Number of Papers Published by Year'
                }
              }
            }}
          />
        </div>
      </div>

      {/* Top Categories */}
      <div className="bg-white rounded-lg shadow p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Top Research Categories</h2>
        <div className="h-96">
          <Bar 
            data={topCategoriesData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                title: {
                  display: true,
                  text: 'Most Active Research Areas'
                }
              }
            }}
          />
        </div>
      </div>

      {/* Top Authors */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Most Prolific Authors</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Author
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Papers
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {staticData.topAuthors.map((author, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {author.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {author.paperCount.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default DatasetInsights; 