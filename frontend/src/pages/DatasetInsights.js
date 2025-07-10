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
import { useTheme } from '../context/ThemeContext';

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
  totalPapers: 1280000,
  totalAuthors: 1238056,
  papersByYear: [
    { year: 2007, count: 12729 },
    { year: 2008, count: 5516 },
    { year: 2009, count: 32191 },
    { year: 2010, count: 1268 },
    { year: 2011, count: 1966 },
    { year: 2012, count: 246 },
    { year: 2013, count: 54 },
    { year: 2014, count: 537 },
    { year: 2015, count: 2514 },
    { year: 2016, count: 2189 },
    { year: 2017, count: 322 },
    { year: 2018, count: 80278 },
    { year: 2019, count: 128603 },
    { year: 2020, count: 164128 },
    { year: 2021, count: 179210 },
    { year: 2022, count: 185280 },
    { year: 2023, count: 215057 },
    { year: 2024, count: 280845 },
    { year: 2025, count: 147067 }
  ],
  topCategories: [
    { name: ["cs.CV"], count: 61667 },
    { name: ["astro-ph"], count: 55208 },
    { name: ["quant-ph"], count: 29485 },
    { name: ["cs.CL"], count: 27177 },
    { name: ["math.AP"], count: 20701 },
    { name: ["cs.LG"], count: 20010 },
    { name: ["astro-ph.GA"], count: 19383 },
    { name: ["cond-mat.mtrl-sci"], count: 18325 },
    { name: ["math.CO"], count: 16855 },
    { name: ["hep-ph"], count: 15564 }
  ],
  topAuthors: [
    { name: ["CMS Collaboration"], paperCount: 651 },
    { name: ["ATLAS Collaboration"], paperCount: 647 },
    { name: ["ALICE Collaboration"], paperCount: 331 },
    { name: ["Noam Soker (Technion", "Israel)"], paperCount: 73 },
    { name: ["Shahar Hod"], paperCount: 73 },
    { name: ["Andronikos Paliathanasis"], paperCount: 68 },
    { name: ["Naoki Kitazawa"], paperCount: 68 },
    { name: ["Stefan Steinerberger"], paperCount: 68 },
    { name: ["Stefano Longhi"], paperCount: 66 },
    { name: ["Sabah Al-Fedaghi"], paperCount: 62 }
  ]
};

function DatasetInsights() {
  const { isDarkMode } = useTheme();

  // Prepare chart data
  const papersByYearData = {
    labels: staticData.papersByYear.map(item => item.year),
    datasets: [
      {
        label: 'Papers Published',
        data: staticData.papersByYear.map(item => item.count),
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        tension: 0.1,
        fill: true,
      },
    ],
  };

  const topCategoriesData = {
    labels: staticData.topCategories.map(cat => cat.name[0]),
    datasets: [
      {
        label: 'Number of Papers',
        data: staticData.topCategories.map(cat => cat.count),
        backgroundColor: 'rgba(99, 102, 241, 0.5)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Number of Papers Published by Year',
        color: isDarkMode ? '#fff' : '#000',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      legend: {
        labels: {
          color: isDarkMode ? '#fff' : '#000',
        },
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Papers: ${context.raw.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      y: {
        ticks: {
          color: isDarkMode ? '#fff' : '#000',
          callback: function(value) {
            return value >= 1000 ? (value/1000).toFixed(0) + 'k' : value;
          }
        },
        grid: {
          color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        },
        title: {
          display: true,
          text: 'Number of Papers',
          color: isDarkMode ? '#fff' : '#000',
        }
      },
      x: {
        ticks: {
          color: isDarkMode ? '#fff' : '#000',
        },
        grid: {
          color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        },
        title: {
          display: true,
          text: 'Year',
          color: isDarkMode ? '#fff' : '#000',
        }
      },
    },
  };

  const barChartOptions = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      title: {
        ...chartOptions.plugins.title,
        text: 'Top Research Categories',
      },
    },
    scales: {
      ...chartOptions.scales,
      x: {
        ...chartOptions.scales.x,
        title: {
          ...chartOptions.scales.x.title,
          text: 'Category',
        },
      },
    },
  };

  return (
    <div className={`container mx-auto px-4 py-8 ${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <h1 className={`text-3xl font-bold mb-8 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>arXiv Dataset Insights</h1>
      
      {/* Dataset Overview */}
      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-6 mb-8`}>
        <h2 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Dataset Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
            <h3 className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>Total Papers</h3>
            <p className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{staticData.totalPapers.toLocaleString()}</p>
          </div>
          <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
            <h3 className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>Total Authors</h3>
            <p className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{staticData.totalAuthors.toLocaleString()}</p>
          </div>
          <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
            <h3 className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>Categories</h3>
            <p className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{staticData.topCategories.length}</p>
          </div>
          <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
            <h3 className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>Years Covered</h3>
            <p className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {staticData.papersByYear[0].year} - {staticData.papersByYear[staticData.papersByYear.length - 1].year}
            </p>
          </div>
          <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
            <h3 className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>Peak Year</h3>
            <p className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {staticData.papersByYear.reduce((max, current) => 
                current.count > max.count ? current : max
              ).year}
            </p>
          </div>
          <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
            <h3 className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>Average Papers/Year</h3>
            <p className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {Math.round(staticData.papersByYear.reduce((sum, current) => sum + current.count, 0) / staticData.papersByYear.length).toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Papers by Year */}
      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-6 mb-8`}>
        <h2 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Publication Trends</h2>
        <div className="h-96">
          <Line data={papersByYearData} options={chartOptions} />
        </div>
      </div>

      {/* Top Categories */}
      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-6 mb-8`}>
        <h2 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Top Research Categories</h2>
        <div className="h-96">
          <Bar data={topCategoriesData} options={barChartOptions} />
        </div>
      </div>

      {/* Top Authors */}
      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-6`}>
        <h2 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Most Prolific Authors</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className={isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}>
              <tr>
                <th className={`px-6 py-3 text-left text-xs font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Author
                </th>
                <th className={`px-6 py-3 text-left text-xs font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Papers
                </th>
              </tr>
            </thead>
            <tbody className={`divide-y ${isDarkMode ? 'divide-gray-700 bg-gray-800' : 'divide-gray-200 bg-white'}`}>
              {staticData.topAuthors.map((author, index) => (
                <tr key={index}>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {Array.isArray(author.name) ? author.name.join(' ') : author.name}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>
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