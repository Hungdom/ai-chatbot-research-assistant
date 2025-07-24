import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  ChatBubbleLeftRightIcon, 
  MagnifyingGlassIcon,
  ChartBarIcon,
  BeakerIcon,
  SunIcon,
  MoonIcon
} from '@heroicons/react/24/outline';
import { useTheme } from '../context/ThemeContext';

const navigation = [
  {
    name: 'Chat',
    href: '/chat',
    icon: ChatBubbleLeftRightIcon,
  },
  {
    name: 'Search',
    href: '/search',
    icon: MagnifyingGlassIcon,
  },
  {
    name: 'Dataset',
    href: '/dataset',
    icon: ChartBarIcon,
  },
  {
    name: 'Metrics',
    href: '/metrics',
    icon: BeakerIcon,
  },
];

const Layout = ({ children }) => {
  const location = useLocation();
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <div className={`min-h-screen ${isDarkMode ? 'dark bg-gray-900' : 'bg-gray-100'}`}>
      {/* Navigation - Fixed at top */}
      <nav className={`fixed top-0 left-0 right-0 ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm z-50`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link to="/" className={`text-xl font-bold ${isDarkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>
                  Research Assistant
                </Link>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                {navigation.map((item) => {
                  const isActive = location.pathname.startsWith(item.href);
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                        isActive
                          ? isDarkMode
                            ? 'border-indigo-400 text-gray-100'
                            : 'border-indigo-500 text-gray-900'
                          : isDarkMode
                            ? 'border-transparent text-gray-300 hover:border-gray-600 hover:text-gray-100'
                            : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                      }`}
                    >
                      <item.icon className="h-5 w-5 mr-2" />
                      {item.name}
                    </Link>
                  );
                })}
              </div>
            </div>
            {/* Theme Toggle Button */}
            <div className="flex items-center">
              <button
                onClick={toggleTheme}
                className={`p-2 rounded-lg ${
                  isDarkMode
                    ? 'text-gray-100 hover:bg-gray-700'
                    : 'text-gray-500 hover:bg-gray-100'
                }`}
                aria-label="Toggle theme"
              >
                {isDarkMode ? (
                  <SunIcon className="h-6 w-6" />
                ) : (
                  <MoonIcon className="h-6 w-6" />
                )}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile navigation - Fixed below main nav */}
      <div className={`fixed top-16 left-0 right-0 sm:hidden ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm z-40`}>
        <div className="pt-2 pb-3 space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-3 py-2 text-base font-medium ${
                  isActive
                    ? isDarkMode
                      ? 'bg-gray-700 border-indigo-400 text-indigo-400'
                      : 'bg-indigo-50 border-indigo-500 text-indigo-700'
                    : isDarkMode
                      ? 'border-transparent text-gray-300 hover:bg-gray-700 hover:border-gray-600 hover:text-gray-100'
                      : 'border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700'
                }`}
              >
                <item.icon className="h-5 w-5 mr-2" />
                {item.name}
              </Link>
            );
          })}
        </div>
      </div>

      {/* Main content - Add padding to account for fixed nav */}
      <main className="pt-16 sm:pt-16">
        {children}
      </main>
    </div>
  );
};

export default Layout; 