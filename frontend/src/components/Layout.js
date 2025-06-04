import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  ChatBubbleLeftRightIcon, 
  MagnifyingGlassIcon,
  ChartBarIcon 
} from '@heroicons/react/24/outline';

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
];

const Layout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Navigation - Fixed at top */}
      <nav className="fixed top-0 left-0 right-0 bg-white shadow-sm z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link to="/" className="text-xl font-bold text-indigo-600">
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
                          ? 'border-indigo-500 text-gray-900'
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
          </div>
        </div>
      </nav>

      {/* Mobile navigation - Fixed below main nav */}
      <div className="fixed top-16 left-0 right-0 sm:hidden bg-white shadow-sm z-40">
        <div className="pt-2 pb-3 space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-3 py-2 text-base font-medium ${
                  isActive
                    ? 'bg-indigo-50 border-indigo-500 text-indigo-700'
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