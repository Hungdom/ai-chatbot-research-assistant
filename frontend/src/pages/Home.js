import React from 'react';
import { Link } from 'react-router-dom';
import { ChatBubbleLeftRightIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

const Home = () => {
  const features = [
    {
      name: 'AI Chat Assistant',
      description: 'Get instant answers to your research questions using our AI-powered chat interface.',
      icon: ChatBubbleLeftRightIcon,
      href: '/chat',
    },
    {
      name: 'Paper Search',
      description: 'Search through academic papers with advanced filtering options.',
      icon: MagnifyingGlassIcon,
      href: '/search',
    },
  ];

  return (
    <div className="relative isolate overflow-hidden">
      <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:py-40">
        <div className="mx-auto max-w-2xl lg:mx-0 lg:max-w-xl lg:flex-shrink-0 lg:pt-8">
          <h1 className="mt-10 text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Research Assistant
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Your AI-powered companion for academic research. Get instant answers to your questions and discover relevant papers with ease.
          </p>
          <div className="mt-10 flex items-center gap-x-6">
            <Link
              to="/chat"
              className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              Get started
            </Link>
            <Link to="/search" className="text-sm font-semibold leading-6 text-gray-900">
              Search papers <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
        <div className="mx-auto mt-16 flex max-w-2xl sm:mt-24 lg:ml-10 lg:mr-0 lg:mt-0 lg:max-w-none lg:flex-none xl:ml-32">
          <div className="max-w-3xl flex-none sm:max-w-5xl lg:max-w-none">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2">
              {features.map((feature) => (
                <Link
                  key={feature.name}
                  to={feature.href}
                  className="flex flex-col items-start rounded-2xl bg-white p-8 shadow-sm ring-1 ring-gray-200 hover:ring-indigo-500 transition-all"
                >
                  <div className="rounded-lg bg-indigo-50 p-2">
                    <feature.icon className="h-6 w-6 text-indigo-600" aria-hidden="true" />
                  </div>
                  <h3 className="mt-4 text-lg font-semibold leading-8 text-gray-900">{feature.name}</h3>
                  <p className="mt-2 text-base leading-7 text-gray-600">{feature.description}</p>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 