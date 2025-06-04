import React from 'react';
import { Link } from 'react-router-dom';
import { ChatBubbleLeftRightIcon, MagnifyingGlassIcon, BookOpenIcon, AcademicCapIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

const Home = () => {
  const features = [
    {
      name: 'AI Chat Assistant',
      description: 'Get instant answers to your research questions using our AI-powered chat interface. The assistant can help you understand complex papers, find relevant research, and provide insights.',
      icon: ChatBubbleLeftRightIcon,
      href: '/chat',
    },
    {
      name: 'Paper Search',
      description: 'Search through academic papers with advanced filtering options. Find papers by year, keywords, or categories. Get detailed paper information including abstracts, authors, and citations.',
      icon: MagnifyingGlassIcon,
      href: '/search',
    },
  ];

  const keyFeatures = [
    {
      title: 'Smart Research Assistant',
      description: 'Our AI-powered assistant understands research context and can help you navigate complex academic topics.',
      icon: AcademicCapIcon,
    },
    {
      title: 'Paper Analysis',
      description: 'Get detailed analysis of research papers, including methodology, key findings, and research gaps.',
      icon: DocumentTextIcon,
    },
    {
      title: 'Research Insights',
      description: 'Discover trends, patterns, and connections across multiple papers in your field of interest.',
      icon: BookOpenIcon,
    },
  ];

  return (
    <div className="relative isolate overflow-hidden">
      {/* Hero Section */}
      <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:py-40">
        <div className="mx-auto max-w-2xl lg:mx-0 lg:max-w-xl lg:flex-shrink-0 lg:pt-8">
          <h1 className="mt-10 text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Research Assistant
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Your AI-powered companion for academic research. Get instant answers to your questions, discover relevant papers, and gain deeper insights into your research topics.
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

      {/* Key Features Section */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8 py-24 sm:py-32">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">Key Features</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Everything you need for your research
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Our platform combines AI technology with academic research tools to help you work more efficiently.
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3">
            {keyFeatures.map((feature) => (
              <div key={feature.title} className="flex flex-col">
                <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900">
                  <feature.icon className="h-5 w-5 flex-none text-indigo-600" aria-hidden="true" />
                  {feature.title}
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                  <p className="flex-auto">{feature.description}</p>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {/* Usage Notes Section */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8 py-24 sm:py-32">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">Getting Started</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            How to use Research Assistant
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div className="rounded-2xl bg-white p-8 shadow-sm ring-1 ring-gray-200">
              <h3 className="text-lg font-semibold leading-8 text-gray-900">Chat Interface</h3>
              <ul className="mt-4 space-y-4 text-gray-600">
                <li className="flex gap-x-3">
                  <span className="text-indigo-600">•</span>
                  <span>Start a new chat to ask questions about research topics</span>
                </li>
                <li className="flex gap-x-3">
                  <span className="text-indigo-600">•</span>
                  <span>Get instant answers with relevant paper citations</span>
                </li>
                <li className="flex gap-x-3">
                  <span className="text-indigo-600">•</span>
                  <span>Save and manage your chat history for future reference</span>
                </li>
              </ul>
            </div>
            <div className="rounded-2xl bg-white p-8 shadow-sm ring-1 ring-gray-200">
              <h3 className="text-lg font-semibold leading-8 text-gray-900">Paper Search</h3>
              <ul className="mt-4 space-y-4 text-gray-600">
                <li className="flex gap-x-3">
                  <span className="text-indigo-600">•</span>
                  <span>Use filters to find papers by year or keywords</span>
                </li>
                <li className="flex gap-x-3">
                  <span className="text-indigo-600">•</span>
                  <span>View detailed paper information and abstracts</span>
                </li>
                <li className="flex gap-x-3">
                  <span className="text-indigo-600">•</span>
                  <span>Get AI-generated insights about research trends</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 