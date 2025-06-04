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
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left Column - Text Content */}
          <div className="max-w-2xl mx-auto lg:mx-0">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900">
              Research Assistant
            </h1>
            <p className="mt-6 text-lg sm:text-xl leading-8 text-gray-600">
              Your AI-powered companion for academic research. Get instant answers to your questions, discover relevant papers, and gain deeper insights into your research topics.
            </p>
            <div className="mt-8 sm:mt-10 flex flex-col sm:flex-row items-center gap-4 sm:gap-6">
              <Link
                to="/chat"
                className="w-full sm:w-auto rounded-md bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 text-center"
              >
                Get started
              </Link>
              <Link 
                to="/search" 
                className="w-full sm:w-auto text-base font-semibold leading-6 text-gray-900 text-center"
              >
                Search papers <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>

          {/* Right Column - AI Assistant Mockup */}
          <div className="relative max-w-2xl mx-auto lg:mx-0">
            <div className="absolute -inset-4 bg-gradient-to-r from-indigo-100 to-purple-100 rounded-2xl opacity-50 blur-xl"></div>
            <div className="relative bg-white rounded-2xl shadow-xl overflow-hidden">
              <div className="p-6">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">AI Research Assistant</h3>
                    <p className="text-sm text-gray-500">Ready to help with your research</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">What are the latest developments in quantum computing?</p>
                  </div>
                  <div className="bg-indigo-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Based on recent papers, there have been significant advances in quantum error correction and quantum supremacy demonstrations...</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800">
                        arXiv:2301.12345
                      </span>
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800">
                        Nature, 2023
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Features Section */}
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
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
          <dl className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
            {keyFeatures.map((feature) => (
              <div key={feature.title} className="flex flex-col bg-white p-6 rounded-2xl shadow-sm ring-1 ring-gray-200">
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
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">Getting Started</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            How to use Research Assistant
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2">
            <div className="rounded-2xl bg-white p-6 sm:p-8 shadow-sm ring-1 ring-gray-200">
              <div className="flex items-center space-x-4 mb-6">
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center">
                  <ChatBubbleLeftRightIcon className="w-6 h-6 text-indigo-600" />
                </div>
                <h3 className="text-lg font-semibold leading-8 text-gray-900">Chat Interface</h3>
              </div>
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
            <div className="rounded-2xl bg-white p-6 sm:p-8 shadow-sm ring-1 ring-gray-200">
              <div className="flex items-center space-x-4 mb-6">
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center">
                  <MagnifyingGlassIcon className="w-6 h-6 text-indigo-600" />
                </div>
                <h3 className="text-lg font-semibold leading-8 text-gray-900">Paper Search</h3>
              </div>
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

      {/* Decorative Elements */}
      <div className="absolute inset-x-0 top-0 -z-10 transform-gpu overflow-hidden blur-3xl" aria-hidden="true">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"></div>
      </div>
      <div className="absolute inset-x-0 bottom-0 -z-10 transform-gpu overflow-hidden blur-3xl" aria-hidden="true">
        <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]"></div>
      </div>
    </div>
  );
};

export default Home; 