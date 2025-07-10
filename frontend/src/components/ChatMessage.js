import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { UserCircleIcon, SparklesIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../context/ThemeContext';

const formatDate = (date) => {
  const d = new Date(date);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const month = months[d.getMonth()];
  const day = d.getDate();
  const hours = d.getHours();
  const minutes = d.getMinutes();
  const ampm = hours >= 12 ? 'pm' : 'am';
  const formattedHours = hours % 12 || 12;
  const formattedMinutes = minutes.toString().padStart(2, '0');
  return `${month} ${day}, ${formattedHours}:${formattedMinutes} ${ampm}`;
};

const ChatMessage = ({ message }) => {
  const { isDarkMode } = useTheme();
  const isUser = message.role === 'user';
  const timestamp = message.timestamp ? new Date(message.timestamp) : new Date();

  const proseClasses = isDarkMode
    ? 'prose-invert prose-headings:text-white prose-p:text-white prose-strong:text-white prose-code:text-white prose-pre:bg-gray-900 prose-pre:text-white [&_a]:text-indigo-300 [&_a:hover]:text-indigo-200 [&_ul]:text-white [&_ol]:text-white [&_li]:text-white [&_blockquote]:text-gray-300 [&_blockquote]:border-gray-600 [&_hr]:border-gray-600 [&_table]:text-white [&_th]:border-gray-600 [&_td]:border-gray-600 [&_*]:text-white prose-headings:font-semibold prose-headings:tracking-tight prose-p:leading-7 prose-ul:my-6 prose-ul:ml-6 prose-ul:list-disc prose-ol:my-6 prose-ol:ml-6 prose-ol:list-decimal prose-li:my-2 prose-li:leading-7 prose-blockquote:mt-6 prose-blockquote:border-l-2 prose-blockquote:pl-6 prose-blockquote:italic prose-hr:my-8 prose-table:w-full prose-table:border-collapse prose-th:border prose-th:border-gray-600 prose-th:px-4 prose-th:py-2 prose-td:border prose-td:border-gray-600 prose-td:px-4 prose-td:py-2'
    : 'prose-headings:text-gray-900 prose-p:text-gray-900 prose-strong:text-gray-900 prose-code:text-gray-900 prose-pre:bg-gray-100 prose-pre:text-gray-900 prose-headings:font-semibold prose-headings:tracking-tight prose-p:leading-7 prose-ul:my-6 prose-ul:ml-6 prose-ul:list-disc prose-ol:my-6 prose-ol:ml-6 prose-ol:list-decimal prose-li:my-2 prose-li:leading-7 prose-blockquote:mt-6 prose-blockquote:border-l-2 prose-blockquote:pl-6 prose-blockquote:italic prose-hr:my-8 prose-table:w-full prose-table:border-collapse prose-th:border prose-th:border-gray-200 prose-th:px-4 prose-th:py-2 prose-td:border prose-td:border-gray-200 prose-td:px-4 prose-td:py-2';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-3xl ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-2 sm:space-x-4`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 ${isUser ? 'ml-2 sm:ml-4' : 'mr-2 sm:mr-4'}`}>
          {isUser ? (
            <div className={`w-8 h-8 rounded-full ${isDarkMode ? 'bg-indigo-900' : 'bg-indigo-100'} flex items-center justify-center`}>
              <UserCircleIcon className={`w-6 h-6 ${isDarkMode ? 'text-indigo-300' : 'text-indigo-600'}`} />
            </div>
          ) : (
            <div className={`w-8 h-8 rounded-full ${isDarkMode ? 'bg-purple-900' : 'bg-purple-100'} flex items-center justify-center`}>
              <SparklesIcon className={`w-6 h-6 ${isDarkMode ? 'text-purple-300' : 'text-purple-600'}`} />
            </div>
          )}
        </div>

        {/* Message content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <div className="flex items-center space-x-2 mb-1">
            <span className={`text-sm font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {isUser ? 'You' : 'Assistant'}
            </span>
            <span className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>
              {formatDate(timestamp)}
            </span>
          </div>
          
          <div className={`
            rounded-2xl px-4 py-2.5 max-w-2xl
            ${isUser 
              ? 'bg-indigo-600 text-white' 
              : isDarkMode
                ? 'bg-gray-800 text-white shadow-sm ring-1 ring-gray-700'
                : 'bg-gray-50 text-gray-900 shadow-sm ring-1 ring-gray-200'
            }
          `}>
            {message.html_content ? (
              <div 
                className={`prose prose-sm sm:prose-base max-w-none ${proseClasses}`}
                dangerouslySetInnerHTML={{ __html: message.html_content }}
              />
            ) : (
              <ReactMarkdown
                className={`prose prose-sm sm:prose-base max-w-none ${proseClasses}`}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-lg"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage; 