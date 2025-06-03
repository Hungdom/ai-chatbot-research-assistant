import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

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
  const isUser = message.role === 'user';
  const timestamp = message.timestamp ? new Date(message.timestamp) : new Date();

  return (
    <div className={`chat-message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-header">
        <span className="message-role">{isUser ? 'You' : 'Assistant'}</span>
        <span className="message-time">
          {formatDate(timestamp)}
        </span>
      </div>
      <div className="message-content">
        {message.html_content ? (
          <div 
            className="html-content"
            dangerouslySetInnerHTML={{ __html: message.html_content }}
          />
        ) : (
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
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
      <style jsx>{`
        .chat-message {
          padding: 1rem;
          margin: 0.5rem 0;
          border-radius: 8px;
          max-width: 100%;
        }

        .user {
          background-color: #f0f7ff;
          margin-left: 2rem;
        }

        .assistant {
          background-color: #ffffff;
          margin-right: 2rem;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }

        .message-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
          font-size: 0.875rem;
        }

        .message-role {
          font-weight: 600;
          color: #2c3e50;
        }

        .message-time {
          color: #666;
          font-size: 0.75rem;
        }

        .message-content {
          line-height: 1.6;
        }

        .html-content {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        .html-content :global(.paper-card) {
          background: #fff;
          border: 1px solid #e1e4e8;
          border-radius: 6px;
          padding: 1.5rem;
          margin: 1rem 0;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .html-content :global(.paper-title) {
          margin: 0 0 0.5rem;
          font-size: 1.2rem;
        }

        .html-content :global(.paper-title a) {
          color: #0366d6;
          text-decoration: none;
        }

        .html-content :global(.paper-title a:hover) {
          text-decoration: underline;
        }

        .html-content :global(.paper-authors) {
          color: #586069;
          margin-bottom: 0.5rem;
        }

        .html-content :global(.category-tag) {
          display: inline-block;
          background: #e1ecf4;
          color: #39739d;
          padding: 0.2rem 0.6rem;
          border-radius: 3px;
          margin: 0.2rem;
          font-size: 0.9rem;
        }

        .html-content :global(.paper-abstract) {
          margin: 1rem 0;
        }

        .html-content :global(.paper-metadata) {
          color: #586069;
          font-size: 0.9rem;
          margin-top: 1rem;
        }

        .html-content :global(.paper-metadata span) {
          margin-right: 1rem;
        }

        .html-content :global(.similarity-score) {
          background: #f1f8ff;
          color: #0366d6;
          padding: 0.3rem 0.6rem;
          border-radius: 3px;
          display: inline-block;
          margin-top: 0.5rem;
        }

        .html-content :global(.section-title) {
          color: #2c3e50;
          border-bottom: 2px solid #eee;
          padding-bottom: 0.5rem;
          margin-top: 1.5rem;
        }

        .html-content :global(.trend-list),
        .html-content :global(.gaps-list),
        .html-content :global(.follow-up-list) {
          list-style: none;
          padding-left: 0;
        }

        .html-content :global(.trend-item) {
          color: #0366d6;
        }

        .html-content :global(.trend-count) {
          color: #586069;
          font-size: 0.9rem;
        }

        .html-content :global(.follow-up-list li) {
          margin: 0.5rem 0;
          padding-left: 1.5rem;
          position: relative;
        }

        .html-content :global(.follow-up-list li i) {
          position: absolute;
          left: 0;
          color: #0366d6;
        }

        /* Add Font Awesome for icons */
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
      `}</style>
    </div>
  );
};

export default ChatMessage; 