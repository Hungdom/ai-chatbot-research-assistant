# Research Assistant Frontend

A modern, responsive React application that provides an intuitive interface for AI-powered academic paper search and research assistance.

## 🚀 Overview

This frontend application serves as the user interface for the Research Assistant platform, offering:

- **Interactive Chat Interface**: AI-powered conversational search and research assistance
- **Advanced Paper Search**: Comprehensive academic paper discovery with filtering
- **Dataset Insights**: Analytics and visualizations of research trends
- **Responsive Design**: Mobile-first design with dark/light theme support
- **Modern UI/UX**: Clean, professional interface built with React and Tailwind CSS

## 🛠️ Technology Stack

### Core Technologies
- **React 18.2.0**: Modern React with hooks and concurrent features
- **React Router DOM 6.22.1**: Client-side routing and navigation
- **Tailwind CSS 3.4.1**: Utility-first CSS framework
- **Heroicons**: Beautiful hand-crafted SVG icons

### Key Libraries
- **Axios 1.6.7**: HTTP client for API communication
- **React Markdown 9.0.1**: Markdown rendering with syntax highlighting
- **React Syntax Highlighter 15.5.0**: Code syntax highlighting
- **Headless UI 1.7.18**: Unstyled, accessible UI components
- **Tailwind Forms 0.5.7**: Form styling utilities

### Development Tools
- **React Scripts 5.0.1**: Create React App build tools
- **PostCSS 8.4.35**: CSS processing
- **Autoprefixer 10.4.17**: CSS vendor prefixing

## 🏗️ Architecture

### Project Structure

```
frontend/
├── public/                    # Static assets
│   ├── index.html            # HTML template
│   ├── manifest.json         # PWA configuration
│   └── favicon.svg          # Application icon
├── src/                      # Source code
│   ├── components/          # Reusable UI components
│   │   ├── Layout.js        # Main layout wrapper
│   │   ├── ChatMessage.js   # Chat message component
│   │   ├── ChatSessions.js  # Session management
│   │   └── ArxivCard.js     # Paper display card
│   ├── pages/               # Main application pages
│   │   ├── Home.js          # Landing page
│   │   ├── Chat.js          # Chat interface
│   │   ├── Search.js        # Paper search
│   │   └── DatasetInsights.js # Analytics dashboard
│   ├── context/             # React Context providers
│   │   └── ThemeContext.js  # Theme management
│   ├── config.js            # Configuration settings
│   ├── App.js               # Main app component
│   ├── index.js             # Application entry point
│   └── index.css            # Global styles
├── package.json             # Dependencies and scripts
├── tailwind.config.js       # Tailwind CSS configuration
├── postcss.config.js        # PostCSS configuration
└── Dockerfile               # Container configuration
```

### Component Architecture

#### Layout Component
- **Navigation**: Fixed top navigation with theme toggle
- **Responsive Design**: Mobile-first responsive layout
- **Dark/Light Theme**: System preference detection and manual toggle

#### Page Components
1. **Home**: Landing page with feature overview
2. **Chat**: Interactive AI chat interface with session management
3. **Search**: Advanced paper search with filters
4. **DatasetInsights**: Analytics and trend visualization

#### Shared Components
- **ChatMessage**: Message bubble with markdown rendering
- **ChatSessions**: Session list with management actions
- **ArxivCard**: Paper information display card

## 📋 Prerequisites

- Node.js 18+ (LTS recommended)
- npm or yarn package manager
- Running backend API (see backend README)

## 🛠️ Installation

### 1. Clone and Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Or using yarn
yarn install
```

### 2. Environment Configuration

Create a `.env` file in the frontend directory:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Optional: Enable development features
REACT_APP_DEV_MODE=true
```

### 3. Additional Dependencies

The application may require additional dependencies for enhanced features:

```bash
# Install additional packages (if not already in package.json)
npm install react-markdown remark-gfm react-syntax-highlighter chart.js react-chartjs-2
```

## 🚀 Running the Application

### Development Mode

```bash
# Start development server
npm start

# Or using yarn
yarn start
```

The application will be available at `http://localhost:3000`

### Production Build

```bash
# Create production build
npm run build

# Serve production build locally
npx serve -s build
```

### Docker Deployment

```bash
# Build Docker image
docker build -t research-assistant-frontend .

# Run container
docker run -p 3000:3000 research-assistant-frontend
```

## 🎨 Features

### 🏠 Home Page
- **Feature Overview**: Comprehensive introduction to platform capabilities
- **Navigation Cards**: Quick access to main features
- **Responsive Layout**: Optimized for all device sizes

### 💬 Chat Interface
- **AI-Powered Conversations**: Natural language research assistance
- **Session Management**: Persistent chat sessions with history
- **Markdown Support**: Rich text formatting in responses
- **Context Awareness**: Maintains conversation context across sessions

### 🔍 Search Functionality
- **Advanced Filtering**: Search by year, keywords, and categories
- **Paper Cards**: Detailed paper information display
- **Responsive Results**: Optimized viewing for search results
- **Quick Actions**: Easy access to paper details and actions

### 📊 Dataset Insights
- **Research Analytics**: Trends and patterns in academic papers
- **Interactive Visualizations**: Charts and graphs for data insights
- **Category Analysis**: Breakdown by research categories
- **Temporal Analysis**: Time-based research trends

### 🎨 Theme System
- **Dark/Light Mode**: System preference detection
- **Manual Toggle**: User-controlled theme switching
- **Persistent Settings**: Theme preference storage
- **Smooth Transitions**: Animated theme changes

## ⚙️ Configuration

### API Configuration

The application connects to the backend API through configuration in `src/config.js`:

```javascript
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
```

### Tailwind CSS Configuration

Tailwind is configured in `tailwind.config.js`:

```javascript
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### PWA Configuration

The application is configured as a Progressive Web App in `public/manifest.json`:

- **App Name**: Research Assistant
- **Theme Color**: #4F46E5 (Indigo)
- **Display Mode**: Standalone
- **Icons**: SVG favicon support

## 🧪 Development Guidelines

### Code Structure

```bash
# Component naming convention
ComponentName.js

# Page naming convention
PageName.js

# Context naming convention
ContextName.js
```

### Styling Guidelines

- Use Tailwind CSS utility classes
- Follow mobile-first responsive design
- Implement dark mode support for all components
- Use consistent spacing and typography scales

### State Management

- Use React Context for global state (theme, user preferences)
- Use local state for component-specific data
- Implement proper error boundaries for robust UX

## 📱 Responsive Design

### Breakpoints
- **Mobile**: Default (< 640px)
- **Tablet**: sm (640px+)
- **Desktop**: md (768px+)
- **Large Desktop**: lg (1024px+)

### Design Principles
- Mobile-first approach
- Touch-friendly interface elements
- Optimized navigation for all screen sizes
- Consistent spacing and typography

## 🔧 Available Scripts

```bash
# Development
npm start          # Start development server
npm run build      # Create production build
npm run test       # Run test suite
npm run eject      # Eject from Create React App

# Linting and formatting
npm run lint       # Run ESLint
npm run format     # Format code with Prettier
```

## 🐳 Docker Support

### Dockerfile Features
- **Multi-stage build**: Optimized for production
- **Node.js 18 Alpine**: Lightweight base image
- **Dependency caching**: Efficient builds
- **Port 3000**: Standard React development port

### Docker Commands

```bash
# Build image
docker build -t research-assistant-frontend .

# Run container
docker run -p 3000:3000 research-assistant-frontend

# Run with environment variables
docker run -p 3000:3000 \
  -e REACT_APP_API_URL=http://your-api-url \
  research-assistant-frontend
```

## 🔍 API Integration

### Axios Configuration

The frontend uses Axios for API communication:

```javascript
import axios from 'axios';
import { API_BASE_URL } from './config';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});
```

### Key API Endpoints

- `POST /api/chat` - Chat interactions
- `POST /api/search` - Paper search
- `GET /api/sessions` - Session management
- `GET /api/health` - Health check

## 🎯 Performance Optimization

### Built-in Optimizations
- **Code Splitting**: Automatic route-based splitting
- **Tree Shaking**: Removes unused code
- **Asset Optimization**: Minification and compression
- **Caching**: Browser caching strategies

### Best Practices
- Lazy loading for large components
- Memoization for expensive calculations
- Optimized re-renders with React.memo
- Efficient state updates

## 🚨 Troubleshooting

### Common Issues

1. **Build Errors**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **API Connection Issues**
   - Check `REACT_APP_API_URL` in `.env`
   - Verify backend is running on specified port
   - Check for CORS issues

3. **Styling Issues**
   - Ensure Tailwind CSS is properly configured
   - Check for conflicting CSS rules
   - Verify dark mode classes are applied

4. **Routing Problems**
   - Verify React Router configuration
   - Check for conflicting routes
   - Ensure proper navigation links

### Debug Mode

Enable debug mode for additional logging:

```env
REACT_APP_DEV_MODE=true
```

## 📊 Browser Support

### Supported Browsers
- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

### Features
- **ES6+ Support**: Modern JavaScript features
- **CSS Grid**: Advanced layout capabilities
- **Flexbox**: Flexible layouts
- **Progressive Web App**: PWA features

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards

- Follow ESLint configuration
- Use Prettier for code formatting
- Write meaningful commit messages
- Include comments for complex logic

## 🔧 Customization

### Theming

Customize the theme in `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: '#4F46E5',
      secondary: '#10B981',
    },
    fontFamily: {
      sans: ['Inter', 'sans-serif'],
    },
  },
}
```

### Component Customization

- Modify components in `src/components/`
- Update styling with Tailwind classes
- Add new features to existing components

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review the backend README for API issues
- Open an issue on GitHub
- Contact the development team

## 🚀 Future Enhancements

- Real-time collaborative features
- Advanced data visualization
- Offline support with service workers
- Enhanced accessibility features
- Mobile app version 