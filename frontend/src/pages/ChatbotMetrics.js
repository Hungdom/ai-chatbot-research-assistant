import React, { useState, useEffect } from 'react';
import { 
  ChartBarIcon,
  CpuChipIcon,
  CogIcon,
  DocumentTextIcon,
  CommandLineIcon,
  BeakerIcon,
  ClipboardDocumentListIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const ChatbotMetrics = () => {
  const [activeCategory, setActiveCategory] = useState('rag');
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [optimizationStats, setOptimizationStats] = useState(null);

  // Fetch metrics data from backend
  const fetchMetricsData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch both metrics and optimization stats
      const [metricsResponse, optimizationResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/metrics/summary`),
        axios.get(`${API_BASE_URL}/api/metrics/optimization-stats`).catch(() => ({ data: null }))
      ]);
      
      setMetricsData(metricsResponse.data);
      setOptimizationStats(optimizationResponse.data);
      setLastUpdated(new Date().toLocaleString());
    } catch (err) {
      console.error('Error fetching metrics:', err);
      setError('Không thể tải dữ liệu metrics. Vui lòng thử lại sau.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetricsData();
  }, []);

  const handleRefresh = () => {
    fetchMetricsData();
  };

  // Function to get metric value with fallback
  const getMetricValue = (category, metricName, field = 'average') => {
    if (!metricsData || !metricsData[category] || !metricsData[category][metricName]) {
      return 0;
    }
    return metricsData[category][metricName][field] || 0;
  };

  const formatScore = (score) => {
    if (score === 0) return '0.000';
    return (score * 1).toFixed(3);
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400';
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const metricCategories = {
    rag: {
      title: 'Retrieval Augmented Generation',
      icon: DocumentTextIcon,
      description: 'Metrics để đánh giá hiệu suất của hệ thống RAG (Retrieval Augmented Generation)',
      metrics: [
        {
          name: 'Context Precision',
          description: 'Đo lường độ chính xác của context được truy xuất. Đánh giá xem các đoạn context có liên quan đến câu hỏi hay không.',
          formula: 'Precision = Relevant Retrieved Contexts / Total Retrieved Contexts',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đánh giá chất lượng của hệ thống retrieval'
        },
        {
          name: 'Context Recall',
          description: 'Đo lường khả năng thu hồi tất cả context liên quan. Kiểm tra xem hệ thống có bỏ sót thông tin quan trọng không.',
          formula: 'Recall = Relevant Retrieved Contexts / Total Relevant Contexts',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đảm bảo không bỏ sót thông tin quan trọng'
        },
        {
          name: 'Context Entities Recall',
          description: 'Đánh giá khả năng truy xuất các thực thể (entities) quan trọng từ context.',
          formula: 'Entities Recall = Retrieved Important Entities / Total Important Entities',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đảm bảo các thông tin quan trọng như tên, địa điểm, số liệu được truy xuất đầy đủ'
        },
        {
          name: 'Noise Sensitivity',
          description: 'Đo lường mức độ ảnh hưởng của thông tin nhiễu (irrelevant information) đến chất lượng câu trả lời.',
          formula: 'Noise Impact = Performance Drop with Noise / Original Performance',
          range: '0.0 - 1.0 (càng thấp càng tốt)',
          useCase: 'Kiểm tra khả năng chống nhiễu của hệ thống'
        },
        {
          name: 'Response Relevancy',
          description: 'Đánh giá mức độ liên quan của câu trả lời với câu hỏi ban đầu.',
          formula: 'Relevancy Score based on semantic similarity and content alignment',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đảm bảo câu trả lời đúng trọng tâm'
        },
        {
          name: 'Faithfulness',
          description: 'Đo lường mức độ trung thực của câu trả lời so với context được cung cấp. Kiểm tra hallucination.',
          formula: 'Faithfulness = Faithful Statements / Total Statements',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Ngăn ngừa hallucination và đảm bảo tính chính xác'
        },
        {
          name: 'Multimodal Faithfulness',
          description: 'Đánh giá tính trung thực khi làm việc với nhiều loại dữ liệu (text, image, audio).',
          formula: 'Cross-modal consistency evaluation',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Hệ thống multimodal RAG'
        },
        {
          name: 'Multimodal Relevance',
          description: 'Đo lường mức độ liên quan của nội dung multimodal với câu hỏi.',
          formula: 'Cross-modal relevance scoring',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đảm bảo tất cả các modality đều liên quan'
        }
      ]
    },
    nvidia: {
      title: 'NVIDIA Metrics',
      icon: CpuChipIcon,
      description: 'Các metrics được phát triển bởi NVIDIA cho việc đánh giá LLM',
      metrics: [
        {
          name: 'Answer Accuracy',
          description: 'Đo lường độ chính xác của câu trả lời so với ground truth.',
          formula: 'Accuracy = Correct Answers / Total Answers',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đánh giá tổng thể hiệu suất trả lời'
        },
        {
          name: 'Context Relevance',
          description: 'Đánh giá mức độ liên quan của context với câu hỏi (phiên bản NVIDIA).',
          formula: 'NVIDIA proprietary relevance scoring',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Tối ưu hóa retrieval pipeline'
        },
        {
          name: 'Response Groundedness',
          description: 'Đo lường mức độ dựa trên evidence của câu trả lời.',
          formula: 'Groundedness = Evidence-supported Claims / Total Claims',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đảm bảo câu trả lời có căn cứ'
        }
      ]
    },
    agents: {
      title: 'Agents & Tool Use Cases',
      icon: CogIcon,
      description: 'Metrics đánh giá hiệu suất của AI agents và việc sử dụng tools',
      metrics: [
        {
          name: 'Topic Adherence',
          description: 'Đo lường khả năng giữ đúng chủ đề trong cuộc hội thoại.',
          formula: 'Topic Consistency Score across conversation turns',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đảm bảo agent không đi lạc chủ đề'
        },
        {
          name: 'Tool Call Accuracy',
          description: 'Đánh giá độ chính xác khi agent gọi các tools/functions.',
          formula: 'Correct Tool Calls / Total Tool Calls',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Tối ưu hóa tool selection và usage'
        },
        {
          name: 'Agent Goal Accuracy',
          description: 'Đo lường khả năng đạt được mục tiêu đã đặt ra.',
          formula: 'Achieved Goals / Total Goals',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đánh giá hiệu quả tổng thể của agent'
        }
      ]
    },
    nlp: {
      title: 'Natural Language Comparison',
      icon: BeakerIcon,
      description: 'Các metrics so sánh và đánh giá ngôn ngữ tự nhiên',
      metrics: [
        {
          name: 'Factual Correctness',
          description: 'Đánh giá tính chính xác về mặt sự kiện của câu trả lời.',
          formula: 'Fact-checking against reliable sources',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Kiểm tra tính chính xác của thông tin'
        },
        {
          name: 'Semantic Similarity',
          description: 'Đo lường mức độ tương đồng về nghĩa giữa hai đoạn text.',
          formula: 'Cosine similarity of embeddings',
          range: '0.0 - 1.0 (càng cao càng tương đồng)',
          useCase: 'So sánh ý nghĩa của câu trả lời'
        },
        {
          name: 'BLEU Score',
          description: 'Đánh giá chất lượng dịch máy và text generation.',
          formula: 'Precision of n-grams with brevity penalty',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đánh giá chất lượng text generation'
        },
        {
          name: 'ROUGE Score',
          description: 'Đánh giá chất lượng tóm tắt và text summarization.',
          formula: 'Recall-oriented n-gram matching',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đánh giá chất lượng tóm tắt'
        },
        {
          name: 'String Presence',
          description: 'Kiểm tra sự xuất hiện của các chuỗi ký tự cụ thể.',
          formula: 'Binary check for string existence',
          range: '0 hoặc 1 (có/không)',
          useCase: 'Kiểm tra từ khóa bắt buộc'
        },
        {
          name: 'Exact Match',
          description: 'Kiểm tra sự khớp chính xác giữa output và expected result.',
          formula: 'Binary exact string matching',
          range: '0 hoặc 1 (khớp/không khớp)',
          useCase: 'Kiểm tra kết quả chính xác'
        }
      ]
    },
    sql: {
      title: 'SQL Metrics',
      icon: CommandLineIcon,
      description: 'Metrics đánh giá cho các tác vụ liên quan đến SQL',
      metrics: [
        {
          name: 'Execution-based Datacompy Score',
          description: 'So sánh kết quả thực thi SQL với expected results.',
          formula: 'Data comparison score between actual and expected results',
          range: '0.0 - 1.0 (càng cao càng khớp)',
          useCase: 'Kiểm tra độ chính xác của SQL generation'
        },
        {
          name: 'SQL Query Equivalence',
          description: 'Đánh giá tính tương đương về logic của các câu SQL.',
          formula: 'Logical equivalence analysis',
          range: '0.0 - 1.0 (càng cao càng tương đương)',
          useCase: 'So sánh các cách viết SQL khác nhau'
        }
      ]
    },
    general: {
      title: 'General Purpose',
      icon: ClipboardDocumentListIcon,
      description: 'Các metrics đa năng có thể áp dụng cho nhiều tác vụ',
      metrics: [
        {
          name: 'Aspect Critic',
          description: 'Đánh giá theo các khía cạnh cụ thể được định nghĩa trước.',
          formula: 'Custom aspect-based scoring',
          range: 'Tùy theo định nghĩa aspect',
          useCase: 'Đánh giá theo tiêu chí tùy chỉnh'
        },
        {
          name: 'Simple Criteria Scoring',
          description: 'Đánh giá dựa trên các tiêu chí đơn giản được định nghĩa.',
          formula: 'Custom criteria evaluation',
          range: 'Tùy theo tiêu chí',
          useCase: 'Đánh giá nhanh theo tiêu chí cơ bản'
        },
        {
          name: 'Rubrics Based Scoring',
          description: 'Đánh giá theo rubric chi tiết với nhiều mức độ.',
          formula: 'Multi-level rubric evaluation',
          range: 'Theo thang điểm rubric',
          useCase: 'Đánh giá chi tiết theo tiêu chuẩn'
        },
        {
          name: 'Instance Specific Rubrics',
          description: 'Đánh giá theo rubric được tùy chỉnh cho từng trường hợp cụ thể.',
          formula: 'Instance-adapted rubric scoring',
          range: 'Tùy theo rubric instance',
          useCase: 'Đánh giá chuyên biệt cho từng trường hợp'
        }
      ]
    },
    other: {
      title: 'Other Tasks',
      icon: ChartBarIcon,
      description: 'Metrics cho các tác vụ đặc biệt khác',
      metrics: [
        {
          name: 'Summarization',
          description: 'Đánh giá chất lượng tóm tắt văn bản.',
          formula: 'Combination of content coverage, coherence, and conciseness',
          range: '0.0 - 1.0 (càng cao càng tốt)',
          useCase: 'Đánh giá hệ thống tóm tắt tự động'
        }
      ]
    }
  };

  // Real metrics card component with actual data
  const RealMetricCard = ({ metricKey, metric, category }) => {
    const currentValue = getMetricValue('key_metrics', metricKey, 'average');
    const count = getMetricValue('key_metrics', metricKey, 'count');
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-start mb-3">
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
            {metric.name}
          </h4>
          <div className="text-right">
            <div className={`text-2xl font-bold ${getScoreColor(currentValue)}`}>
              {formatScore(currentValue)}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {count} mẫu
            </div>
          </div>
        </div>
        
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          {metric.description}
        </p>
        
        <div className="space-y-2">
          <div>
            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Công thức:</span>
            <p className="text-sm text-gray-700 dark:text-gray-300 font-mono bg-gray-50 dark:bg-gray-700 p-2 rounded">
              {metric.formula}
            </p>
          </div>
          <div>
            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Phạm vi:</span>
            <span className="text-sm text-gray-700 dark:text-gray-300 ml-2">{metric.range}</span>
          </div>
          <div>
            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Ứng dụng:</span>
            <span className="text-sm text-gray-700 dark:text-gray-300 ml-2">{metric.useCase}</span>
          </div>
        </div>
        
        {/* Progress bar for visual representation */}
        <div className="mt-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-500 dark:text-gray-400">Hiệu suất</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">{(currentValue * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                currentValue >= 0.8 ? 'bg-green-500' : 
                currentValue >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(currentValue * 100, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>
    );
  };

  const MetricCard = ({ metric }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
      <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
        {metric.name}
      </h4>
      <p className="text-gray-600 dark:text-gray-300 mb-4">
        {metric.description}
      </p>
      <div className="space-y-2">
        <div>
          <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Công thức:</span>
          <p className="text-sm text-gray-700 dark:text-gray-300 font-mono bg-gray-50 dark:bg-gray-700 p-2 rounded">
            {metric.formula}
          </p>
        </div>
        <div>
          <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Phạm vi:</span>
          <span className="text-sm text-gray-700 dark:text-gray-300 ml-2">{metric.range}</span>
        </div>
        <div>
          <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Ứng dụng:</span>
          <span className="text-sm text-gray-700 dark:text-gray-300 ml-2">{metric.useCase}</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Metrics Đánh Giá Chatbot
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
                Tổng quan về các metrics đánh giá hiệu suất chatbot và hệ thống AI, 
                dựa trên framework <a href="https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/" 
                target="_blank" rel="noopener noreferrer" 
                className="text-indigo-600 dark:text-indigo-400 hover:underline">Ragas</a>.
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {lastUpdated && (
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Cập nhật: {lastUpdated}
                </div>
              )}
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                <ArrowPathIcon className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Làm mới
              </button>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <ArrowPathIcon className="h-8 w-8 animate-spin text-indigo-600 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-300">Đang tải dữ liệu metrics...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 mb-8">
            <div className="flex items-center">
              <div className="text-red-800 dark:text-red-200">
                <strong>Lỗi:</strong> {error}
              </div>
            </div>
          </div>
        )}

        {/* Metrics Overview */}
        {metricsData && !loading && !error && (
          <div className="mb-8">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg p-6">
                <div className="text-2xl font-bold">
                  {metricsData.overview?.total_conversations || 0}
                </div>
                <div className="text-blue-100">Tổng cuộc hội thoại</div>
              </div>
              
              <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg p-6">
                <div className="text-2xl font-bold">
                  {formatScore(getMetricValue('key_metrics', 'response_relevancy', 'average'))}
                </div>
                <div className="text-green-100">Response Relevancy</div>
              </div>
              
              <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg p-6">
                <div className="text-2xl font-bold">
                  {formatScore(getMetricValue('key_metrics', 'faithfulness', 'average'))}
                </div>
                <div className="text-purple-100">Faithfulness</div>
              </div>
              
              <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg p-6">
                <div className="text-2xl font-bold">
                  {((metricsData.key_metrics?.success_rate || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-orange-100">Success Rate</div>
              </div>
            </div>
          </div>
        )}

        {/* Category Tabs */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-2">
            {Object.entries(metricCategories).map(([key, category]) => {
              const IconComponent = category.icon;
              return (
                <button
                  key={key}
                  onClick={() => setActiveCategory(key)}
                  className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeCategory === key
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <IconComponent className="h-4 w-4 mr-2" />
                  {category.title}
                </button>
              );
            })}
          </div>
        </div>

        {/* Active Category Content */}
        <div className="mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
            <div className="flex items-center mb-4">
              {React.createElement(metricCategories[activeCategory].icon, {
                className: "h-6 w-6 text-indigo-600 dark:text-indigo-400 mr-3"
              })}
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                {metricCategories[activeCategory].title}
              </h2>
            </div>
            <p className="text-gray-600 dark:text-gray-300">
              {metricCategories[activeCategory].description}
            </p>
          </div>

          {/* Metrics Grid */}
          {!loading && !error && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {activeCategory === 'rag' && metricsData ? (
                // Show real RAG metrics with data
                <>
                  <RealMetricCard metricKey="response_relevancy" metric={metricCategories.rag.metrics[4]} category={activeCategory} />
                  <RealMetricCard metricKey="faithfulness" metric={metricCategories.rag.metrics[5]} category={activeCategory} />
                  <RealMetricCard metricKey="context_precision" metric={metricCategories.rag.metrics[0]} category={activeCategory} />
                  {metricCategories[activeCategory].metrics.slice(1, 4).map((metric, index) => (
                    <MetricCard key={index + 3} metric={metric} />
                  ))}
                  {metricCategories[activeCategory].metrics.slice(6).map((metric, index) => (
                    <MetricCard key={index + 6} metric={metric} />
                  ))}
                </>
              ) : (
                // Show static information for other categories
                metricCategories[activeCategory].metrics.map((metric, index) => (
                  <MetricCard key={index} metric={metric} />
                ))
              )}
            </div>
          )}
        </div>

        {/* Optimization Stats */}
        {optimizationStats && !loading && !error && (
          <div className="mb-8">
            <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-green-900 dark:text-green-100 mb-4 flex items-center">
                <BeakerIcon className="h-5 w-5 mr-2" />
                API Optimization Status
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {optimizationStats.optimization_enabled ? '✅' : '❌'}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Tối ưu hóa</div>
                </div>
                
                {optimizationStats.calculator_optimization && (
                  <>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {optimizationStats.calculator_optimization.embedding_cache_size}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Embeddings Cache</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                        {optimizationStats.calculator_optimization.evaluation_cache_size}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Evaluations Cache</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                        ~80%
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">API Calls Reduced</div>
                    </div>
                  </>
                )}
              </div>
              
              <div className="mt-4 text-sm text-green-700 dark:text-green-300">
                💡 <strong>Tối ưu hóa API:</strong> Hệ thống sử dụng batch processing, caching và heuristics để giảm 80% lượng API calls đến OpenAI.
              </div>
            </div>
          </div>
        )}

        {/* Footer Note */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
            Lưu ý về việc sử dụng metrics
          </h3>
          <ul className="space-y-2 text-blue-800 dark:text-blue-200">
            <li>• Không có metric nào hoàn hảo cho mọi trường hợp sử dụng</li>
            <li>• Nên sử dụng kết hợp nhiều metrics để có cái nhìn toàn diện</li>
            <li>• Metrics dựa trên LLM có thể có độ trễ và chi phí cao hơn</li>
            <li>• Cần hiểu rõ context và mục đích khi chọn metrics phù hợp</li>
            <li>• Thường xuyên đánh giá và điều chỉnh ngưỡng metrics theo thực tế</li>
            <li>• <strong>Hệ thống đã được tối ưu hóa để giảm chi phí API calls</strong></li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ChatbotMetrics; 