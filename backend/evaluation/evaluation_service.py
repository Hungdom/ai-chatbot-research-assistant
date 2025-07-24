import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from .optimized_metrics_calculator import OptimizedMetricsCalculator

class EvaluationService:
    """
    Service layer for chatbot evaluation metrics
    Provides caching and high-level interface for metrics calculation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = OptimizedMetricsCalculator()
        self._cache = {}
        self._cache_timeout = timedelta(minutes=30)  # Cache results for 30 minutes
        
    async def get_current_metrics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Get current chatbot metrics for the specified time range
        """
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Check cache first
            cache_key = f"metrics_{start_time.isoformat()}_{end_time.isoformat()}"
            if self._is_cache_valid(cache_key):
                self.logger.info("Returning cached metrics")
                return self._cache[cache_key]["data"]
            
            # Calculate fresh metrics
            self.logger.info(f"Calculating metrics for {time_range_hours}h period")
            metrics = await self.metrics_calculator.calculate_all_metrics(
                time_range=(start_time, end_time)
            )
            
            # Cache the results
            self._cache[cache_key] = {
                "data": metrics,
                "timestamp": datetime.utcnow()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a quick summary of key metrics
        """
        try:
            # Get metrics for last 24 hours
            full_metrics = await self.get_current_metrics(24)
            
            if "error" in full_metrics:
                return full_metrics
            
            # Extract key summary metrics
            summary = {
                "overview": {
                    "total_conversations": full_metrics.get("total_conversations", 0),
                    "analysis_period_hours": 24,
                    "last_updated": full_metrics.get("calculated_at")
                },
                "key_metrics": {
                    "response_relevancy": self._extract_metric_summary(
                        full_metrics.get("rag_metrics", {}).get("response_relevancy", {})
                    ),
                    "faithfulness": self._extract_metric_summary(
                        full_metrics.get("rag_metrics", {}).get("faithfulness", {})
                    ),
                    "context_precision": self._extract_metric_summary(
                        full_metrics.get("rag_metrics", {}).get("context_precision", {})
                    ),
                    "success_rate": full_metrics.get("performance_metrics", {}).get("success_rate", 0)
                },
                "performance": {
                    "avg_papers_per_query": full_metrics.get("performance_metrics", {}).get("average_papers_per_query", 0),
                    "avg_response_length": full_metrics.get("performance_metrics", {}).get("average_response_length", 0),
                    "total_sessions": full_metrics.get("conversation_metrics", {}).get("total_sessions", 0)
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def get_detailed_metrics(self, category: str = "all") -> Dict[str, Any]:
        """
        Get detailed metrics for a specific category
        """
        try:
            full_metrics = await self.get_current_metrics(24)
            
            if "error" in full_metrics:
                return full_metrics
            
            if category == "all":
                return full_metrics
            elif category == "rag":
                return {
                    "category": "RAG Metrics",
                    "description": "Retrieval Augmented Generation metrics",
                    "metrics": full_metrics.get("rag_metrics", {}),
                    "total_conversations": full_metrics.get("total_conversations", 0)
                }
            elif category == "performance":
                return {
                    "category": "Performance Metrics", 
                    "description": "System performance and efficiency metrics",
                    "metrics": full_metrics.get("performance_metrics", {}),
                    "total_conversations": full_metrics.get("total_conversations", 0)
                }
            elif category == "conversation":
                return {
                    "category": "Conversation Metrics",
                    "description": "User interaction and conversation flow metrics", 
                    "metrics": full_metrics.get("conversation_metrics", {}),
                    "total_conversations": full_metrics.get("total_conversations", 0)
                }
            else:
                return {"error": f"Unknown category: {category}. Use 'all', 'rag', 'performance', or 'conversation'"}
                
        except Exception as e:
            self.logger.error(f"Error getting detailed metrics for {category}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def get_historical_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get historical metrics trends over multiple days
        """
        try:
            historical_data = []
            
            # Calculate metrics for each day
            for day_offset in range(days):
                end_time = datetime.utcnow() - timedelta(days=day_offset)
                start_time = end_time - timedelta(days=1)
                
                day_metrics = await self.metrics_calculator.calculate_all_metrics(
                    time_range=(start_time, end_time)
                )
                
                if "error" not in day_metrics and day_metrics.get("total_conversations", 0) > 0:
                    historical_data.append({
                        "date": start_time.strftime("%Y-%m-%d"),
                        "conversations": day_metrics.get("total_conversations", 0),
                        "response_relevancy": day_metrics.get("rag_metrics", {}).get("response_relevancy", {}).get("average", 0),
                        "faithfulness": day_metrics.get("rag_metrics", {}).get("faithfulness", {}).get("average", 0),
                        "context_precision": day_metrics.get("rag_metrics", {}).get("context_precision", {}).get("average", 0),
                        "success_rate": day_metrics.get("performance_metrics", {}).get("success_rate", 0)
                    })
            
            return {
                "period_days": days,
                "data_points": len(historical_data),
                "historical_data": list(reversed(historical_data))  # Oldest first
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical metrics: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific session
        """
        try:
            metrics = await self.metrics_calculator.calculate_all_metrics(
                session_ids=[session_id]
            )
            
            return {
                "session_id": session_id,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session metrics for {session_id}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def get_metrics_definitions(self) -> Dict[str, Any]:
        """
        Get definitions and explanations of all metrics
        """
        return {
            "rag_metrics": {
                "context_precision": {
                    "name": "Context Precision",
                    "description": "Đo lường độ chính xác của context được truy xuất",
                    "formula": "Relevant Retrieved Contexts / Total Retrieved Contexts", 
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "Retrieval Quality"
                },
                "context_recall": {
                    "name": "Context Recall",
                    "description": "Đo lường khả năng thu hồi tất cả context liên quan",
                    "formula": "Retrieved Relevant Contexts / Total Relevant Contexts",
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "Retrieval Completeness"
                },
                "faithfulness": {
                    "name": "Faithfulness",
                    "description": "Đo lường mức độ trung thực của câu trả lời so với context",
                    "formula": "Faithful Statements / Total Statements",
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "Response Quality"
                },
                "response_relevancy": {
                    "name": "Response Relevancy", 
                    "description": "Đánh giá mức độ liên quan của câu trả lời với câu hỏi",
                    "formula": "Semantic similarity between query and response",
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "Response Quality"
                },
                "noise_sensitivity": {
                    "name": "Noise Sensitivity",
                    "description": "Đo lường mức độ ảnh hưởng của thông tin nhiễu",
                    "formula": "1.0 - (Low Relevance Papers / Total Papers)",
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "Robustness"
                }
            },
            "performance_metrics": {
                "success_rate": {
                    "name": "Success Rate",
                    "description": "Tỷ lệ truy vấn trả về kết quả thành công",
                    "formula": "Queries with Results / Total Queries", 
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "System Performance"
                },
                "average_papers_per_query": {
                    "name": "Average Papers per Query",
                    "description": "Số lượng papers trung bình được trả về mỗi truy vấn",
                    "formula": "Total Papers Retrieved / Total Queries",
                    "range": "0.0+ (tùy thuộc hệ thống)",
                    "category": "System Performance"
                },
                "average_response_length": {
                    "name": "Average Response Length",
                    "description": "Độ dài trung bình của câu trả lời",
                    "formula": "Total Response Characters / Total Responses",
                    "range": "0+ characters",
                    "category": "Response Characteristics"
                }
            },
            "conversation_metrics": {
                "topic_adherence": {
                    "name": "Topic Adherence", 
                    "description": "Đo lường khả năng giữ đúng chủ đề trong cuộc hội thoại",
                    "formula": "Average semantic similarity between consecutive queries",
                    "range": "0.0 - 1.0 (càng cao càng tốt)",
                    "category": "Conversation Flow"
                },
                "average_session_length": {
                    "name": "Average Session Length",
                    "description": "Số lượng tin nhắn trung bình mỗi session",
                    "formula": "Total Messages / Total Sessions", 
                    "range": "1.0+ messages",
                    "category": "User Engagement"
                }
            }
        }
    
    def _extract_metric_summary(self, metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract summary info from a metric data dict
        """
        if not metric_data:
            return {"average": 0, "count": 0}
        
        return {
            "average": round(metric_data.get("average", 0), 3),
            "count": metric_data.get("count", 0),
            "min": round(metric_data.get("min", 0), 3),
            "max": round(metric_data.get("max", 0), 3)
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid
        """
        if cache_key not in self._cache:
            return False
        
        cache_entry = self._cache[cache_key]
        cache_age = datetime.utcnow() - cache_entry["timestamp"]
        
        return cache_age < self._cache_timeout
    
    def clear_cache(self):
        """
        Clear all cached metrics data
        """
        self._cache.clear()
        # Also clear calculator's internal caches
        if hasattr(self.metrics_calculator, 'clear_cache'):
            self.metrics_calculator.clear_cache()
        self.logger.info("Metrics cache cleared")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API optimization
        """
        calculator_stats = {}
        if hasattr(self.metrics_calculator, '_embedding_cache'):
            calculator_stats = {
                "embedding_cache_size": len(self.metrics_calculator._embedding_cache),
                "evaluation_cache_size": len(self.metrics_calculator._evaluation_cache),
                "total_cached_items": len(self.metrics_calculator._embedding_cache) + len(self.metrics_calculator._evaluation_cache)
            }
        
        return {
            "service_cache_size": len(self._cache),
            "cache_timeout_minutes": self._cache_timeout.total_seconds() / 60,
            "calculator_optimization": calculator_stats,
            "optimization_enabled": hasattr(self.metrics_calculator, '_embedding_cache')
        } 