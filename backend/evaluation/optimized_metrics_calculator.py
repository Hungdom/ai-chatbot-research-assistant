import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from database import get_db, ChatSession, Arxiv
import openai
import re
from sklearn.metrics.pairwise import cosine_similarity
import statistics

class OptimizedMetricsCalculator:
    """
    Optimized version of MetricsCalculator that minimizes OpenAI API calls
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Cache for embeddings and evaluations
        self._embedding_cache = {}
        self._evaluation_cache = {}
        
    async def calculate_all_metrics(self, session_ids: Optional[List[str]] = None, 
                                  time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Calculate all available metrics with optimized API usage
        """
        try:
            with get_db() as db:
                # Get sessions to analyze
                sessions = self._get_sessions_for_analysis(db, session_ids, time_range)
                
                if not sessions:
                    return {"error": "No sessions found for analysis"}
                
                # Extract conversation data
                conversations = self._extract_conversations(sessions)
                
                if not conversations:
                    return {"error": "No valid conversations found"}
                
                # Calculate metrics with batching and caching
                rag_metrics = await self._calculate_rag_metrics_optimized(conversations, db)
                performance_metrics = self._calculate_performance_metrics(conversations)
                conversation_metrics = await self._calculate_conversation_metrics_optimized(conversations)
                
                return {
                    "total_conversations": len(conversations),
                    "analysis_period": {
                        "start": min(conv["timestamp"] for conv in conversations),
                        "end": max(conv["timestamp"] for conv in conversations)
                    },
                    "rag_metrics": rag_metrics,
                    "performance_metrics": performance_metrics,
                    "conversation_metrics": conversation_metrics,
                    "calculated_at": datetime.utcnow().isoformat(),
                    "optimization_stats": {
                        "embedding_cache_hits": len(self._embedding_cache),
                        "evaluation_cache_hits": len(self._evaluation_cache)
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def _calculate_rag_metrics_optimized(self, conversations: List[Dict[str, Any]], db: Session) -> Dict[str, Any]:
        """
        Calculate RAG metrics with optimized API usage
        """
        if not conversations:
            return {}
        
        # Pre-compute all embeddings in batch
        await self._precompute_embeddings_batch(conversations)
        
        context_precision_scores = []
        context_recall_scores = []
        faithfulness_scores = []
        response_relevancy_scores = []
        noise_sensitivity_scores = []
        
        for conv in conversations:
            try:
                # Context Precision: Use heuristics + selective LLM calls
                if conv["arxiv_papers"]:
                    precision = await self._calculate_context_precision_optimized(
                        conv["user_query"], conv["arxiv_papers"]
                    )
                    context_precision_scores.append(precision)
                
                # Context Recall: Use embedding similarity instead of LLM
                recall = self._calculate_context_recall_optimized(
                    conv["user_query"], conv["arxiv_papers"], db
                )
                context_recall_scores.append(recall)
                
                # Faithfulness: Use keyword matching + selective LLM
                faithfulness = await self._calculate_faithfulness_optimized(
                    conv["assistant_response"], conv["arxiv_papers"]
                )
                faithfulness_scores.append(faithfulness)
                
                # Response Relevancy: Use cached embeddings
                relevancy = self._calculate_response_relevancy_optimized(
                    conv["user_query"], conv["assistant_response"]
                )
                response_relevancy_scores.append(relevancy)
                
                # Noise Sensitivity: Pure heuristic approach
                noise_sensitivity = self._calculate_noise_sensitivity(conv)
                if noise_sensitivity is not None:
                    noise_sensitivity_scores.append(noise_sensitivity)
                
            except Exception as e:
                self.logger.warning(f"Error calculating RAG metrics for conversation: {str(e)}")
                continue
        
        return {
            "context_precision": self._create_metric_summary(context_precision_scores),
            "context_recall": self._create_metric_summary(context_recall_scores),
            "faithfulness": self._create_metric_summary(faithfulness_scores),
            "response_relevancy": self._create_metric_summary(response_relevancy_scores),
            "noise_sensitivity": self._create_metric_summary(noise_sensitivity_scores)
        }
    
    async def _precompute_embeddings_batch(self, conversations: List[Dict[str, Any]]):
        """
        Pre-compute all needed embeddings in efficient batches
        """
        texts_to_embed = set()
        
        # Collect all unique texts that need embedding
        for conv in conversations:
            texts_to_embed.add(conv["user_query"])
            texts_to_embed.add(conv["assistant_response"])
        
        # Remove already cached embeddings
        texts_to_embed = {text for text in texts_to_embed if text not in self._embedding_cache}
        
        if not texts_to_embed:
            return
        
        self.logger.info(f"Computing embeddings for {len(texts_to_embed)} unique texts")
        
        # Process in batches to avoid API rate limits
        batch_size = 20  # OpenAI allows up to 2048 texts per request
        text_list = list(texts_to_embed)
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            try:
                # Single API call for multiple texts
                response = await openai.Embedding.acreate(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                # Cache all embeddings from this batch
                for j, text in enumerate(batch):
                    self._embedding_cache[text] = response['data'][j]['embedding']
                    
            except Exception as e:
                self.logger.warning(f"Error computing embeddings for batch: {str(e)}")
                # Fallback: compute individually
                for text in batch:
                    try:
                        embedding = await self._get_embedding_single(text)
                        self._embedding_cache[text] = embedding
                    except Exception as individual_error:
                        self.logger.warning(f"Failed to get embedding for text: {individual_error}")
    
    async def _calculate_context_precision_optimized(self, query: str, papers: List[Dict[str, Any]]) -> float:
        """
        Calculate Context Precision with reduced API calls using heuristics first
        """
        if not papers:
            return 0.0
        
        relevant_count = 0
        
        for paper in papers:
            # Use heuristic first (keyword matching, similarity scores)
            heuristic_relevance = self._judge_paper_relevance_heuristic(query, paper)
            
            if heuristic_relevance > 0.7:  # High confidence - skip LLM
                relevant_count += 1
            elif heuristic_relevance < 0.3:  # Low confidence - skip LLM
                continue
            else:  # Medium confidence - use LLM for decisive cases only
                is_relevant = await self._judge_paper_relevance_llm_cached(query, paper)
                if is_relevant:
                    relevant_count += 1
        
        return relevant_count / len(papers)
    
    def _judge_paper_relevance_heuristic(self, query: str, paper: Dict[str, Any]) -> float:
        """
        Fast heuristic-based relevance scoring without API calls
        """
        query_words = set(query.lower().split())
        
        # Title matching (weighted heavily)
        title = paper.get('title', '').lower()
        title_words = set(title.split())
        title_overlap = len(query_words.intersection(title_words)) / max(len(query_words), 1)
        
        # Abstract matching  
        abstract = paper.get('abstract', '').lower()
        abstract_words = set(abstract.split())
        abstract_overlap = len(query_words.intersection(abstract_words)) / max(len(query_words), 1)
        
        # Use existing similarity score if available
        similarity_score = paper.get('similarity_score', 0.5)
        
        # Combine heuristics
        heuristic_score = (
            title_overlap * 0.4 + 
            abstract_overlap * 0.3 + 
            similarity_score * 0.3
        )
        
        return min(heuristic_score, 1.0)
    
    async def _judge_paper_relevance_llm_cached(self, query: str, paper: Dict[str, Any]) -> bool:
        """
        LLM-based relevance judgment with caching
        """
        cache_key = f"relevance_{hash(query)}_{hash(paper.get('title', ''))}"
        
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]
        
        try:
            paper_info = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:300]}"
            
            prompt = f"""
            Query: {query}
            
            Paper:
            {paper_info}
            
            Is this paper relevant to the query? Answer only 'yes' or 'no'.
            """
            
            completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content.lower().strip()
            result = "yes" in response
            
            # Cache the result
            self._evaluation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.warning(f"Error judging paper relevance: {str(e)}")
            return True  # Default to relevant if error
    
    def _calculate_context_recall_optimized(self, query: str, retrieved_papers: List[Dict[str, Any]], 
                                          db: Session) -> float:
        """
        Calculate Context Recall using embedding similarity instead of LLM calls
        """
        try:
            if not retrieved_papers:
                return 0.0
            
            # Get query embedding from cache
            query_embedding = self._embedding_cache.get(query)
            if not query_embedding:
                return 0.5  # Default score if no embedding
            
            # Simple approximation: compare with a sample of potentially relevant papers
            broader_papers = self._get_broader_paper_set(query, db, limit=20)  # Reduced limit
            
            if not broader_papers:
                return 1.0 if retrieved_papers else 0.0
            
            # Use title similarity as proxy for relevance
            retrieved_titles = {paper.get("title", "") for paper in retrieved_papers}
            relevant_retrieved = sum(1 for paper in broader_papers 
                                   if any(title in paper.title for title in retrieved_titles))
            
            return min(relevant_retrieved / len(broader_papers), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating context recall: {str(e)}")
            return 0.5
    
    async def _calculate_faithfulness_optimized(self, response: str, papers: List[Dict[str, Any]]) -> float:
        """
        Calculate Faithfulness using keyword matching + selective LLM verification
        """
        if not response or not papers:
            return 0.5
        
        # Extract claims using simple heuristics
        claims = self._extract_claims_simple(response)
        if not claims:
            return 1.0
        
        faithful_claims = 0
        
        for claim in claims:
            # Use keyword matching first
            keyword_faithfulness = self._verify_claim_keyword_matching(claim, papers)
            
            if keyword_faithfulness > 0.8:  # High confidence
                faithful_claims += 1
            elif keyword_faithfulness < 0.2:  # Low confidence
                continue
            else:  # Use LLM only for uncertain cases
                is_faithful = await self._verify_claim_faithfulness_llm_cached(claim, papers)
                if is_faithful:
                    faithful_claims += 1
        
        return faithful_claims / len(claims)
    
    def _extract_claims_simple(self, response: str) -> List[str]:
        """
        Simple claim extraction without complex NLP
        """
        # Split by sentences and filter
        sentences = re.split(r'[.!?]+', response)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 15 and 
                not sentence.startswith(('How', 'What', 'Why', 'Where', 'When')) and
                not sentence.lower().startswith(('hello', 'hi', 'thank', 'please', 'let me'))):
                claims.append(sentence)
        
        return claims[:5]  # Limit to 5 claims to reduce API calls
    
    def _verify_claim_keyword_matching(self, claim: str, papers: List[Dict[str, Any]]) -> float:
        """
        Verify claim using keyword matching against paper abstracts
        """
        claim_words = set(claim.lower().split())
        
        # Create context from papers
        paper_texts = []
        for paper in papers[:3]:  # Limit to first 3 papers
            text = f"{paper.get('title', '')} {paper.get('abstract', '')[:200]}"
            paper_texts.append(text.lower())
        
        full_context = " ".join(paper_texts)
        context_words = set(full_context.split())
        
        # Calculate overlap
        overlap = len(claim_words.intersection(context_words)) / max(len(claim_words), 1)
        return min(overlap * 2, 1.0)  # Boost the score
    
    async def _verify_claim_faithfulness_llm_cached(self, claim: str, papers: List[Dict[str, Any]]) -> bool:
        """
        LLM-based faithfulness verification with caching
        """
        cache_key = f"faithful_{hash(claim)}_{hash(str([p.get('title', '') for p in papers[:2]]))}"
        
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]
        
        try:
            # Limit context size
            context = "\n\n".join([
                f"Paper: {paper.get('title', '')}\n{paper.get('abstract', '')[:200]}"
                for paper in papers[:2]  # Only use first 2 papers
            ])
            
            prompt = f"""
            Context from research papers:
            {context}
            
            Claim: {claim}
            
            Is this claim supported by the provided context? Answer only 'yes' or 'no'.
            """
            
            completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content.lower().strip()
            result = "yes" in response
            
            # Cache the result
            self._evaluation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.warning(f"Error verifying claim faithfulness: {str(e)}")
            return True
    
    def _calculate_response_relevancy_optimized(self, query: str, response: str) -> float:
        """
        Calculate Response Relevancy using cached embeddings
        """
        try:
            # Get embeddings from cache
            query_embedding = self._embedding_cache.get(query)
            response_embedding = self._embedding_cache.get(response)
            
            if not query_embedding or not response_embedding:
                return 0.5  # Default score if embeddings not available
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
            
            # Normalize to 0-1 range
            return (similarity + 1) / 2
            
        except Exception as e:
            self.logger.warning(f"Error calculating response relevancy: {str(e)}")
            return 0.5
    
    async def _calculate_conversation_metrics_optimized(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate conversation metrics with optimized API usage
        """
        # Group by session
        sessions = {}
        for conv in conversations:
            session_id = conv["session_id"]
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(conv)
        
        session_lengths = [len(convs) for convs in sessions.values()]
        
        # Topic adherence using cached embeddings
        topic_adherence_scores = []
        for session_convs in sessions.values():
            if len(session_convs) > 1:
                adherence = self._calculate_topic_adherence_optimized(session_convs)
                topic_adherence_scores.append(adherence)
        
        return {
            "total_sessions": len(sessions),
            "average_session_length": np.mean(session_lengths) if session_lengths else 0,
            "topic_adherence": {
                "average": np.mean(topic_adherence_scores) if topic_adherence_scores else 1.0,
                "count": len(topic_adherence_scores)
            },
            "session_stats": {
                "min_length": np.min(session_lengths) if session_lengths else 0,
                "max_length": np.max(session_lengths) if session_lengths else 0,
                "median_length": np.median(session_lengths) if session_lengths else 0
            }
        }
    
    def _calculate_topic_adherence_optimized(self, session_conversations: List[Dict[str, Any]]) -> float:
        """
        Calculate topic adherence using cached embeddings
        """
        if len(session_conversations) < 2:
            return 1.0
        
        try:
            # Get embeddings from cache
            query_embeddings = []
            for conv in session_conversations:
                embedding = self._embedding_cache.get(conv["user_query"])
                if embedding:
                    query_embeddings.append(embedding)
            
            if len(query_embeddings) < 2:
                return 0.8  # Default reasonable score
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(query_embeddings) - 1):
                sim = cosine_similarity([query_embeddings[i]], [query_embeddings[i + 1]])[0][0]
                similarities.append((sim + 1) / 2)  # Normalize to 0-1
            
            return np.mean(similarities) if similarities else 1.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating topic adherence: {str(e)}")
            return 0.8
    
    # Helper methods (reused from original)
    def _get_sessions_for_analysis(self, db: Session, session_ids: Optional[List[str]], 
                                 time_range: Optional[Tuple[datetime, datetime]]) -> List[ChatSession]:
        """Same as original implementation"""
        query = db.query(ChatSession)
        
        if session_ids:
            query = query.filter(ChatSession.session_id.in_(session_ids))
        
        if time_range:
            start_time, end_time = time_range
            query = query.filter(
                and_(
                    ChatSession.updated_at >= start_time,
                    ChatSession.updated_at <= end_time
                )
            )
        else:
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.filter(ChatSession.updated_at >= week_ago)
        
        return query.order_by(desc(ChatSession.updated_at)).all()
    
    def _extract_conversations(self, sessions: List[ChatSession]) -> List[Dict[str, Any]]:
        """Same as original implementation"""
        conversations = []
        
        for session in sessions:
            messages = session.messages or []
            context = session.context or {}
            
            for i in range(0, len(messages) - 1, 2):
                if (i + 1 < len(messages) and 
                    messages[i].get("role") == "user" and 
                    messages[i + 1].get("role") == "assistant"):
                    
                    user_msg = messages[i]
                    assistant_msg = messages[i + 1]
                    
                    conversations.append({
                        "session_id": session.session_id,
                        "user_query": user_msg.get("content", ""),
                        "assistant_response": assistant_msg.get("content", ""),
                        "html_response": assistant_msg.get("html_content", ""),
                        "timestamp": user_msg.get("timestamp", session.updated_at.isoformat()),
                        "context": context,
                        "arxiv_papers": context.get("arxiv", []),
                        "intent": context.get("intent", {})
                    })
        
        return conversations
    
    def _calculate_performance_metrics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Same as original implementation"""
        paper_counts = []
        response_lengths = []
        queries_with_results = 0
        
        for conv in conversations:
            paper_count = len(conv.get("arxiv_papers", []))
            paper_counts.append(paper_count)
            
            if paper_count > 0:
                queries_with_results += 1
            
            response_length = len(conv.get("assistant_response", ""))
            response_lengths.append(response_length)
        
        return {
            "average_papers_per_query": np.mean(paper_counts) if paper_counts else 0,
            "success_rate": queries_with_results / len(conversations) if conversations else 0,
            "average_response_length": np.mean(response_lengths) if response_lengths else 0,
            "paper_retrieval_stats": {
                "min": np.min(paper_counts) if paper_counts else 0,
                "max": np.max(paper_counts) if paper_counts else 0,
                "median": np.median(paper_counts) if paper_counts else 0
            }
        }
    
    def _calculate_noise_sensitivity(self, conversation: Dict[str, Any]) -> Optional[float]:
        """Same as original implementation"""
        papers = conversation.get("arxiv_papers", [])
        if len(papers) < 3:
            return None
        
        low_relevance_count = 0
        for paper in papers:
            similarity_score = paper.get("similarity_score", 0.5)
            if similarity_score < 0.3:
                low_relevance_count += 1
        
        noise_ratio = low_relevance_count / len(papers)
        return 1.0 - noise_ratio
    
    def _get_broader_paper_set(self, query: str, db: Session, limit: int = 20) -> List[Arxiv]:
        """Reduced limit version"""
        try:
            keywords = query.lower().split()
            
            query_obj = db.query(Arxiv)
            
            conditions = []
            for keyword in keywords[:3]:  # Limit to first 3 keywords
                conditions.extend([
                    Arxiv.title.ilike(f"%{keyword}%"),
                    Arxiv.abstract.ilike(f"%{keyword}%")
                ])
            
            if conditions:
                query_obj = query_obj.filter(or_(*conditions))
            
            return query_obj.order_by(desc(Arxiv.published_date)).limit(limit).all()
            
        except Exception as e:
            self.logger.warning(f"Error getting broader paper set: {str(e)}")
            return []
    
    def _create_metric_summary(self, scores: List[float]) -> Dict[str, Any]:
        """Create summary statistics for a metric"""
        if not scores:
            return {"average": 0, "min": 0, "max": 0, "count": 0}
        
        return {
            "average": np.mean(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "count": len(scores)
        }
    
    async def _get_embedding_single(self, text: str) -> List[float]:
        """Get embedding for single text"""
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    
    def clear_cache(self):
        """Clear all caches"""
        self._embedding_cache.clear()
        self._evaluation_cache.clear()
        self.logger.info("All caches cleared") 