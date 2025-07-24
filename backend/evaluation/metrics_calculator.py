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

class MetricsCalculator:
    """
    Calculate various metrics for chatbot evaluation based on Ragas framework
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def calculate_all_metrics(self, session_ids: Optional[List[str]] = None, 
                                  time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Calculate all available metrics for given sessions or time range
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
                
                # Calculate different categories of metrics
                rag_metrics = await self._calculate_rag_metrics(conversations, db)
                performance_metrics = self._calculate_performance_metrics(conversations)
                conversation_metrics = await self._calculate_conversation_metrics(conversations)
                
                return {
                    "total_conversations": len(conversations),
                    "analysis_period": {
                        "start": min(conv["timestamp"] for conv in conversations),
                        "end": max(conv["timestamp"] for conv in conversations)
                    },
                    "rag_metrics": rag_metrics,
                    "performance_metrics": performance_metrics,
                    "conversation_metrics": conversation_metrics,
                    "calculated_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _get_sessions_for_analysis(self, db: Session, session_ids: Optional[List[str]], 
                                 time_range: Optional[Tuple[datetime, datetime]]) -> List[ChatSession]:
        """
        Get chat sessions for analysis based on filters
        """
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
            # Default to last 7 days if no time range specified
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.filter(ChatSession.updated_at >= week_ago)
        
        return query.order_by(desc(ChatSession.updated_at)).all()
    
    def _extract_conversations(self, sessions: List[ChatSession]) -> List[Dict[str, Any]]:
        """
        Extract conversation pairs (user query + assistant response) from sessions
        """
        conversations = []
        
        for session in sessions:
            messages = session.messages or []
            context = session.context or {}
            
            # Process message pairs
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
    
    async def _calculate_rag_metrics(self, conversations: List[Dict[str, Any]], db: Session) -> Dict[str, Any]:
        """
        Calculate RAG-specific metrics (Context Precision, Recall, Faithfulness, Response Relevancy)
        """
        if not conversations:
            return {}
        
        context_precision_scores = []
        context_recall_scores = []
        faithfulness_scores = []
        response_relevancy_scores = []
        noise_sensitivity_scores = []
        
        for conv in conversations:
            try:
                # Context Precision: How relevant are the retrieved papers?
                if conv["arxiv_papers"]:
                    precision = await self._calculate_context_precision(
                        conv["user_query"], conv["arxiv_papers"]
                    )
                    context_precision_scores.append(precision)
                
                # Context Recall: Are we retrieving all relevant information?
                recall = await self._calculate_context_recall(
                    conv["user_query"], conv["arxiv_papers"], db
                )
                context_recall_scores.append(recall)
                
                # Faithfulness: Is the response faithful to the context?
                faithfulness = await self._calculate_faithfulness(
                    conv["assistant_response"], conv["arxiv_papers"]
                )
                faithfulness_scores.append(faithfulness)
                
                # Response Relevancy: Is the response relevant to the query?
                relevancy = await self._calculate_response_relevancy(
                    conv["user_query"], conv["assistant_response"]
                )
                response_relevancy_scores.append(relevancy)
                
                # Noise Sensitivity: How well does the system handle irrelevant context?
                noise_sensitivity = self._calculate_noise_sensitivity(conv)
                if noise_sensitivity is not None:
                    noise_sensitivity_scores.append(noise_sensitivity)
                
            except Exception as e:
                self.logger.warning(f"Error calculating RAG metrics for conversation: {str(e)}")
                continue
        
        return {
            "context_precision": {
                "average": np.mean(context_precision_scores) if context_precision_scores else 0,
                "min": np.min(context_precision_scores) if context_precision_scores else 0,
                "max": np.max(context_precision_scores) if context_precision_scores else 0,
                "count": len(context_precision_scores)
            },
            "context_recall": {
                "average": np.mean(context_recall_scores) if context_recall_scores else 0,
                "min": np.min(context_recall_scores) if context_recall_scores else 0,
                "max": np.max(context_recall_scores) if context_recall_scores else 0,
                "count": len(context_recall_scores)
            },
            "faithfulness": {
                "average": np.mean(faithfulness_scores) if faithfulness_scores else 0,
                "min": np.min(faithfulness_scores) if faithfulness_scores else 0,
                "max": np.max(faithfulness_scores) if faithfulness_scores else 0,
                "count": len(faithfulness_scores)
            },
            "response_relevancy": {
                "average": np.mean(response_relevancy_scores) if response_relevancy_scores else 0,
                "min": np.min(response_relevancy_scores) if response_relevancy_scores else 0,
                "max": np.max(response_relevancy_scores) if response_relevancy_scores else 0,
                "count": len(response_relevancy_scores)
            },
            "noise_sensitivity": {
                "average": np.mean(noise_sensitivity_scores) if noise_sensitivity_scores else 0,
                "count": len(noise_sensitivity_scores)
            }
        }
    
    async def _calculate_context_precision(self, query: str, papers: List[Dict[str, Any]]) -> float:
        """
        Calculate Context Precision: Relevant Retrieved Contexts / Total Retrieved Contexts
        """
        if not papers:
            return 0.0
        
        relevant_count = 0
        
        for paper in papers:
            # Use LLM to judge relevance
            is_relevant = await self._judge_paper_relevance(query, paper)
            if is_relevant:
                relevant_count += 1
        
        return relevant_count / len(papers)
    
    async def _calculate_context_recall(self, query: str, retrieved_papers: List[Dict[str, Any]], 
                                      db: Session) -> float:
        """
        Calculate Context Recall: Retrieved Relevant Contexts / Total Relevant Contexts
        This is approximated by comparing with a larger set of potentially relevant papers
        """
        try:
            # Get a broader set of papers for comparison
            broader_papers = self._get_broader_paper_set(query, db, limit=50)
            
            if not broader_papers:
                return 1.0 if retrieved_papers else 0.0
            
            # Count how many of the broader set were actually retrieved
            retrieved_ids = {paper.get("arxiv_id", "") for paper in retrieved_papers}
            relevant_retrieved = sum(1 for paper in broader_papers if paper.arxiv_id in retrieved_ids)
            
            return relevant_retrieved / len(broader_papers) if broader_papers else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating context recall: {str(e)}")
            return 0.5  # Default neutral score
    
    async def _calculate_faithfulness(self, response: str, papers: List[Dict[str, Any]]) -> float:
        """
        Calculate Faithfulness: Faithful Statements / Total Statements
        """
        if not response or not papers:
            return 0.5  # Neutral score if no context
        
        # Extract claims from the response
        claims = self._extract_claims(response)
        if not claims:
            return 1.0  # No claims means perfect faithfulness
        
        faithful_claims = 0
        
        for claim in claims:
            # Check if claim is supported by the papers
            is_faithful = await self._verify_claim_faithfulness(claim, papers)
            if is_faithful:
                faithful_claims += 1
        
        return faithful_claims / len(claims)
    
    async def _calculate_response_relevancy(self, query: str, response: str) -> float:
        """
        Calculate Response Relevancy using semantic similarity
        """
        try:
            # Get embeddings for query and response
            query_embedding = await self._get_embedding(query)
            response_embedding = await self._get_embedding(response)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
            
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2
            
        except Exception as e:
            self.logger.warning(f"Error calculating response relevancy: {str(e)}")
            return 0.5  # Default neutral score
    
    def _calculate_noise_sensitivity(self, conversation: Dict[str, Any]) -> Optional[float]:
        """
        Calculate Noise Sensitivity: Impact of irrelevant information
        """
        papers = conversation.get("arxiv_papers", [])
        if len(papers) < 3:  # Need sufficient papers to assess noise
            return None
        
        # Simple heuristic: ratio of low-relevance papers
        # This is a simplified version - ideally would compare performance with/without noise
        low_relevance_count = 0
        for paper in papers:
            # Simple heuristic: papers with low similarity scores might be noise
            similarity_score = paper.get("similarity_score", 0.5)
            if similarity_score < 0.3:  # Threshold for "noise"
                low_relevance_count += 1
        
        noise_ratio = low_relevance_count / len(papers)
        # Lower noise sensitivity score is better (less affected by noise)
        return 1.0 - noise_ratio
    
    def _calculate_performance_metrics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics (response time, throughput, etc.)
        """
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
    
    async def _calculate_conversation_metrics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate conversation-level metrics
        """
        # Group by session to analyze conversation flow
        sessions = {}
        for conv in conversations:
            session_id = conv["session_id"]
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(conv)
        
        session_lengths = [len(convs) for convs in sessions.values()]
        
        # Topic adherence analysis
        topic_adherence_scores = []
        for session_convs in sessions.values():
            if len(session_convs) > 1:
                adherence = await self._calculate_topic_adherence(session_convs)
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
    
    # Helper methods
    
    async def _judge_paper_relevance(self, query: str, paper: Dict[str, Any]) -> bool:
        """
        Use LLM to judge if a paper is relevant to the query
        """
        try:
            paper_info = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:500]}"
            
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
            return "yes" in response
            
        except Exception as e:
            self.logger.warning(f"Error judging paper relevance: {str(e)}")
            return True  # Default to relevant if error
    
    def _get_broader_paper_set(self, query: str, db: Session, limit: int = 50) -> List[Arxiv]:
        """
        Get a broader set of papers for recall calculation
        """
        try:
            # Simple keyword-based broader search
            keywords = query.lower().split()
            
            query_obj = db.query(Arxiv)
            
            # Add filters for any keyword in title or abstract
            conditions = []
            for keyword in keywords:
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
    
    def _extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from the response
        """
        # Simple approach: split by sentences and filter for factual statements
        sentences = re.split(r'[.!?]+', response)
        
        # Filter out questions, greetings, and non-factual statements
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 10 and 
                not sentence.startswith(('How', 'What', 'Why', 'Where', 'When')) and
                not sentence.lower().startswith(('hello', 'hi', 'thank', 'please'))):
                claims.append(sentence)
        
        return claims[:10]  # Limit to first 10 claims for efficiency
    
    async def _verify_claim_faithfulness(self, claim: str, papers: List[Dict[str, Any]]) -> bool:
        """
        Verify if a claim is supported by the provided papers
        """
        try:
            # Create context from papers
            context = "\n\n".join([
                f"Paper: {paper.get('title', '')}\n{paper.get('abstract', '')[:300]}"
                for paper in papers[:5]  # Limit context size
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
            return "yes" in response
            
        except Exception as e:
            self.logger.warning(f"Error verifying claim faithfulness: {str(e)}")
            return True  # Default to faithful if error
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI API
        """
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    
    async def _calculate_topic_adherence(self, session_conversations: List[Dict[str, Any]]) -> float:
        """
        Calculate how well the conversation stays on topic
        """
        if len(session_conversations) < 2:
            return 1.0
        
        try:
            # Get embeddings for all queries in the session
            query_embeddings = []
            for conv in session_conversations:
                embedding = await self._get_embedding(conv["user_query"])
                query_embeddings.append(embedding)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(query_embeddings) - 1):
                sim = cosine_similarity([query_embeddings[i]], [query_embeddings[i + 1]])[0][0]
                similarities.append((sim + 1) / 2)  # Normalize to 0-1
            
            return np.mean(similarities) if similarities else 1.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating topic adherence: {str(e)}")
            return 0.8  # Default reasonable score 