import logging
from typing import Dict, Any, List, Optional
import openai
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BaseAgent:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = "text-embedding-ada-002"
        self.chat_model = "gpt-4"
        self.conversation_history = []

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using OpenAI's API
        """
        self.logger.info(f"Generating embedding for text: {text[:100]}...")
        try:
            response = await openai.Embedding.acreate(
                model=self.embedding_model,
                input=text
            )
            embedding = response['data'][0]['embedding']
            self.logger.info("Successfully generated embedding")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model: str = "gpt-4"
    ) -> str:
        """
        Generate a completion using OpenAI's API with the new message format
        """
        self.logger.info(f"Generating completion with temperature {temperature}")
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=temperature
            )
            self.logger.info("Successfully generated completion")
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating completion: {str(e)}")
            raise

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")

    async def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user's intent from their query
        """
        self.logger.info("Analyzing user intent")
        completion = await self.generate_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant helping to analyze user queries. Extract the main topic, time period, specific requirements, and type of information needed. Return the analysis as a JSON object."
                },
                {
                    "role": "user",
                    "content": f"Analyze this research query: {query}"
                }
            ],
            temperature=0.3
        )
        
        try:
            intent = json.loads(completion)  # Using json.loads instead of eval for safe parsing
            self.logger.info(f"Successfully analyzed intent: {intent}")
            return intent
        except Exception as e:
            self.logger.error(f"Error parsing intent: {str(e)}")
            return {
                "Main topic or research area": "Not specified",
                "Time period": "Not specified",
                "Specific requirements or constraints": "Not specified",
                "Type of information needed": "Not specified"
            }

    async def generate_follow_up_questions(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """
        Generate relevant follow-up questions based on the query and intent
        """
        self.logger.info("Generating follow-up questions")
        completion = await self.generate_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant helping to generate relevant follow-up questions. Generate 3 questions that would help clarify or expand the user's research query."
                },
                {
                    "role": "user",
                    "content": f"Generate follow-up questions for this query: {query}\nUser's intent: {intent}"
                }
            ],
            temperature=0.7
        )
        
        try:
            # Try to parse as JSON first
            questions = eval(completion)  # Using eval since the response might be a Python list
            if isinstance(questions, list):
                self.logger.info(f"Generated {len(questions)} follow-up questions")
                return questions
        except Exception:
            # If not JSON, split by newlines and clean up
            questions = [line.strip() for line in completion.split('\n') if line.strip()]
            self.logger.info(f"Generated {len(questions)} follow-up questions")
            return questions 