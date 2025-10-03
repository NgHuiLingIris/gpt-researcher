import logging
import os
import uuid
import json
from fastapi import WebSocket
from typing import Iterable, List, Dict, Any

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from gpt_researcher.memory import Memory
from gpt_researcher.config.config import Config
from tavily import TavilyClient
from datetime import datetime

# Setup logging
# Get logger instance
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_tools():
    """Define tools for OpenAI to use"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "quick_search",
                "description": "Search for current events or online information when you need new knowledge that doesn't exist in the current context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    return tools

class ChatAgentWithMemory:
    def __init__(
        self,
        report: str,
        config_path="default",
        headers=None,
        vector_store=None
    ):
        self.report = report
        self.headers = headers
        self.config = Config(config_path)
        self.vector_store = vector_store
        self.retriever = None
        self.search_metadata = None
        
        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        
        # Process document and create vector store if not provided
        logger.info("Setting up vector store for document retrieval")
        if not self.vector_store:
            self._setup_vector_store()

    def retrieve_local_context(self, query: str, k: int = 4) -> List[str]:
        if not self.retriever:
            return []
        docs = self.retriever.get_relevant_documents(query)
        # docs could be strings if you used add_texts; otherwise extract page_content
        chunks = []
        for d in docs:
            if isinstance(d, str):
                chunks.append(d)
            else:
                chunks.append(getattr(d, "page_content", str(d)))
        return chunks[:k]

    def _setup_vector_store(self):
        """Setup vector store for document retrieval"""
        # Process document into chunks
        documents = self._process_document(self.report)
        
        # Create unique thread ID
        self.thread_id = str(uuid.uuid4())
        
        # Setup embeddings and vector store
        cfg = Config()
        self.embedding = Memory(
            cfg.embedding_provider,
            cfg.embedding_model,
            **cfg.embedding_kwargs
        ).get_embeddings()
        
        # Create vector store and retriever
        self.vector_store = InMemoryVectorStore(self.embedding)
        self.vector_store.add_texts(documents)
        self.retriever = self.vector_store.as_retriever(k=4)
        
    def _process_document(self, report):
        """Split Report into Chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.split_text(report)
        return documents

    def quick_search(self, query):
        """Perform a web search for current information using Tavily"""
        try:
            logger.info(f"Performing web search for: {query}")
            results = self.tavily_client.search(query=query, max_results=5)
            
            # Store search metadata for frontend
            self.search_metadata = {
                "query": query,
                "sources": [
                    {"title": result.get("title", ""), 
                     "url": result.get("url", ""),
                     "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")}
                    for result in results.get("results", [])
                ]
            }
            
            return results
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "results": []
            }

    def handle_tool_calls(self, messages, response_message):

        cfg = Config()
        """Handle tool calls from OpenAI"""
        tool_calls_metadata = []
        
        # First, add the assistant's message with the tool_calls to the messages
        messages.append({
            "role": "assistant",
            "content": response_message.content if response_message.content else None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        })
        
        # Then process each tool call
        for tool_call in response_message.tool_calls:
            function_args = json.loads(tool_call.function.arguments)
            
            if tool_call.function.name == "quick_search":
                query = function_args.get("query")
                
                # Perform web search
                search_results = self.quick_search(query)
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "quick_search",
                    "content": json.dumps(search_results)
                })
                
                # Add metadata for this tool call
                tool_calls_metadata.append({
                    "tool": "quick_search",
                    "query": query,
                    "search_metadata": self.search_metadata
                })
        
        # Get a new response from the model with the tool results
        second_response = client.chat.completions.create(
            model=cfg.fast_llm_model,
            messages=messages,
        )
        
        return second_response.choices[0].message.content, tool_calls_metadata

    def process_chat_completion(self, messages: List[Dict[str, str]], stream: bool = False):
        """
        Process chat completion using OpenAI's API.

        Returns:
        - If stream == False:
            (content: str, tool_calls_metadata: List[Dict])
        - If stream == True:
            (token_iterator: Iterable[str], tool_calls_metadata: List[Dict])

        Streaming behavior:
        - If the model requests tools, we execute them and then simulate streaming by chunking the
            final text returned from handle_tool_calls (simple, reliable).
        - If no tools are requested, we use OpenAI's native streaming to yield tokens as they arrive.
        """
        cfg = Config()

        # First pass: let the model decide whether to use tools
        response = client.chat.completions.create(
            model=cfg.fast_llm_model,
            messages=messages,
            tools=get_tools(),
            # Do NOT set stream=True here; we need to inspect tool_calls synchronously first.
        )

        response_message = response.choices[0].message

        # If the response contains tool calls, execute them
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            # Execute tools and get a finalized answer (non-stream)
            content, tool_calls_metadata = self.handle_tool_calls(messages, response_message)

            if not stream:
                return content, tool_calls_metadata

            # Simulate streaming by chunking the content
            def _simulate_stream(text: str, chunk_size: int = 40) -> Iterable[str]:
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]

            return _simulate_stream(content), tool_calls_metadata

        # No tools: either return the full text or stream directly from OpenAI
        if not stream:
            return response_message.content, []

        # Real streaming path when tools are NOT used
        stream_resp = client.chat.completions.create(
            model=cfg.fast_llm_model,
            messages=messages,
            stream=True,
            # Intentionally omit tools here to avoid new tool calls in the streaming pass
        )

        def _iter_tokens() -> Iterable[str]:
            for chunk in stream_resp:
                # OpenAI SDK: token deltas are in choices[0].delta.content
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    yield token

        return _iter_tokens(), []

    async def chat(self, messages, websocket=None, stream: bool = False):
        """
        Chat with OpenAI directly.

        Args:
            messages: List of chat messages with role and content.
            websocket: Optional WebSocket for token-by-token streaming to the client.
            stream: If True, uses streaming; otherwise returns a full response.

        Returns:
            tuple:
            - ai_message (str): The assistant message (full text).
            - tool_calls_metadata (list[dict]): Metadata about tool usage.
        """
        try:
            # Format system prompt with the report context
            system_prompt = f"""
            You are GPT Researcher, an autonomous research agent created by an open source community at https://github.com/assafelovic/gpt-researcher, homepage: https://gptr.dev. 
            To learn more about GPT Researcher you can suggest to check out: https://docs.gptr.dev.
            
            This is a chat about a research report that you created. Answer based on the given context and report.
            You must include citations to your answer based on the report.
            
            You may use the quick_search tool when the user asks about information that might require current data 
            not found in the report, such as recent events, updated statistics, or news. If there's no report available,
            you can use the quick_search tool to find information online.
            
            You must respond in markdown format. You must make it readable with paragraphs, tables, etc when possible. 
            Remember that you're answering in a chat not a report.
            
            Assume the current time is: {datetime.now()}.
            
            Report: {self.report}
            """

            # Format message history for OpenAI input
            formatted_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                if "role" in msg and "content" in msg:
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    logger.warning(f"Skipping message with missing role or content: {msg}")

            # If you do local RAG, inject it here just like in chat()
            last_user_query = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_query = m.get("content", "")
                    break
            local_ctx = getattr(self, "retrieve_local_context", lambda *_: [])(last_user_query, k=4)
            if local_ctx:
                formatted_messages.append({
                    "role": "system",
                    "content": "Relevant local context (retrieved from the report):\n\n" + "\n\n".join(f"- {c}" for c in local_ctx)
                })

            if stream:
                # Streaming mode
                token_iter, tool_calls_metadata = self.process_chat_completion(formatted_messages, stream=True)
                accumulated = []
                for token in token_iter:
                    accumulated.append(token)
                    if websocket:
                        await websocket.send_text(token)
                ai_message = "".join(accumulated)
            else:
                # Non-streaming mode
                ai_message, tool_calls_metadata = self.process_chat_completion(formatted_messages, stream=False)

            # Provide fallback response if message is empty
            if not ai_message:
                logger.warning("No AI message content found in response, using fallback message")
                ai_message = "I apologize, but I couldn't generate a proper response. Please try asking your question again."

            logger.info(
                f"Generated response: {ai_message[:100]}..." if len(ai_message) > 100 else f"Generated response: {ai_message}"
            )

            # Return both the message and any metadata about tools used
            return ai_message, tool_calls_metadata

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise
    def get_context(self):
        """return the current context of the chat"""
        return self.report