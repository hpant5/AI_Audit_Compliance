import os
import warnings
from typing import List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

class LegalResearchChatbot:
    def __init__(self, chroma_db_path="./chroma_legal_db"):
        """Initialize legal research assistant"""
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.3,  # Balanced temperature
        )
        
        # Connect to ChromaDB (suppress output)
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Load collections
        try:
            self.case_law = self.client.get_collection("legal_cases_bert")
        except:
            self.case_law = None
        
        try:
            self.federal_law = self.client.get_collection("federal_laws_bert")
        except:
            self.federal_law = None
        
        # Conversation history
        self.history = []
    
    def search_sources(self, query: str) -> Dict:
        """Search both databases and return sources"""
        sources = {
            'case_law': [],
            'federal_law': []
        }
        
        # Search case law
        if self.case_law:
            try:
                results = self.case_law.query(query_texts=[query], n_results=5)
                if results['documents'][0]:
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                        sources['case_law'].append({
                            'text': doc[:1000],
                            'case_name': meta.get('case_name', 'Unknown'),
                            'state': meta.get('state', 'Unknown'),
                            'court': meta.get('court', 'Unknown'),
                            'citation': meta.get('citation', 'Unknown'),
                            'date': meta.get('decision_date', 'Unknown'),
                            'type': meta.get('case_type', 'Unknown')
                        })
            except:
                pass
        
        # Search federal law
        if self.federal_law:
            try:
                results = self.federal_law.query(query_texts=[query], n_results=5)
                if results['documents'][0]:
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                        sources['federal_law'].append({
                            'text': doc[:1000],
                            'title': meta.get('title', 'Unknown'),
                            'subject': meta.get('subject', 'Unknown'),
                            'cfr_title': meta.get('cfr_title_number', 'Unknown')
                        })
            except:
                pass
        
        return sources
    
    def format_context(self, sources: Dict) -> str:
        """Format sources into context"""
        context = []
        
        if sources['case_law']:
            context.append("=== RELEVANT CASE LAW ===\n")
            for i, case in enumerate(sources['case_law'], 1):
                context.append(f"Case {i}: {case['case_name']}")
                context.append(f"State: {case['state']}, Court: {case['court']}")
                context.append(f"Content: {case['text']}\n")
        
        if sources['federal_law']:
            context.append("\n=== RELEVANT FEDERAL REGULATIONS ===\n")
            for i, law in enumerate(sources['federal_law'], 1):
                context.append(f"Regulation {i}: {law['title']}")
                context.append(f"Content: {law['text']}\n")
        
        return "\n".join(context)
    
    def format_history(self) -> str:
        """Format recent conversation history"""
        if not self.history:
            return ""
        
        history_text = "\n=== CONVERSATION CONTEXT ===\n"
        for entry in self.history[-3:]:
            history_text += f"\nUser: {entry['question']}\nAssistant: {entry['answer'][:250]}...\n"
        return history_text
    
    def answer(self, question: str) -> str:
        """Generate helpful answer with or without database sources"""
        
        # Search for relevant sources
        sources = self.search_sources(question)
        has_sources = bool(sources['case_law'] or sources['federal_law'])
        
        # Build context
        history = self.format_history()
        
        if has_sources:
            # We have database sources - use them
            context = self.format_context(sources)
            
            system_prompt = """You are a helpful legal research assistant. 

Your goal is to provide practical, useful answers to legal questions.

When you have relevant legal sources available:
- Use the information from those sources to inform your answer
- Provide specific, detailed guidance based on the legal principles in the sources
- Make your answer practical and actionable

When answering:
- Be direct and helpful
- Focus on what the person should do or know
- Explain legal concepts in clear, understandable language
- Provide practical next steps
- Be professional but conversational

Do not:
- Simply list what's in or not in the database
- Focus on limitations of your sources
- Over-cite or create a academic paper
- Refuse to help because sources aren't perfect

Your job is to be genuinely helpful while being legally sound."""

            user_prompt = f"""Question: {question}

{history}

Available Legal Information:
{context}

Provide a helpful, practical answer to this person's question. Use the legal information available to give them good guidance."""

        else:
            # No database sources - provide general guidance
            system_prompt = """You are a helpful legal research assistant.

Your goal is to provide practical, useful guidance on legal questions.

When answering:
- Provide general legal guidance based on common legal principles
- Explain what typically happens in these situations
- Give practical steps the person should take
- Explain key legal concepts they should understand
- Recommend when to seek professional help

Be:
- Direct and helpful
- Practical and actionable
- Professional but approachable
- Clear about when they need a lawyer

Do not:
- Refuse to help
- Focus on what you don't know
- Give overly cautious non-answers
- Make the person feel helpless"""

            user_prompt = f"""Question: {question}

{history}

Provide helpful, practical legal guidance for this question. Focus on what they should know and do."""
        
        # Get response from LLM
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        answer = response.content
        
        # Save to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'had_sources': has_sources
        })
        
        return answer
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*70)
        print("LEGAL RESEARCH ASSISTANT")
        print("="*70)
        print("\nAsk me any legal question. Type 'quit' to exit.\n")
        print("="*70 + "\n")
        
        while True:
            try:
                question = input("Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!\n")
                    break
                
                print("\nThinking...\n")
                answer = self.answer(question)
                print(answer)
                print("\n" + "-"*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\nI apologize, but I encountered an error. Please try rephrasing your question.\n")

def main():
    chatbot = LegalResearchChatbot()
    chatbot.chat()

if __name__ == "__main__":
    main()