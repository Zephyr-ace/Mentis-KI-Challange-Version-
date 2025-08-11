from core.retriever import Retriever
from core.llm import LLM_OA
from config.prompts import mentisPrompt


class MentisChat:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLM_OA("o3")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def chat(self, user_query: str) -> str:
        """Generate answer using only the main retriever (Mentis)"""
        
        # 1. Retrieve from main retriever only
        main_retrieval_output = self.retriever.retrieve(user_query)
        main_results = main_retrieval_output.get("results", {}) if isinstance(main_retrieval_output, dict) else main_retrieval_output
        
        # 2. Format results
        main_context = self._format_main_results(main_results)
        
        # 3. Generate answer with Mentis prompt
        return self._generate_answer(user_query, main_context)
    
    def _format_main_results(self, results: dict[str, list]) -> str:
        """Format results from main retriever (structured objects by category)"""
        if not results:
            return "No relevant information found."
        
        formatted_parts = []
        for category, items in results.items():
            if items:
                formatted_parts.append(f"\n--- {category} ---")
                for item in items:
                    # Extract the model instance (item is tuple of (model, score))
                    model_instance = item[0] if isinstance(item, tuple) else item
                    
                    # Format based on model type
                    if hasattr(model_instance, 'title') and hasattr(model_instance, 'description'):
                        formatted_parts.append(f"• {model_instance.title}: {model_instance.description}")
                    elif hasattr(model_instance, 'name') and hasattr(model_instance, 'description'):
                        formatted_parts.append(f"• {model_instance.name}: {model_instance.description}")
                    elif hasattr(model_instance, 'content'):
                        formatted_parts.append(f"• {model_instance.content}")
                    else:
                        formatted_parts.append(f"• {str(model_instance)}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No relevant information found."
    
    def _generate_answer(self, user_query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context and Mentis prompt"""
        prompt = f"""{mentisPrompt}

User Question: {user_query}

Retrieved Information:
{context}

Answer:"""
        
        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"