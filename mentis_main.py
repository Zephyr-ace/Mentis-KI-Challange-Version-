#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from core.mentis_chat import MentisChat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Check if required environment variables are set
    if not os.getenv("USER_ID"):
        print("L USER_ID environment variable is required")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("L OPENAI_API_KEY environment variable is required")
        return
    
    print("> Mentis initialized - Advanced semantic diary analysis!")
    print("Type 'quit' to exit\n")
    
    # Initialize Mentis chat system
    with MentisChat() as mentis:
        while True:
            try:
                # Get user input
                user_query = input("=Ask Mentis about your diary: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("=K Goodbye from Mentis!")
                    break
                
                if not user_query:
                    print("Please enter a question.")
                    continue
                
                print("\n= Mentis is analyzing your semantic knowledge graph...")
                
                # Get response from Mentis
                response = mentis.chat(user_query)
                
                # Display result
                print("\n" + "="*80)
                print("=🧠 MENTIS RESPONSE")
                print("="*80)
                print(response)
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n=K Goodbye from Mentis!")
                break
            except Exception as e:
                print(f"L Error: {e}")
                print("Please try again.\n")

if __name__ == "__main__":
    main()