import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load or initialize knowledge base
try:
    with open("chat_kb.pkl", "rb") as f:
        training_data = pickle.load(f)
except:
    training_data = {
        "hi": "Hello! How can I help you?",
        "hello": "Hi there!",
        "how are you": "I’m doing great, thanks!",
        "what is your name": "I’m a simple chatbot.",
        "bye": "Goodbye! Have a nice day."
    }

vectorizer = TfidfVectorizer().fit(training_data.keys())
X_train = vectorizer.transform(training_data.keys())

# Short-term memory (for current session)
session_memory = {}

def chatbot_response(user_input):
    global X_train, vectorizer, session_memory
    
    # Check for context-aware placeholders
    if "{name}" in session_memory.get("last_response", ""):
        return session_memory["last_response"].replace("{name}", session_memory.get("user_name", "friend"))
    
    # Vectorize input
    X_input = vectorizer.transform([user_input.lower()])
    
    # Find closest match
    similarities = cosine_similarity(X_input, X_train)
    best_match = similarities.argmax()
    
    if similarities[0, best_match] < 0.3:
        # Ask user for new response
        print("Bot: I don’t know how to respond. How should I reply?")
        new_response = input("You (teach me): ")
        training_data[user_input.lower()] = new_response
        
        # Update vectorizer and training vectors
        vectorizer = TfidfVectorizer().fit(training_data.keys())
        X_train = vectorizer.transform(training_data.keys())
        
        # Save knowledge base
        with open("chat_kb.pkl", "wb") as f:
            pickle.dump(training_data, f)
        
        return "Got it! I’ll remember that."
    
    # Example of remembering user info
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[-1].strip().capitalize()
        session_memory["user_name"] = name
        response = f"Nice to meet you, {name}!"
    else:
        key = list(training_data.keys())[best_match]
        response = training_data[key]
    
    session_memory["last_response"] = response
    return response

# Chat loop
print("Bot: Hello! Type 'quit' to exit.")
while True:
    user = input("You: ")
    if user.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(user))
