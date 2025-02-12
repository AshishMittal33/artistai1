import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import requests

# Load Firebase credentials from Streamlit secrets
firebase_secrets = st.secrets["firebase_credentials"]
cred_dict = {
    "type": firebase_secrets["type"],
    "project_id": firebase_secrets["project_id"],
    "private_key_id": firebase_secrets["private_key_id"],
    "private_key": firebase_secrets["private_key"],
    "client_email": firebase_secrets["client_email"],
    "client_id": firebase_secrets["client_id"],
    "auth_uri": firebase_secrets["auth_uri"],
    "token_uri": firebase_secrets["token_uri"],
    "auth_provider_x509_cert_url": firebase_secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": firebase_secrets["client_x509_cert_url"],
    "universe_domain": firebase_secrets["universe_domain"]
}

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()
collection_name = "game_data"  # Firestore collection for storing data

# Load Groq API key from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]

# Initialize LlamaIndex embedding model
embedding_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = embedding_model

# Load and index documents from Firestore
@st.cache_resource
def load_and_index_documents():
    try:
        docs = db.collection(collection_name).stream()
        # Convert Firestore documents to LlamaIndex Document objects
        documents = [Document(text=f"Location Name: {doc.to_dict()['location_name']}\nBasic Details: {doc.to_dict()['basic_details']}") for doc in docs]
        index = VectorStoreIndex.from_documents(documents)
        return index
    except Exception as e:
        st.error(f"⚠️ Error loading and indexing documents: {e}")
        return None

# Create index from Firestore data
index = load_and_index_documents()
retriever = index.as_retriever()

# Helper function to add data to Firestore
def add_text_to_firestore(location_name, basic_details):
    try:
        db.collection(collection_name).add({
            "location_name": location_name,
            "basic_details": basic_details
        })
        return True
    except Exception as e:
        st.error(f"⚠️ Error adding text to Firestore: {e}")
        return False

# Retrieve all data from Firestore as a single text blob
def get_all_data_from_firestore():
    try:
        docs = db.collection(collection_name).stream()
        all_data = "\n".join([f"Location Name: {doc.to_dict()['location_name']}\nBasic Details: {doc.to_dict()['basic_details']}" for doc in docs])
        return all_data.lower()
    except Exception as e:
        st.error(f"⚠️ Error retrieving data from Firestore: {e}")
        return ""

def retrieve_relevant_text(query):
    try:
        results = retriever.retrieve(query)
        st.write(results)  # Log results to Streamlit for debugging
        # Extract text from NodeWithScore objects
        retrieved_text = "\n".join([result.node.text for result in results])
        return retrieved_text
    except Exception as e:
        st.error(f"⚠️ Error retrieving relevant text: {e}")
        return ""

def find_location_description(location_name, data_text):
    try:
        lines = data_text.split("\n")
        for i, line in enumerate(lines):
            if location_name.lower() in line.lower() and "location name:" in line.lower():
                # Return the location name and its corresponding basic details
                return f"{line}\n{lines[i+1]}"
        return None
    except Exception as e:
        st.error(f"⚠️ Error finding location description: {e}")
        return None

# Chat logic
def chat_with_bot(user_message, data_text, last_location=None):
    user_message_lower = user_message.lower()

    # Retrieve relevant context using LlamaIndex RAG
    retrieved_context = retrieve_relevant_text(user_message)

    if user_message_lower.startswith("artist ai"):
        command = user_message[len("artist ai"):].strip()
        command_lower = command.lower()
        system_message = (
            "You are a creative AI assistant for game artists specializing in pixel art. "
            "You help design locations and buildings with detailed visual descriptions, "
            "including colors, textures, architecture, and artistic composition. "
            "Focus on making your answers detailed, formatted, and tailored for pixel art environments."
        )

        if command_lower.startswith("describe "):
            location_name = command[9:].strip()
            location_details = find_location_description(location_name, data_text)

            if location_details:
                prompt = f"""
                Strictly describe the location '{location_name}' as it is in the document.
                Provide a well-organized format with these sections:
                1. **Location Overview**: <br>
                   A brief summary of the location, including its purpose and significance. <br><br>
                   
                2. **Visual Details**: <br>
                   Describe the environment, lighting, colors, and overall atmosphere. Mention how these details would look in a pixel art style, including suggestions for textures and palette choices. <br><br>
                   
                3. **Key Areas and Features**: <br>
                   Highlight at least 7 or more the key landmarks, streets, or zones with sizes measured in tiles (e.g., 4x4 tiles, 8x8 tiles)  . <br><br>
                   
                4. **Building Styles**: <br>
                   Describe the architecture and materials used for buildings. Include guidance on roof shapes, window placements, and how to pixelate these details effectively. <br><br>
                   
                5. **Layout and Composition**: <br>
                   Explain how the location is arranged spatially. Provide ideas for creating depth and perspective in pixel art. <br><br>

                6. **population**: <br>
                   Explain how the location population like how much npc can be used types of npc. Provide ideas for creating depth and perspective in pixel art. <br><br>

                7. **quests**: <br>
                   Explain ideas of quests that can make around the town and rewards and things player gets. <br><br>

                6. **activities**: <br>
                   activites that a player can do around the town give ideas. <br><br>
                """
                last_location = location_name  # Update last location
            else:
                prompt = f"Could not find location '{location_name}' in the document."
                last_location = None  # Reset last location
        elif last_location and command_lower in {"population", "population density", "how many people can live here?"}:
            prompt = f"Estimate the population density for the '{last_location}' area based on its size and typical characteristics."
        else:
            prompt = command
    else:
        system_message = "You are a helpful assistant."
        prompt = user_message

    # Enhance prompt with retrieved knowledge
    if retrieved_context:
        prompt = f"Relevant Information:\n{retrieved_context}\n\nUser Query:\n{prompt}"

    # API Configuration
    api_key = groq_api_key  # Use the API key from Streamlit secrets
    url = "https://api.groq.com/openai/v1/chat/completions"
    model_id = "llama3-8b-8192"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            chatbot_response = response.json()
            bot_message = chatbot_response["choices"][0]["message"]["content"]
            return bot_message, last_location
        else:
            return "An error occurred while processing your request.", last_location
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again.", last_location
    except Exception as e:
        return f"An unexpected error occurred: {e}", last_location

# Streamlit UI
st.title("AI Chat for Game Artists")

# Load data from Firestore
data_text = get_all_data_from_firestore()
last_location = None

# Chat section
st.header("Chat with Artist AI")
user_input = st.text_input("Enter your message:")
if st.button("Send"):
    if user_input:
        response, last_location = chat_with_bot(user_input, data_text, last_location)
        st.markdown(f"**AI Response:** {response}")
    else:
        st.warning("⚠️ Please enter a message.")

# Add to Firestore section
st.header("Add Text to Online Database")
location_name = st.text_input("Enter location name:")
basic_details = st.text_area("Enter basic details:")
if st.button("Add to Database"):
    if location_name and basic_details:
        success = add_text_to_firestore(location_name, basic_details)
        if success:
            st.success("✅ Text successfully added to the database.")
        else:
            st.error("⚠️ Failed to add text to the database.")
    else:
        st.warning("⚠️ Please provide both location name and basic details.")
