"""
Chat service for answering questions about LADI classification data
Uses OpenAI GPT-4 with LangChain and Firestore context
"""

import os
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import firebase_admin
from firebase_admin import credentials, firestore

# Store conversation histories by session ID
conversation_histories = {}

# Initialize Firebase once at module level
db = None
try:
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Not initialized yet, initialize now
        FIREBASE_CRED_PATH = "firebase-credentials.json"
        if os.path.exists(FIREBASE_CRED_PATH):
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
            print("[chat_service] Firebase initialized")
        else:
            print(f"[chat_service] Warning: Firebase credentials not found at {FIREBASE_CRED_PATH}")
    
    # Get Firestore client
    db = firestore.client()
    print("[chat_service] Firestore client connected")
    
except Exception as e:
    print(f"[chat_service] Warning: Could not initialize Firebase: {e}")
    db = None

class ChatService:
    def __init__(self, db_client=None):
        """
        Initialize chat service with OpenAI and Firestore
        
        Args:
            db_client: Firestore client instance (optional, will use module-level db if None)
        """
        # Get OpenAI API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize LangChain ChatOpenAI with error handling
        try:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                openai_api_key=api_key,
                request_timeout=30  # 30 second timeout
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {e}")
            raise
        
        # Use provided db_client or fall back to module-level db
        self.db = db_client if db_client is not None else db
        
        if not self.db:
            print("[chat_service] Warning: No Firestore connection available. Chat will work with limited context.")
        
        # System prompt that defines the assistant's role
        self.system_prompt = """You are an AI assistant helping analyze disaster response data from satellite imagery. 
You have access to LADI (Large-scale Aerial Dataset for Disaster Intelligence) classification data stored in Firestore.

The data includes:
- Image classifications from satellite/aerial imagery
- 12 disaster-related labels: flooding_any, flooding_structures, buildings_any, buildings_affected_or_greater, 
  buildings_minor_or_greater, debris_any, roads_any, roads_damage, trees_any, trees_damage, bridges_any, water_any
- Tile-based coverage at zoom level 20
- Confidence scores for each classification
- Geographic coordinates for each classified tile

When answering questions:
1. Use the provided Firestore data context to give accurate answers
2. Cite specific numbers and statistics when available
3. Explain spatial patterns when relevant (e.g., "most flooding is concentrated in the northern area")
4. If asked about specific classifications, explain what they mean in disaster response context
5. Be concise but informative
6. If data is insufficient to answer, say so clearly

Always ground your answers in the actual data provided, not general knowledge about disasters."""

    def get_firestore_context(self) -> Dict:
        """
        Gather context from Firestore about LADI classifications
        
        Returns:
            Dictionary with statistics and sample tile data
        """
        # Handle case where Firestore is not available
        if not self.db:
            return {
                'statistics': {'total_tiles': 0, 'by_label': {}, 'confidence_ranges': {}},
                'sample_tiles': [],
                'error': 'Firestore not available'
            }
        
        try:
            # Get total count first (fast)
            collection_ref = self.db.collection('image_classifications')

            # Query all classifications from the collection (user requested full DB coverage)
            # WARNING: this may be slow for very large collections; ensure your environment
            # can handle iterating over all documents. We still limit the number of
            # sample tiles included in the prompt to avoid huge prompts.
            classifications = collection_ref.stream()
            
            stats = {
                'total_tiles': 0,
                'by_label': {},
                'confidence_ranges': {},
                'coordinates_sample': []
            }
            
            tiles_data = []
            
            for doc in classifications:
                data = doc.to_dict()
                props = data.get('properties', {})
                geom = data.get('geometry', {})
                
                stats['total_tiles'] += 1
                
                # Primary label statistics
                primary_label = props.get('primary_label')
                primary_conf = props.get('primary_confidence', 0)
                
                if primary_label:
                    stats['by_label'][primary_label] = stats['by_label'].get(primary_label, 0) + 1
                    
                    # Track confidence ranges
                    if primary_label not in stats['confidence_ranges']:
                        stats['confidence_ranges'][primary_label] = {'min': 1.0, 'max': 0.0, 'avg_sum': 0, 'count': 0}
                    
                    conf_data = stats['confidence_ranges'][primary_label]
                    conf_data['min'] = min(conf_data['min'], primary_conf)
                    conf_data['max'] = max(conf_data['max'], primary_conf)
                    conf_data['avg_sum'] += primary_conf
                    conf_data['count'] += 1
                
                # Store tile data (limit to avoid huge context)
                if len(tiles_data) < 100:
                    tiles_data.append({
                        'primary_label': primary_label,
                        'confidence': primary_conf,
                        'coordinates': geom.get('coordinates'),
                        'tile': props.get('tile'),
                        'significant_labels': props.get('significant_labels', [])
                    })
            
            # Calculate average confidences
            for label, conf_data in stats['confidence_ranges'].items():
                if conf_data['count'] > 0:
                    conf_data['average'] = conf_data['avg_sum'] / conf_data['count']
                    del conf_data['avg_sum']
                    del conf_data['count']
            
            # Add note that this is based on full collection iteration
            stats['note'] = f"Statistics based on {stats['total_tiles']} tiles (full collection)"
            
            return {
                'statistics': stats,
                'sample_tiles': tiles_data
            }
            
        except Exception as e:
            print(f"Error gathering Firestore context: {e}")
            return {
                'statistics': {'total_tiles': 0, 'by_label': {}, 'confidence_ranges': {}},
                'sample_tiles': [],
                'error': str(e)
            }

    def format_context_for_prompt(self, context: Dict) -> str:
        """
        Format Firestore context into a readable string for GPT
        
        Args:
            context: Dictionary from get_firestore_context()
            
        Returns:
            Formatted string describing the data
        """
        stats = context.get('statistics', {})
        sample_tiles = context.get('sample_tiles', [])
        
        lines = ["=== Current LADI Classification Data ===\n"]
        
        # Total tiles
        total = stats.get('total_tiles', 0)
        lines.append(f"Total classified tiles: {total}\n")
        
        # Label breakdown
        if stats.get('by_label'):
            lines.append("\nClassification breakdown:")
            for label, count in sorted(stats['by_label'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"  - {label}: {count} tiles ({percentage:.1f}%)")
        
        # Confidence ranges
        if stats.get('confidence_ranges'):
            lines.append("\n\nConfidence score ranges by label:")
            for label, conf_data in stats['confidence_ranges'].items():
                avg = conf_data.get('average', 0)
                min_conf = conf_data.get('min', 0)
                max_conf = conf_data.get('max', 0)
                lines.append(f"  - {label}: avg={avg:.1%}, range=[{min_conf:.1%} - {max_conf:.1%}]")
        
        # Sample tiles for spatial context
        if sample_tiles and len(sample_tiles) > 0:
            lines.append(f"\n\nSample tiles (showing {len(sample_tiles)} of {total}):")
            for i, tile in enumerate(sample_tiles[:10]):  # Show first 10
                coords = tile.get('coordinates', [])
                lat = coords[1] if len(coords) > 1 else 'N/A'
                lon = coords[0] if len(coords) > 0 else 'N/A'
                lines.append(f"  {i+1}. {tile['primary_label']} at ({lat}, {lon}), confidence: {tile['confidence']:.1%}")
        
        return "\n".join(lines)

    def get_response(self, user_message: str, session_id: str = 'default') -> str:
        """
        Get a response from GPT-4 based on user message and Firestore context
        If the user asks for a count of a specific label, query Firestore directly for that label and return the count before sending to LLM.
        """
        try:
            # Get or create conversation history for this session
            if session_id not in conversation_histories:
                conversation_histories[session_id] = []
            history = conversation_histories[session_id]

            # Check for direct count queries (e.g., "how many tiles have buildings affected")
            import re
            label_map = {
                'buildings_affected_or_greater': ['buildings affected', 'affected buildings', 'buildings_affected_or_greater'],
                'buildings_minor_or_greater': ['buildings minor', 'minor buildings', 'buildings_minor_or_greater'],
                'water_any': ['water', 'water_any'],
                'buildings_any': ['buildings any', 'buildings_any'],
                'trees_damage': ['trees damage', 'trees_damage'],
                'debris_any': ['debris', 'debris_any'],
                'roads_damage': ['roads damage', 'roads_damage'],
                'roads_any': ['roads any', 'roads_any'],
                'flooding_any': ['flooding', 'flooding_any'],
                'bridges_any': ['bridges', 'bridges_any'],
                'flooding_structures': ['flooding structures', 'flooding_structures'],
                'trees_any': ['trees any', 'trees_any'],
            }
            matched_label = None
            for label, patterns in label_map.items():
                for pat in patterns:
                    if re.search(rf"\b{re.escape(pat)}\b", user_message, re.IGNORECASE):
                        matched_label = label
                        break
                if matched_label:
                    break

            if matched_label and self.db:
                # Query Firestore for the count of tiles with this label
                try:
                    coll = self.db.collection('image_classifications')
                    # Query for primary_label == matched_label
                    docs = coll.where('properties.primary_label', '==', matched_label).stream()
                    count = sum(1 for _ in docs)
                    # Compose direct answer
                    answer = f"There are {count} tiles classified as '{matched_label}' in the LADI disaster response data."
                    # Optionally add a short explanation
                    if matched_label == 'buildings_affected_or_greater':
                        answer += " This classification refers to buildings that have been affected by the disaster to a certain degree, which could range from minor damage to complete destruction."
                    # Update conversation history
                    history.append(HumanMessage(content=user_message))
                    history.append(AIMessage(content=answer))
                    if len(history) > 20:
                        conversation_histories[session_id] = history[-20:]
                    return answer
                except Exception as e:
                    print(f"Error querying Firestore for label count: {e}")
                    # Fallback to LLM if query fails

            # Otherwise, proceed with normal LLM context
            try:
                context = self.get_firestore_context()
                context_str = self.format_context_for_prompt(context)
            except Exception as ctx_error:
                print(f"Warning: Could not gather Firestore context: {ctx_error}")
                context_str = "Note: Unable to access current classification data from Firestore."

            messages = [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=context_str)
            ]
            messages.extend(history[-10:])
            messages.append(HumanMessage(content=user_message))
            try:
                response = self.llm.invoke(messages)
            except Exception as llm_error:
                print(f"Error calling GPT-4: {llm_error}")
                return "I apologize, but I'm having trouble connecting to the AI service right now. Please try again in a moment."
            history.append(HumanMessage(content=user_message))
            history.append(AIMessage(content=response.content))
            if len(history) > 20:
                conversation_histories[session_id] = history[-20:]
            return response.content
        except Exception as e:
            print(f"Error getting chat response: {e}")
            import traceback
            traceback.print_exc()
            return f"I apologize, but I encountered an error processing your request. Please try asking your question in a different way."

    def clear_history(self, session_id: str = 'default'):
        """Clear conversation history for a session"""
        if session_id in conversation_histories:
            del conversation_histories[session_id]
    # Streaming response is removed. Use get_response() for synchronous replies.

# Singleton instance
_chat_service_instance = None

def get_chat_service(db_client=None):
    """Get or create ChatService singleton"""
    global _chat_service_instance
    if _chat_service_instance is None:
        # Use module-level db if no client provided
        _chat_service_instance = ChatService(db_client or db)
    else:
        # If a db_client is provided after the singleton was created, ensure the
        # existing instance uses the provided Firestore client. This handles the
        # import-time initialization order where `chat_service` may be imported
        # before the main app initializes Firebase.
        if db_client is not None:
            try:
                _chat_service_instance.db = db_client
            except Exception:
                pass
    return _chat_service_instance
