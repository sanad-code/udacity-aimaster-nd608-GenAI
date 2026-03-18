import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path
import os

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Create list of directories in case it is a directory and its name contains "chroma" or "db" (case-insensitive)
    chroma_dirs = [d for d in current_dir.iterdir() 
                   if d.is_dir() and ('chroma' in d.name.lower() or 'db' in d.name.lower())]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry for privacy
                    allow_reset=True             # Allow database reset for development
                )
            )
            
            # Retrieve list of available collections from the database
            collections = client.list_collections()
            
            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}_{collection.name}"
                
                # Build information dictionary
                # First I will check the document count to handle the fallback of unsupported operations.
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = 0
                
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({chroma_dir.name}) - {doc_count} docs",
                    "doc_count": doc_count
                }
        
        # Handle connection exceptions
        except Exception as e:
            # Create fallback entry for inaccessible directories
            key = f"{chroma_dir.name}_error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (Error: {str(e)[:250]})",
                "doc_count": 0
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chromadb persistentclient
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(
            anonymized_telemetry=False,  # Disable telemetry for privacy
            allow_reset=True             # Allow database reset for development
        )
    )
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return None, False, "OpenAI API key not found in environment variables"
    
    # Create OpenAI embedding function matching the one used during collection creation
    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )
    
    # Return the collection with the collection_name and embedding function
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    # I have added the other two return values along with collection as this is what is expected when this function is called in the chat.py
    return collection, True, None

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() not in ["all",  "", "none"]:
        # Create filter dictionary with appropriate field-value pairs
        # I have decided to use mission for filtering as this is what is contained in the data_text folder
        where_filter = {"mission": mission_filter}

    # Execute database query with the following parameters
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context with deduplication and score sorting.

    Deduplicates identical or near-identical chunks and sorts results by
    similarity score so the most relevant documents appear first.
    """
    if not documents:
        return ""

    # Assign default distances if not provided (preserve original order)
    if distances is None:
        distances = list(range(len(documents)))

    # Bundle documents with their metadata and distances, then sort by
    # distance (lower distance = higher similarity in ChromaDB)
    combined = sorted(
        zip(documents, metadatas, distances),
        key=lambda x: x[2]
    )

    # Deduplicate: skip documents whose text has already been seen.
    # Uses a normalized comparison (stripped whitespace) so near-identical
    # chunks with minor whitespace differences are also caught.
    seen_texts: set = set()
    unique_items = []
    for doc, meta, dist in combined:
        normalized = doc.strip()
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_items.append((doc, meta, dist))

    # Initialize list with header text for context section
    context_sections = ["**** RAG Context *****"]

    # Loop through deduplicated and sorted documents
    for i, (doc, meta, _dist) in enumerate(unique_items):
        # Extract mission information from metadata with fallback value
        mission = meta.get("mission", "Unknown Mission")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()

        # Extract source information from metadata with fallback value
        source = meta.get("source", "Unknown Source")

        # Extract category information from metadata with fallback value
        category = meta.get("document_category", "General")
        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()

        # Create formatted source header with index number and extracted information
        header = f"\n--- Source {i+1}: {mission} | {source} | {category} ---"
        # Add source header to context parts list
        context_sections.append(header)

        # Check document length and truncate if necessary
        if len(doc) > 2500:
            context_sections.append(doc[:2500] + "... [Max Length of 2500 char reached!]")
        else:
            # Add full document content to context parts list
            context_sections.append(doc)

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_sections)