"""
Enterprise Knowledge Graph Construction & Querying Agent
A multi-agent AI system for building and querying knowledge graphs
"""

import streamlit as st
import networkx as nx
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from config import AZURE_ENDPOINT, AZURE_API_KEY, DEPLOYMENT_NAME, API_VERSION, EMBEDDING_MODEL
# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Visualization
from pyvis.network import Network
import tempfile

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Knowledge Graph Agent",
    page_icon="🧠",
    layout="wide"
)

# Initialize LLM
@st.cache_resource
def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
        deployment_name=DEPLOYMENT_NAME,
        temperature=1
    )

@st.cache_resource
def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
        model=EMBEDDING_MODEL
    )

# ============================================================================
# STATE DEFINITION FOR LANGGRAPH
# ============================================================================

class GraphState(TypedDict):
    """State for the knowledge graph construction workflow"""
    documents: List[Document]
    text_chunks: List[str]
    entities: List[Dict]
    relationships: List[Dict]
    validated_entities: List[Dict]
    validated_relationships: List[Dict]
    graph: nx.Graph
    vector_store: Any
    processing_log: List[str]
    error: str

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = nx.Graph()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None

# ============================================================================
# AGENT 1: DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessorAgent:
    """Processes and chunks documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """Process raw text into chunks"""
        chunks = self.text_splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]
        return documents
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process uploaded file"""
        try:
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load based on file type
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path)
                docs = loader.load()
            else:
                # Plain text
                text = uploaded_file.getvalue().decode('utf-8')
                docs = [Document(page_content=text, metadata={'source': uploaded_file.name})]
            
            # Clean up
            os.unlink(tmp_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(docs)
            return chunks
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return []

# ============================================================================
# AGENT 2: ENTITY EXTRACTOR
# ============================================================================

class EntityExtractorAgent:
    """Extracts entities from text using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text"""
        prompt = f"""Extract entities from the following text.

Text: {text}

Extract the following types of entities:
- Person (people, roles)
- Organization (companies, institutions)
- Location (places, cities, countries)
- Concept (technologies, ideas, topics)
- Product (products, services)

Return ONLY a JSON array with this format:
[
  {{"name": "entity name", "type": "Person|Organization|Location|Concept|Product", "properties": {{"key": "value"}}}},
  ...
]

Return ONLY valid JSON, no other text."""

        try:
            messages = [
                SystemMessage(content="You are an expert entity extractor. Extract entities with high precision. Return only valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            # Clean response
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            entities = json.loads(result.strip())
            return entities if isinstance(entities, list) else []
            
        except Exception as e:
            st.warning(f"Entity extraction error: {e}")
            return []

# ============================================================================
# AGENT 3: RELATIONSHIP ANALYZER
# ============================================================================

class RelationshipAnalyzerAgent:
    """Analyzes relationships between entities"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Find relationships between entities"""
        if len(entities) < 2:
            return []
        
        entity_names = [e['name'] for e in entities]
        
        prompt = f"""Given the following entities from a text:
{json.dumps(entity_names, indent=2)}

And the original text:
{text}

Identify relationships between these entities.

Return ONLY a JSON array with this format:
[
  {{"source": "entity1", "target": "entity2", "relation": "WORKS_FOR|LOCATED_IN|PART_OF|RELATED_TO", "confidence": 0.0-1.0}},
  ...
]

Return ONLY valid JSON, no other text."""

        try:
            messages = [
                SystemMessage(content="You are an expert at identifying relationships between entities. Return only valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            # Clean response
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            relationships = json.loads(result.strip())
            return relationships if isinstance(relationships, list) else []
            
        except Exception as e:
            st.warning(f"Relationship analysis error: {e}")
            return []

# ============================================================================
# AGENT 4: KNOWLEDGE GRAPH BUILDER
# ============================================================================

class KnowledgeGraphBuilderAgent:
    """Builds and maintains the knowledge graph"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def add_entity(self, entity: Dict):
        """Add entity to graph"""
        name = entity.get('name')
        entity_type = entity.get('type', 'Unknown')
        properties = entity.get('properties', {}).copy()
        
        if name:
            # Add type to properties
            properties['type'] = entity_type
            
            # Add or update node
            if self.graph.has_node(name):
                # Update properties
                self.graph.nodes[name].update(properties)
            else:
                # Add new node
                self.graph.add_node(name, **properties)
    
    def add_relationship(self, relationship: Dict):
        """Add relationship to graph"""
        source = relationship.get('source')
        target = relationship.get('target')
        relation = relationship.get('relation', 'RELATED_TO')
        confidence = relationship.get('confidence', 0.8)
        
        if source and target:
            # Ensure both nodes exist
            if not self.graph.has_node(source):
                self.graph.add_node(source, type='Unknown')
            if not self.graph.has_node(target):
                self.graph.add_node(target, type='Unknown')
            
            # Add edge
            self.graph.add_edge(source, target, relation=relation, confidence=confidence)
    
    def get_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'node_types': self._count_node_types(),
            'relation_types': self._count_relation_types()
        }
    
    def _count_node_types(self) -> Dict:
        """Count nodes by type"""
        types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            types[node_type] = types.get(node_type, 0) + 1
        return types
    
    def _count_relation_types(self) -> Dict:
        """Count edges by relation type"""
        relations = {}
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'RELATED_TO')
            relations[relation] = relations.get(relation, 0) + 1
        return relations

# ============================================================================
# AGENT 5: QUERY AGENT
# ============================================================================

class QueryAgent:
    """Answers questions using the knowledge graph"""
    
    def __init__(self, llm, graph: nx.Graph, vector_store=None):
        self.llm = llm
        self.graph = graph
        self.vector_store = vector_store
    
    def query(self, question: str) -> str:
        """Answer question using knowledge graph"""
        
        # Get relevant context from vector store
        context = ""
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
            except:
                pass
        
        # Get graph info
        graph_info = self._get_relevant_graph_info(question)
        
        # Generate answer
        prompt = f"""Answer the following question using the knowledge graph and context provided.

Question: {question}

Knowledge Graph Information:
{graph_info}

Document Context:
{context}

Provide a clear, concise answer based on the available information. If the information is not available, say so."""

        try:
            print("prompt",prompt)
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions using a knowledge graph and document context."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _get_relevant_graph_info(self, question: str) -> str:
        """Extract relevant information from graph"""
        # Simple keyword matching
        question_lower = question.lower()
        relevant_info = []
        
        # Find mentioned nodes
        for node in self.graph.nodes():
            if node.lower() in question_lower:
                # Get node info
                node_data = self.graph.nodes[node]
                relevant_info.append(f"Entity: {node} (Type: {node_data.get('type', 'Unknown')})")
                
                # Get neighbors
                neighbors = list(self.graph.neighbors(node))
                if neighbors:
                    for neighbor in neighbors[:5]:  # Limit to 5
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        relation = edge_data.get('relation', 'RELATED_TO') if edge_data else 'RELATED_TO'
                        relevant_info.append(f"  - {relation} -> {neighbor}")
        
        if not relevant_info:
            # Return general graph stats
            stats = f"Graph contains {self.graph.number_of_nodes()} entities and {self.graph.number_of_edges()} relationships."
            return stats
        
        return "\n".join(relevant_info[:20])  # Limit output

# ============================================================================
# AGENT 6: VALIDATOR
# ============================================================================

class ValidatorAgent:
    """Validates extracted data quality"""
    
    def validate_entity(self, entity: Dict) -> Tuple[bool, str]:
        """Validate entity"""
        if not entity.get('name'):
            return False, "Missing entity name"
        
        if not entity.get('type'):
            return False, "Missing entity type"
        
        valid_types = ['Person', 'Organization', 'Location', 'Concept', 'Product']
        if entity['type'] not in valid_types:
            return False, f"Invalid entity type: {entity['type']}"
        
        return True, "Valid"
    
    def validate_relationship(self, relationship: Dict) -> Tuple[bool, str]:
        """Validate relationship"""
        if not relationship.get('source') or not relationship.get('target'):
            return False, "Missing source or target"
        
        confidence = relationship.get('confidence', 0.8)
        if confidence < 0.6:
            return False, f"Low confidence: {confidence}"
        
        return True, "Valid"

# ============================================================================
# LANGGRAPH WORKFLOW - MULTI-AGENT ORCHESTRATION
# ============================================================================

class KnowledgeGraphWorkflow:
    """LangGraph workflow for orchestrating agents"""
    
    def __init__(self, llm, graph: nx.Graph, extract_entities=True, extract_relationships=True, validate_data=True):
        self.llm = llm
        self.graph = graph
        self.extract_entities = extract_entities
        self.extract_relationships = extract_relationships
        self.validate_data = validate_data
        
        self.doc_processor = DocumentProcessorAgent()
        self.entity_extractor = EntityExtractorAgent(llm)
        self.relationship_analyzer = RelationshipAnalyzerAgent(llm)
        self.graph_builder = KnowledgeGraphBuilderAgent(graph)
        self.validator = ValidatorAgent()
        
        # Build the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes (agents)
        workflow.add_node("process_documents", self._process_documents_node)
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("validate_entities", self._validate_entities_node)
        workflow.add_node("analyze_relationships", self._analyze_relationships_node)
        workflow.add_node("validate_relationships", self._validate_relationships_node)
        workflow.add_node("build_graph", self._build_graph_node)
        workflow.add_node("create_vector_store", self._create_vector_store_node)
        
        # Define edges (workflow)
        workflow.set_entry_point("process_documents")
        workflow.add_edge("process_documents", "extract_entities")
        workflow.add_edge("extract_entities", "validate_entities")
        workflow.add_edge("validate_entities", "analyze_relationships")
        workflow.add_edge("analyze_relationships", "validate_relationships")
        workflow.add_edge("validate_relationships", "build_graph")
        workflow.add_edge("build_graph", "create_vector_store")
        workflow.add_edge("create_vector_store", END)
        
        return workflow.compile()
    
    # Node functions (each represents an agent's action)
    
    def _process_documents_node(self, state: GraphState) -> GraphState:
        """Agent 1: Document Processor"""
        try:
            text_chunks = []
            for doc in state['documents']:
                text_chunks.append(doc.page_content)
            
            state['text_chunks'] = text_chunks
            state['processing_log'].append(f"✓ Processed {len(text_chunks)} document chunks")
            
        except Exception as e:
            state['error'] = f"Document processing error: {e}"
        
        return state
    
    def _extract_entities_node(self, state: GraphState) -> GraphState:
        """Agent 2: Entity Extractor"""
        try:
            all_entities = []
            
            for chunk in state['text_chunks'][:10]:  # Process first 10 chunks
                entities = self.entity_extractor.extract_entities(chunk)
                all_entities.extend(entities)
            
            state['entities'] = all_entities
            state['processing_log'].append(f"✓ Extracted {len(all_entities)} entities")
            
        except Exception as e:
            state['error'] = f"Entity extraction error: {e}"
        
        return state
    
    def _validate_entities_node(self, state: GraphState) -> GraphState:
        """Agent 6: Validator (for entities)"""
        try:
            validated = []
            
            for entity in state['entities']:
                is_valid, msg = self.validator.validate_entity(entity)
                if is_valid:
                    validated.append(entity)
            
            state['validated_entities'] = validated
            state['processing_log'].append(f"✓ Validated {len(validated)}/{len(state['entities'])} entities")
            
        except Exception as e:
            state['error'] = f"Entity validation error: {e}"
        
        return state
    
    def _analyze_relationships_node(self, state: GraphState) -> GraphState:
        """Agent 3: Relationship Analyzer"""
        try:
            all_relationships = []
            
            # Process relationships for each chunk with entities
            for i, chunk in enumerate(state['text_chunks'][:10]):
                if i < len(state['validated_entities']):
                    # Get entities from this chunk
                    chunk_entities = state['validated_entities'][:min(10, len(state['validated_entities']))]
                    
                    if len(chunk_entities) >= 2:
                        relationships = self.relationship_analyzer.analyze_relationships(chunk, chunk_entities)
                        all_relationships.extend(relationships)
            
            state['relationships'] = all_relationships
            state['processing_log'].append(f"✓ Found {len(all_relationships)} relationships")
            
        except Exception as e:
            state['error'] = f"Relationship analysis error: {e}"
        
        return state
    
    def _validate_relationships_node(self, state: GraphState) -> GraphState:
        """Agent 6: Validator (for relationships)"""
        try:
            validated = []
            
            for relationship in state['relationships']:
                is_valid, msg = self.validator.validate_relationship(relationship)
                if is_valid:
                    validated.append(relationship)
            
            state['validated_relationships'] = validated
            state['processing_log'].append(f"✓ Validated {len(validated)}/{len(state['relationships'])} relationships")
            
        except Exception as e:
            state['error'] = f"Relationship validation error: {e}"
        
        return state
    
    def _build_graph_node(self, state: GraphState) -> GraphState:
        """Agent 4: Knowledge Graph Builder"""
        try:
            # Add entities
            for entity in state['validated_entities']:
                self.graph_builder.add_entity(entity)
            
            # Add relationships
            for relationship in state['validated_relationships']:
                self.graph_builder.add_relationship(relationship)
            
            state['graph'] = self.graph
            stats = self.graph_builder.get_stats()
            state['processing_log'].append(f"✓ Built graph: {stats['nodes']} nodes, {stats['edges']} edges")
            
        except Exception as e:
            state['error'] = f"Graph building error: {e}"
        
        return state
    
    def _create_vector_store_node(self, state: GraphState) -> GraphState:
        """Create vector store from documents"""
        try:
            if state['documents']:
                embeddings = get_embeddings()
                vector_store = FAISS.from_documents(state['documents'], embeddings)
                state['vector_store'] = vector_store
                state['processing_log'].append(f"✓ Created vector store with {len(state['documents'])} documents")
        except Exception as e:
            state['error'] = f"Vector store error: {e}"
        
        return state
    
    def run(self, documents: List[Document]) -> GraphState:
        """Execute the workflow"""
        
        # Initialize state
        initial_state = GraphState(
            documents=documents,
            text_chunks=[],
            entities=[],
            relationships=[],
            validated_entities=[],
            validated_relationships=[],
            graph=self.graph,
            vector_store=None,
            processing_log=[],
            error=""
        )
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state

# ============================================================================
# GRAPH VISUALIZATION
# ============================================================================

def visualize_graph(graph: nx.Graph, height="600px"):
    """Visualize knowledge graph using Pyvis"""
    
    if graph.number_of_nodes() == 0:
        st.warning("Graph is empty. Add some documents first!")
        return
    
    # Create Pyvis network
    net = Network(height=height, width="100%", bgcolor="#222222", font_color="white")
    
    # Color mapping for node types
    colors = {
        'Person': '#FF6B6B',
        'Organization': '#4ECDC4',
        'Location': '#45B7D1',
        'Concept': '#FFA07A',
        'Product': '#98D8C8',
        'Unknown': '#CCCCCC'
    }
    
    # Add nodes
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        color = colors.get(node_type, '#CCCCCC')
        
        title = f"Type: {node_type}"
        for key, value in data.items():
            if key != 'type':
                title += f"<br>{key}: {value}"
        
        net.add_node(node, label=node, color=color, title=title, size=25)
    
    # Add edges
    for source, target, data in graph.edges(data=True):
        relation = data.get('relation', 'RELATED_TO')
        confidence = data.get('confidence', 0.8)
        
        net.add_edge(source, target, title=f"{relation} ({confidence:.2f})", 
                    label=relation, color="#888888")
    
    # Physics settings
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    # Save and display
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            temp_path = f.name
            net.save_graph(temp_path)
        
        # Read and display
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Display with larger height
        st.components.v1.html(html_content, height=700, scrolling=True)
        
    except Exception as e:
        st.error(f"Visualization error: {e}")
        st.info("Try using a smaller graph or refresh the page.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🧠 Enterprise Knowledge Graph Agent")
    st.markdown("### Multi-Agent AI System for Knowledge Graph Construction & Querying")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Graph Statistics")
        
        if st.session_state.knowledge_graph.number_of_nodes() > 0:
            builder = KnowledgeGraphBuilderAgent(st.session_state.knowledge_graph)
            stats = builder.get_stats()
            
            st.metric("Nodes", stats['nodes'])
            st.metric("Edges", stats['edges'])
            
            st.subheader("Node Types")
            for node_type, count in stats['node_types'].items():
                st.write(f"- {node_type}: {count}")
            
            st.subheader("Relationship Types")
            for relation, count in stats['relation_types'].items():
                st.write(f"- {relation}: {count}")
        else:
            st.info("No graph data yet")
        
        st.markdown("---")
        
        # Save/Load Graph
        st.subheader("💾 Persistence")
        
        if st.button("Save Graph"):
            try:
                with open("knowledge_graph.pkl", 'wb') as f:
                    pickle.dump(st.session_state.knowledge_graph, f)
                st.success("Graph saved!")
            except Exception as e:
                st.error(f"Error saving: {e}")
        
        if st.button("Load Graph"):
            try:
                with open("knowledge_graph.pkl", 'rb') as f:
                    st.session_state.knowledge_graph = pickle.load(f)
                st.success("Graph loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading: {e}")
        
        if st.button("Clear Graph", type="secondary"):
            st.session_state.knowledge_graph = nx.Graph()
            st.session_state.vector_store = None
            st.session_state.documents = []
            st.success("Graph cleared!")
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Build Graph", "🔍 Query Graph", "📊 Visualize", "📝 Logs"])
    
    # TAB 1: BUILD GRAPH
    with tab1:
        st.header("Build Knowledge Graph from Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_files = st.file_uploader(
                "Upload Documents (PDF, TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True
            )
            
            # Text input
            text_input = st.text_area("Or paste text here:", height=150)
        
        with col2:
            st.markdown("**Processing Options**")
            extract_entities = st.checkbox("Extract Entities", value=True)
            extract_relationships = st.checkbox("Extract Relationships", value=True)
            validate_data = st.checkbox("Validate Extractions", value=True)
        
        if st.button("🚀 Process & Build Graph", type="primary"):
            llm = get_llm()
            
            # Initialize document processor to collect input documents
            doc_processor = DocumentProcessorAgent()
            all_documents = []
            
            with st.spinner("Processing documents..."):
                # Process uploaded files
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        st.info(f"Processing: {uploaded_file.name}")
                        docs = doc_processor.process_uploaded_file(uploaded_file)
                        all_documents.extend(docs)
                        st.session_state.processing_log.append(f"Processed {uploaded_file.name}: {len(docs)} chunks")
                
                # Process text input
                if text_input.strip():
                    docs = doc_processor.process_text(text_input, {'source': 'text_input'})
                    all_documents.extend(docs)
                    st.session_state.processing_log.append(f"Processed text input: {len(docs)} chunks")
                
                st.success(f"Processed {len(all_documents)} document chunks")
            
            if all_documents:
                # Create and run the LangGraph workflow
                with st.spinner("Running multi-agent workflow..."):
                    workflow = KnowledgeGraphWorkflow(
                        llm=llm,
                        graph=st.session_state.knowledge_graph,
                        extract_entities=extract_entities,
                        extract_relationships=extract_relationships,
                        validate_data=validate_data
                    )
                    
                    # Run the workflow with progress tracking
                    progress_bar = st.progress(0)
                    st.info("🤖 Agent Pipeline: Document Processing → Entity Extraction → Validation → Relationship Analysis → Validation → Graph Building → Vector Store Creation")
                    
                    # Execute the workflow
                    final_state = workflow.run(all_documents)
                    
                    # Store workflow state in session
                    st.session_state.workflow_state = final_state
                    
                    # Update session state with results
                    if final_state.get("vector_store"):
                        st.session_state.vector_store = final_state["vector_store"]
                    
                    # Log processing details
                    for log_entry in final_state.get("processing_log", []):
                        st.session_state.processing_log.append(log_entry)
                    
                    progress_bar.progress(1.0)
                    
                    if final_state.get("error"):
                        st.error(f"Workflow error: {final_state['error']}")
                    else:
                        st.success("✅ Knowledge graph built successfully using LangGraph workflow!")
                        st.session_state.processing_log.append(f"Built graph: {st.session_state.knowledge_graph.number_of_nodes()} nodes, {st.session_state.knowledge_graph.number_of_edges()} edges")
                        st.rerun()
            else:
                st.warning("No documents to process. Please upload files or enter text.")
    
    # TAB 2: QUERY GRAPH
    with tab2:
        st.header("Query Knowledge Graph")
        
        if st.session_state.knowledge_graph.number_of_nodes() == 0:
            st.warning("⚠️ Graph is empty. Please build a graph first in the 'Build Graph' tab.")
        else:
            question = st.text_input("Ask a question:", placeholder="e.g., Who works for Microsoft?")
            
            if st.button("🔍 Search", type="primary") and question:
                with st.spinner("Searching knowledge graph..."):
                    llm = get_llm()
                    query_agent = QueryAgent(llm, st.session_state.knowledge_graph, st.session_state.vector_store)
                    answer = query_agent.query(question)
                    
                    st.markdown("### Answer")
                    st.markdown(answer)
            
            # Example queries
            st.markdown("---")
            st.markdown("**Example Questions:**")
            st.markdown("- What entities are in the graph?")
            st.markdown("- Who is related to [entity name]?")
            st.markdown("- What organizations are mentioned?")
            st.markdown("- Summarize the key concepts")
    
    # TAB 3: VISUALIZE
    with tab3:
        st.header("Knowledge Graph Visualization")
        
        if st.session_state.knowledge_graph.number_of_nodes() == 0:
            st.warning("⚠️ Graph is empty. Please build a graph first.")
        else:
            # Visualization options
            viz_option = st.radio("Visualization Type:", ["Interactive (Pyvis)", "Simple List View"], horizontal=True)
            
            if viz_option == "Interactive (Pyvis)":
                st.markdown("**Interactive Graph** (drag nodes, zoom in/out)")
                with st.spinner("Rendering graph visualization..."):
                    visualize_graph(st.session_state.knowledge_graph)
            else:
                # Simple list view
                st.markdown("### Entities")
                graph = st.session_state.knowledge_graph
                
                # Group by type
                entities_by_type = {}
                for node, data in graph.nodes(data=True):
                    node_type = data.get('type', 'Unknown')
                    if node_type not in entities_by_type:
                        entities_by_type[node_type] = []
                    entities_by_type[node_type].append(node)
                
                for entity_type, nodes in entities_by_type.items():
                    with st.expander(f"{entity_type} ({len(nodes)})", expanded=False):
                        for node in sorted(nodes)[:20]:  # Show first 20
                            st.write(f"• {node}")
                
                st.markdown("### Sample Relationships")
                edge_count = 0
                for source, target, data in graph.edges(data=True):
                    if edge_count >= 20:
                        break
                    relation = data.get('relation', 'RELATED_TO')
                    st.write(f"• {source} **{relation}** {target}")
                    edge_count += 1
            
            # Graph export options
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export as JSON"):
                    graph_data = nx.node_link_data(st.session_state.knowledge_graph)
                    json_str = json.dumps(graph_data, indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "knowledge_graph.json",
                        "application/json"
                    )
            
            with col2:
                if st.button("Export as GraphML"):
                    import io
                    buffer = io.BytesIO()
                    # Write to string first
                    graphml_str = "\n".join(nx.generate_graphml(st.session_state.knowledge_graph))
                    buffer.write(graphml_str.encode())
                    buffer.seek(0)
                    st.download_button(
                        "Download GraphML",
                        buffer,
                        "knowledge_graph.graphml",
                        "application/xml"
                    )
    
    # TAB 4: LOGS
    with tab4:
        st.header("Processing Logs & Workflow State")
        
        # Workflow State Section
        if st.session_state.workflow_state:
            st.subheader("🔄 LangGraph Workflow State")
            
            workflow_state = st.session_state.workflow_state
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Processed", len(workflow_state.get("documents", [])))
            with col2:
                st.metric("Entities Validated", len(workflow_state.get("validated_entities", [])))
            with col3:
                st.metric("Relationships Validated", len(workflow_state.get("validated_relationships", [])))
            
            # Show workflow execution path
            with st.expander("📊 Workflow Execution Details"):
                st.markdown("**Agent Pipeline:**")
                st.markdown("""
                1. ✅ Document Processing Agent
                2. ✅ Entity Extraction Agent
                3. ✅ Entity Validation Agent
                4. ✅ Relationship Analysis Agent
                5. ✅ Relationship Validation Agent
                6. ✅ Knowledge Graph Builder Agent
                7. ✅ Vector Store Creation
                """)
                
                if workflow_state.get("processing_log"):
                    st.markdown("**Processing Steps:**")
                    for log in workflow_state["processing_log"]:
                        st.text(f"  → {log}")
            
            st.markdown("---")
        
        # Processing Logs Section
        st.subheader("📝 Processing Logs")
        if st.session_state.processing_log:
            for i, log in enumerate(reversed(st.session_state.processing_log), 1):
                st.text(f"{i}. {log}")
        else:
            st.info("No processing logs yet")
        
        if st.button("Clear Logs"):
            st.session_state.processing_log = []
            st.rerun()

if __name__ == "__main__":
    main()
