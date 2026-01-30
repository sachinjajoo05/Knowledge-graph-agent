# Knowledge Graph Agent - Quick Start Guide

## LangGraph StateGraph Implementation ✅

This application now uses **LangGraph StateGraph** for orchestrating multiple agents in a deterministic workflow.

## Setup

1. **Install dependencies:**
```bash
pip install -r kg_requirements.txt
```

2. **Configure Azure OpenAI:**
   - Set your API key in the sidebar
   - Endpoint: `https://dna-rnd-oai.openai.azure.com/`
   - Deployment: `GPT-5`

## Running the App

```bash
streamlit run knowledge_graph_app.py
```

## Multi-Agent Workflow Architecture

The app uses **7 orchestrated agents** in a StateGraph:

```
Documents → Process → Extract Entities → Validate → 
Analyze Relationships → Validate → Build Graph → Create Vectors
```

### Agents:
1. **DocumentProcessorAgent** - Splits documents into chunks
2. **EntityExtractorAgent** - Extracts entities using LLM
3. **ValidatorAgent** - Validates entities
4. **RelationshipAnalyzerAgent** - Identifies relationships using LLM
5. **ValidatorAgent** - Validates relationships
6. **KnowledgeGraphBuilderAgent** - Builds NetworkX graph
7. **Vector Store Creator** - Creates FAISS embeddings

## Features

### 1. Build Graph Tab
- Upload PDF/TXT files or paste text
- Configure entity/relationship extraction
- **LangGraph workflow** processes documents through agent pipeline
- Real-time progress tracking

### 2. Query Graph Tab
- Ask natural language questions
- Semantic search using FAISS + NetworkX graph traversal
- LLM-powered answers

### 3. Visualize Tab
- Interactive 3D graph visualization (Pyvis)
- Export graph as GraphML

### 4. Logs Tab
- **NEW**: Workflow state inspection
- Agent pipeline execution details
- Processing logs

## LangGraph StateGraph Benefits

✅ **Deterministic Workflow** - Clear execution path  
✅ **State Management** - Centralized state across agents  
✅ **Observability** - Track each agent's output  
✅ **Error Handling** - Automatic error propagation  
✅ **Modularity** - Easy to modify agent pipeline  

## State Structure

```python
GraphState:
  - documents: Input documents
  - text_chunks: Processed chunks
  - entities: Extracted entities
  - validated_entities: Clean entities
  - relationships: Extracted relationships
  - validated_relationships: Clean relationships
  - graph: NetworkX graph
  - vector_store: FAISS embeddings
  - processing_log: Execution logs
  - error: Error tracking
```

## Testing

Use the included `sample_enterprise_document.txt` which contains:
- 100+ entities (people, organizations, locations, concepts)
- 200+ relationships
- Rich enterprise data

Expected output: ~300 nodes, ~400 edges

## Documentation

- **LANGGRAPH_WORKFLOW.md** - Detailed workflow architecture
- **KNOWLEDGE_GRAPH_AGENT_DOCUMENTATION.md** - Complete system design

## Technologies

- **LangChain** - LLM orchestration
- **LangGraph** - StateGraph workflow
- **Azure OpenAI** - GPT-5 (temperature=1)
- **NetworkX** - Graph storage
- **FAISS** - Vector embeddings
- **Streamlit** - UI framework
- **Pyvis** - Graph visualization

## Troubleshooting

**No graph visualization?**
- Check that files are uploaded/text is entered
- Click "Process & Build Graph"
- Check Logs tab for errors

**LLM errors?**
- Verify API key is correct
- Ensure temperature=1 (GPT-5 requirement)
- Check Azure endpoint URL

**Import errors?**
- Run: `pip install -r kg_requirements.txt`
- Ensure Python 3.10+

## API Configuration

Required environment:
- `AZURE_OPENAI_API_KEY` - Your API key
- Endpoint: `https://dna-rnd-oai.openai.azure.com/`
- Deployment: `GPT-5`
- Embedding model: `text-embedding-3-large`
- Temperature: `1` (required for GPT-5)

---

**Enjoy building knowledge graphs with multi-agent orchestration!** 🚀
