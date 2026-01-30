# LangGraph StateGraph Workflow - Knowledge Graph Agent

## Overview
This document describes the LangGraph StateGraph workflow implementation for the Enterprise Knowledge Graph multi-agent system.

## Architecture

### State Definition
The workflow uses a `GraphState` TypedDict with the following fields:

```python
class GraphState(TypedDict):
    documents: List          # Input documents from user
    text_chunks: List        # Processed text chunks
    entities: List           # Extracted entities
    relationships: List      # Extracted relationships
    validated_entities: List # Entities after validation
    validated_relationships: List  # Relationships after validation
    graph: Optional[nx.Graph]      # Knowledge graph
    vector_store: Optional[Any]    # FAISS vector store
    processing_log: List           # Execution logs
    error: Optional[str]           # Error tracking
```

## Agent Pipeline (StateGraph Nodes)

The workflow consists of 7 sequential nodes, each representing a specialized agent:

### 1. Document Processing Agent
- **Node**: `process_documents`
- **Input**: Raw documents
- **Output**: Text chunks
- **Function**: Splits documents into processable chunks

### 2. Entity Extraction Agent
- **Node**: `extract_entities`
- **Input**: Text chunks
- **Output**: Raw entities (name, type, properties)
- **Function**: Uses LLM to extract entities from text

### 3. Entity Validation Agent
- **Node**: `validate_entities`
- **Input**: Raw entities
- **Output**: Validated entities
- **Function**: Validates entity format, removes duplicates

### 4. Relationship Analysis Agent
- **Node**: `analyze_relationships`
- **Input**: Text chunks + validated entities
- **Output**: Raw relationships (source, target, type, properties)
- **Function**: Uses LLM to identify relationships between entities

### 5. Relationship Validation Agent
- **Node**: `validate_relationships`
- **Input**: Raw relationships
- **Output**: Validated relationships
- **Function**: Validates relationship format, ensures nodes exist

### 6. Knowledge Graph Builder Agent
- **Node**: `build_graph`
- **Input**: Validated entities + relationships
- **Output**: NetworkX graph
- **Function**: Constructs graph structure

### 7. Vector Store Creation
- **Node**: `create_vector_store`
- **Input**: Documents
- **Output**: FAISS vector store
- **Function**: Creates embeddings for semantic search

## Workflow Execution Flow

```
┌─────────────────────┐
│   User Input        │
│ (Files/Text)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Process Documents   │ ← Agent 1: DocumentProcessorAgent
│  (split chunks)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Extract Entities    │ ← Agent 2: EntityExtractorAgent (LLM)
│  (LLM analysis)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate Entities   │ ← Agent 3: ValidatorAgent
│  (check format)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Analyze Relations   │ ← Agent 4: RelationshipAnalyzerAgent (LLM)
│  (LLM analysis)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate Relations  │ ← Agent 5: ValidatorAgent
│  (check format)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Build Graph         │ ← Agent 6: KnowledgeGraphBuilderAgent
│  (NetworkX)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Create Vector Store │ ← Vector Store Creation
│  (FAISS)            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Final State       │
│ (Graph + Vectors)   │
└─────────────────────┘
```

## Usage Example

```python
# Initialize workflow
workflow = KnowledgeGraphWorkflow(
    llm=llm,
    graph=knowledge_graph,
    extract_entities=True,
    extract_relationships=True,
    validate_data=True
)

# Run workflow
final_state = workflow.run(documents)

# Access results
graph = final_state['graph']
vector_store = final_state['vector_store']
logs = final_state['processing_log']
```

## Benefits of StateGraph Orchestration

1. **State Management**: Centralized state tracking across all agents
2. **Error Handling**: Automatic error propagation and tracking
3. **Observability**: Built-in logging and state inspection
4. **Modularity**: Each agent is an independent node
5. **Scalability**: Easy to add/remove agents or modify workflow
6. **Deterministic**: Clear execution path from start to end

## LangGraph Dependencies

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
```

## Integration Points

- **Streamlit UI**: "Process & Build Graph" button triggers workflow
- **Session State**: `st.session_state.workflow_state` stores final state
- **Logs Tab**: Displays workflow execution details and agent pipeline

## Error Handling

Errors are captured in the `error` field of GraphState and propagated through the workflow. Processing continues but errors are logged for debugging.

## Future Enhancements

- Conditional edges based on validation results
- Parallel entity/relationship extraction for performance
- Human-in-the-loop validation nodes
- Dynamic routing based on document type
- Streaming updates during execution
