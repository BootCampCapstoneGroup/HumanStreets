# HumanStreets System Architecture

This diagram illustrates the moving parts of the HumanStreets application, showing how the Frontend, Backend, Agents, and Database interact.

```mermaid
graph TD
    subgraph "Frontend (React)"
        App["App.jsx<br/>(State Manager)"] --> Map["MapContainer.jsx<br/>(MapLibre/DeckGL)"]
        App --> Chat["ChatSidebar.jsx<br/>(UI & Logic)"]
    end

    subgraph "Backend (FastAPI)"
        API["API Endpoints<br/>(/chat, /layers)"] --> Router["Router Agent<br/>(Orchestrator)"]
        
        Router -->|Simple Queries| SQL["SQL Agent<br/>(PostGIS Expert)"]
        Router -->|Complex Plans| Planner["Planner Agent<br/>(Strategies)"]
        Router -->|Visuals| Viz["Viz Agent<br/>(Map Styles)"]
        
        SQL --> LLM["LLM Service<br/>(Gemini/Local)"]
        SQL --> Spatial["Spatial Service<br/>(GeoPandas)"]
    end

    subgraph "Data Layer"
        Spatial --> DB[("PostgreSQL + PostGIS")]
        DB -- "Neighborhoods<br/>H3 Grid" --> Spatial
    end

    %% Interactions
    Chat -- "User Query" --> API
    API -- "Stream Response" --> Chat
    Spatial -- "GeoJSON" --> API
    Map -- "Rendering" --> Chat
```

## Component Overview

1.  **Frontend**:
    *   **App.jsx**: Manages global state (theme, messages, map layers-to-show).
    *   **MapContainer**: Renders the interactive map using MapLibre (base) and DeckGL (data layers).
    *   **ChatSidebar**: Handles user input and displays streaming responses.

2.  **Backend**:
    *   **Router Agent**: Decides which specialized agent handles the request (e.g., "Show Malaz" -> SQL Agent).
    *   **SQL Agent**: Converts natural language to PostGIS SQL, executes it, and returns GeoJSON.
    *   **Planner Agent**: Breaks down complex tasks (e.g., "Find best place for...") into steps.

3.  **Data**:
    *   **PostgreSQL/PostGIS**: Stores spatial data (neighborhood bounds, walkability scores, H3 hexagons).
