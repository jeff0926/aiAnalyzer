# AI Analyzer

A comprehensive system for analyzing repositories, generating insights, and creating reports. AI Analyzer leverages local and cloud-based LLMs, knowledge graphs, and advanced analysis tools to deliver actionable intelligence.

---

## Features

### Core System
- **Analyzer**: Processes repositories to extract insights and metrics.
- **LLM Router**: Intelligent routing for local and cloud-based LLMs.
- **Repository Cloner**: Fetches and prepares repositories for analysis.
- **Aggregator**: Combines and summarizes analysis results.

### Analysis Agents
- **Base Agent**: Provides foundational analysis functionality.
- **Specialized Agents**: Includes Code, Config, Documentation, Test, Infrastructure, Security, Database, and ARB agents for targeted analysis.

### Knowledge Graph
- Builds and manages a graph representation of repositories.
- Supports advanced querying, metrics calculation, and visualization.

### UI Components
- **Dashboard**: User-friendly interface for managing analyses and viewing results.
- **Graph Visualizer**: Interactive 3D knowledge graph.
- **Report Generator**: Creates customizable reports for different audiences (e.g., executive, technical).

### Utilities
- **Cache**: Manages temporary storage of frequently accessed data.
- **Parser**: Handles JSON and XML conversions.
- **File Utils**: Provides file management operations.
- **Prompt Utils**: Manages prompt creation and LLM interaction.

---

## Installation

### Prerequisites
- Python 3.8 or later
- `pip` (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-analyzer.git
   cd ai-analyzer
   pip install -r requirements.txt


### Running the Application
- **To start the AI Analyzer, execute the main script:**
 ```bash
python main.py
 ```
### Running Tests
 ```bash
python -m unittest discover -s tests ```

ai-analyzer/
│
├── core/
│   ├── analyzer.py
│   ├── aggregator.py
│   ├── llm/
│   │   └── llm_router.py
│   └── repo_cloner.py
│
├── agents/
│   ├── base_agent.py
│   ├── code_agent.py
│   ├── config_agent.py
│   ├── doc_agent.py
│   ├── test_agent.py
│   ├── infra_agent.py
│   ├── security_agent.py
│   ├── db_agent.py
│   └── arb_agent.py
│
├── graph/
│   ├── graph_core.py
│   ├── graph_algorithms.py
│   └── graph_store.py
│
├── ui/
│   ├── index.html
│   ├── styles.css
│   ├── main.js
│   ├── graph_viz.js
│   └── report_gen.js
│
├── utils/
│   ├── cache.py
│   ├── parser.py
│   ├── file_utils.py
│   └── prompts.py
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_agents.py
│   ├── test_graph.py
│   └── test_utils.py
│
└── requirements.txt


