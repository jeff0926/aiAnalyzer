# AI Repository Analyzer

An advanced GitHub repository analysis system powered by local and cloud LLMs, featuring dynamic knowledge graph generation and comprehensive analysis reports.

## 🚀 Features

- **Stream-based Repository Analysis**
  - Real-time file processing
  - Multi-threaded analysis
  - Progress tracking

- **Specialized File Analysis Agents**
  - Source code (.py, .js, .java, .cpp, .go, etc.)
  - Configuration files (.yaml, .json, .toml, .env)
  - Documentation (.md, .rst, .txt, .wiki)
  - Build files (Dockerfile, docker-compose.yml)
  - Infrastructure (.tf, .cf, k8s)
  - Test files
  - Database files
  - Security files

- **Advanced LLM Integration**
  - Local LLM support (LlamaCpp)
    - CodeLlama-34b for code analysis
    - Llama-2-70b for general analysis
  - Cloud LLM integration
    - OpenAI GPT-4
    - Anthropic Claude
    - Azure OpenAI
  - Intelligent routing system
  - Privacy-aware processing

- **Interactive Knowledge Graph**
  - Dynamic graph construction
  - 3D visualization with Three.js
  - Advanced relationship mapping
  - Interactive exploration features

- **Comprehensive Output**
  - JSON repository summaries
  - 3D knowledge graphs
  - ARB packages
  - Security reports
  - Performance analysis
  - Metrics dashboard

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jeff0926/aiAnalyzer.git
   cd aiAnalyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

## 📋 Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended for local LLM processing)
- Node.js 16+ (for UI components)
- Git

## 🔧 Configuration

The system can be configured through:
- Environment variables
- Configuration files (config.yaml)
- Command-line arguments

See `config/README.md` for detailed configuration options.

## 🚀 Usage

1. Start the analysis server:
   ```bash
   python -m aianalyzer.server
   ```

2. Run analysis on a repository:
   ```bash
   python -m aianalyzer.cli analyze https://github.com/user/repo
   ```

3. Access the web interface:
   ```
   http://localhost:3000
   ```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

Full documentation is available at [docs/README.md](docs/README.md).

## 🏗️ Architecture

The system is built with a modular architecture:

```
/core             - Core system components
  /llm           - LLM integration
  analyzer.py    - Main analysis logic
  repo_cloner.py - Repository handling
/agents           - Specialized analysis agents
/graph            - Knowledge graph components
/ui               - Web interface
/utils            - Utility functions
/tests            - Test suite
```

## 🙏 Acknowledgments

- [LlamaCpp](https://github.com/ggerganov/llama.cpp)
- [Three.js](https://threejs.org/)
- [Anthropic](https://www.anthropic.com/)
- [OpenAI](https://openai.com/)
