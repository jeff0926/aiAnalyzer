# AI Analyzer

A comprehensive GitHub repository analysis system that leverages both local and cloud LLMs to generate detailed insights, knowledge graphs, and analysis reports.

## 🚀 Features

- **Stream-based Repository Analysis**: Real-time processing of repository contents
- **Multi-Agent Architecture**: Specialized agents for different file types
- **Hybrid LLM Integration**: Both local and cloud LLM support
- **Interactive Knowledge Graph**: 3D visualization of repository structure
- **Comprehensive Reports**: Including ARB packages, security analysis, and performance metrics

## 🛠️ Technical Stack

- **Language**: Python 3.9+
- **Local LLMs**: LlamaCpp with CodeLlama-34b and Llama-2-70b
- **Cloud LLMs**: OpenAI GPT-4, Anthropic Claude, Azure OpenAI
- **Visualization**: Three.js
- **Testing**: pytest

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/jeff0926/aiAnalyzer.git
cd aiAnalyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
/core               # Core system functionality
  /llm             # LLM integration
  analyzer.py      # Main analysis engine
  repo_cloner.py   # Repository handling
  aggregator.py    # Results aggregation
/agents            # Specialized analysis agents
/graph             # Knowledge graph implementation
/ui                # Web interface
/utils             # Utility functions
/tests             # Test suite
```

## 📊 Usage

```python
from core.analyzer import RepositoryAnalyzer

# Initialize analyzer
analyzer = RepositoryAnalyzer(repo_url="https://github.com/user/repo")

# Run analysis
results = analyzer.analyze()

# Generate reports
analyzer.generate_reports()
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_analyzer.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.