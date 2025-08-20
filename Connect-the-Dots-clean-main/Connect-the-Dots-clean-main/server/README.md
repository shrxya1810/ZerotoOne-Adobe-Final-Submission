# Challenge 1a: PDF Processing Solution – Submission

## 📄 Overview

This repository contains a solution for **Challenge 1a** of the **Adobe India Hackathon 2025**. The goal is to automatically extract structured data (titles and section headings) from PDF files and export it as JSON. The solution is fully containerized with Docker and adheres to strict resource, performance, and architectural constraints.

---

## 🚀 Features

- 📘 Title extraction using multi-strategy ensemble (font size, spacing, center alignment)
- 🧠 Hierarchical heading detection via spatial and sequential analysis
- 🔍 Robust filtering for false positives (form fields, TOC, metadata)
- ✅ Outputs conform to a defined JSON schema
- 🐳 Docker-compatible, isolated, and offline-executable

---

## 🧱 Project Structure

```
Submission/
├── input/                  # Input PDF files
│   ├── file01.pdf
│   ├── file02.pdf
│   ├── file03.pdf
│   ├── file04.pdf
│   ├── file05.pdf
│   └── file06.pdf
├── output/                 # Output JSON files
│   ├── file01.json
│   ├── file02.json
│   ├── file03.json
│   ├── file04.json
│   ├── file05.json
│   └── file06.json
├── generate_outputs.py     # Main entry point
├── pdf_extractor.py        # Title + heading extraction logic
├── hierarchy_enhancer.py   # Hierarchical organization and enhancement
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container specification
└── README.md              # This file
```

---

## 🎯 Approach

### Title Extraction Strategy
The solution employs a **5-strategy ensemble approach** for robust title detection:

1. **Multiline Title Detection** - Identifies titles spanning multiple lines with consistent formatting
2. **Absolute Largest Font Analysis** - Finds the most prominent text by font size
3. **Pattern-Based Detection** - Uses regex patterns for common title formats
4. **Spatial Context Analysis** - Considers positioning and surrounding whitespace
5. **Center-Aligned Prominent Text** - Detects centered, visually prominent text

### Heading Detection Strategy
The heading extraction uses **50+ heuristics** across multiple dimensions:

- **Typography Analysis**: Font size, weight, family, and formatting
- **Spatial Analysis**: Positioning, whitespace, and layout context
- **Content Analysis**: Text patterns, numbering, and semantic indicators
- **Sequential Analysis**: Relationship between consecutive headings
- **False Positive Filtering**: Removes form fields, TOC entries, metadata

### Hierarchy Enhancement
The `HierarchyEnhancer` class provides:
- **Spatial Context Detection**: Identifies headers based on surrounding body text
- **Sequential Group Analysis**: Groups related headings by proximity
- **Conflict Resolution**: Resolves overlapping or conflicting heading candidates
- **Level Assignment**: Assigns H1-H3 levels based on visual hierarchy

---

## 📦 Dependencies & Libraries

### Core Libraries
- **pdfplumber==0.10.0** - Low-level PDF character and layout analysis
- **pandas==2.2.0** - Data manipulation and analysis
- **numpy==1.26.3** - Numerical computations

### Native Python Modules
- **re** - Regular expressions for pattern matching
- **typing** - Type hints for code clarity
- **collections** - Data structures for efficient processing
- **pathlib** - File path handling
- **json** - JSON serialization

### No External Models
The solution uses **pure heuristics and layout analysis** - no machine learning models, APIs, or external services are required.

---

## 🐳 Docker Usage

### 🔨 Build the Image

```bash
docker build --platform=linux/amd64 -t pdf-processor:challenge1a .
```

### 📁 Prepare Input & Output Directories

```bash
# Create output directory (input directory should already exist with PDFs)
mkdir -p ./output/
```

### 🚀 Run the Container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-processor:challenge1a
```

**For Windows PowerShell:**
```powershell
docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output --network none pdf-processor:challenge1a
```

---

## ✅ Output Format

Each output JSON file (e.g., `filename.json`) contains:

```json
{
  "title": "Document Title Here",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Background", "page": 2},
    {"level": "H3", "text": "Technical Details", "page": 3}
  ]
}
```

**Schema Details:**
- `title`: String - The extracted document title
- `outline`: Array of heading objects
  - `level`: String - Heading level ("H1", "H2", "H3")
  - `text`: String - Heading text content
  - `page`: Integer - Page number (0-indexed)

---

## ⚙️ How It Works

1. **Input Processing**: Scans all `.pdf` files in `/app/input`
2. **Title Extraction**: For each PDF, applies 5-strategy ensemble to find the document title
3. **Heading Detection**: Identifies section headings using 50+ heuristics
4. **Hierarchy Enhancement**: Applies spatial and sequential analysis for better structure
5. **Output Generation**: Creates JSON file per PDF in `/app/output`

### Processing Pipeline
```
PDF Input → Character Extraction → Line Analysis → Title Detection → Heading Detection → Hierarchy Enhancement → JSON Output
```

---

## ⏱️ Performance Targets

| Constraint                  | Status ✅ |
|----------------------------|-----------|
| ≤ 10 sec for 50-page PDF   | ✅        |
| ≤ 200MB model size         | ✅        |
| No internet access         | ✅        |
| CPU-only, 8-core           | ✅        |
| ≤ 16 GB memory usage       | ✅        |
| Must run on AMD64          | ✅        |

---

## 🧪 Testing Checklist

- [x] All PDFs in input directory are processed
- [x] JSON outputs generated per PDF
- [x] Output conforms to schema
- [x] Offline-compatible (`--network none`)
- [x] Output within time and memory limits
- [x] Docker image builds and runs on AMD64
- [x] Working Dockerfile in root directory
- [x] All dependencies installed within container

---

## 📎 Technical Notes

- **No Third-Party Models**: Everything is based on heuristics and layout analysis
- **Fully Open Source**: All dependencies listed in `requirements.txt`
- **Offline Execution**: No internet access required during processing
- **Resource Efficient**: Optimized for CPU-only execution with minimal memory footprint
- **Cross-Platform**: Docker container ensures consistent execution across environments

---

## 🔧 Troubleshooting

### Common Issues
1. **Docker not found**: Ensure Docker Desktop is installed and running
2. **Permission errors**: Run Docker commands with appropriate permissions
3. **Volume mounting issues**: Ensure input/output directories exist and are accessible

### Verification Commands
```bash
# Check if image builds successfully
docker build --platform=linux/amd64 -t pdf-processor:test .

# Verify container runs
docker run --rm pdf-processor:test --help
```
