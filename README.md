# WebSim - Web Page Similarity Analysis Tool

WebSim is a sophisticated tool for comparing and analyzing web page similarities based on visual and structural features.

## Overview

WebSim provides three main functionalities:
1. **Indexing URLs**: Create a feature index of web pages for later comparison
2. **Querying**: Compare a URL against the index to find similar pages
3. **Direct Comparison**: Compare two URLs directly

## Features

- Chrome WebDriver integration for rendering web pages
- Visual feature extraction from screenshots
- DOM structure analysis
- Element-based feature extraction
- Similarity scoring with cosine similarity

## Installation

### Prerequisites

- Python 3.8+
- Chrome browser installed
- ChromeDriver (automatically managed by the package)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/websim.git
   cd websim
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

WebSim can be used either through the command-line interface or by importing the package in your Python code.

### Command-line Interface

#### Indexing URLs

```bash
python index_urls.py <url1> <url2> ... <urlN>
# or 
./index_urls.py <url1> <url2> ... <urlN>
```

#### Querying a URL against the index

```bash
python query_similarity.py <url>
# or
./query_similarity.py <url>
```

#### Comparing two URLs directly

```bash
python compare_urls.py <url1> <url2>
# or
./compare_urls.py <url1> <url2>
```

### Python API

```python
from websim.indexer import WebIndexer
from websim.query import SimilarityQuery
from websim.compare import URLComparer

# Index URLs
indexer = WebIndexer()
indexer.index_urls(['https://example.com', 'https://google.com'])
indexer.save_index()

# Query a URL against the index
query = SimilarityQuery()
results = query.find_similar('https://example.org')
for url, score in results:
    print(f"Score: {score:.4f} - URL: {url}")

# Compare two URLs directly
comparer = URLComparer()
similarity = comparer.compare_urls('https://example.com', 'https://example.org')
print(f"Similarity score: {similarity:.4f}")
```

## Advanced Configuration

You can modify the configuration settings in `websim/config.py` to adjust:

- Feature extraction parameters
- Rendering settings
- Similarity thresholds
- Index storage location

## How It Works

1. **Page Processing**:
   - Renders the web page using Chrome WebDriver
   - Extracts DOM structure and elements
   - Takes a screenshot of the page

2. **Feature Extraction**:
   - Processes the page screenshot for visual features
   - Analyzes DOM structure for architectural features
   - Extracts text content and semantic information

3. **Similarity Calculation**:
   - Computes feature vectors for pages
   - Uses cosine similarity to compare vectors

## Examples

### Finding Similar Pages
```bash
python query_similarity.py https://www.bbc.com
```

Sample output:
```
Score: 0.9900 - URL: https://www.bbc.com (QUERY URL)
Score: 0.8759 - URL: https://www.twitter.com
Score: 0.8524 - URL: https://www.amazon.com
```

### Comparing News Sites
```bash
python compare_urls.py https://www.bbc.com https://www.cnn.com
```

Sample output:
```
Similarity between URLs: 0.7842
```

## Limitations

- JavaScript-heavy websites might not render completely within the timeout period
- Very dynamic content may lead to different similarity scores over time
- Pages requiring authentication need special handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Selenium WebDriver team
- OpenCV community
- scikit-learn contributors 