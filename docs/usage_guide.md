# WebSim Usage Guide

This guide provides detailed instructions on how to use WebSim for web page similarity analysis.

## Command Line Interface

WebSim provides three main command-line tools:

### 1. Indexing URLs

The `index_urls.py` script allows you to create a feature index of web pages for later comparison.

```bash
./index_urls.py <url1> <url2> ... <urlN>
```

**Examples:**

Index multiple websites:
```bash
./index_urls.py https://www.example.com https://www.github.com https://www.wikipedia.org
```

**Options:**
- `--help`: Show help message

### 2. Querying Similarities

The `query_similarity.py` script compares a URL against the index to find similar pages.

```bash
./query_similarity.py <url>
```

**Examples:**

Find pages similar to BBC:
```bash
./query_similarity.py https://www.bbc.com
```

**Output:**
```
Score: 0.9900 - URL: https://www.bbc.com (QUERY URL)
Score: 0.8759 - URL: https://www.twitter.com
Score: 0.8524 - URL: https://www.amazon.com
```

**Options:**
- `--help`: Show help message

### 3. Direct URL Comparison

The `compare_urls.py` script allows you to directly compare two URLs without using the index.

```bash
./compare_urls.py <url1> <url2>
```

**Examples:**

Compare BBC and CNN:
```bash
./compare_urls.py https://www.bbc.com https://www.cnn.com
```

**Output:**
```
Similarity between URLs: 0.7842
```

**Options:**
- `--help`: Show help message

## Python API

WebSim can also be used directly from Python code.

### Indexing URLs

```python
from websim.indexer import WebIndexer

# Create indexer
indexer = WebIndexer()

# Index multiple URLs
urls = [
    'https://www.example.com',
    'https://www.github.com',
    'https://www.wikipedia.org'
]
indexer.index_urls(urls)

# Save the index
indexer.save_index()

# Get index information
info = indexer.get_index_info()
print(f"Index contains {info['count']} URLs")
```

### Querying for Similar Pages

```python
from websim.query import SimilarityQuery

# Create query handler
query = SimilarityQuery()

# Find similar pages
results = query.find_similar('https://www.example.org')

# Print results
for url, score in results:
    print(f"Score: {score:.4f} - URL: {url}")
```

### Comparing Two URLs Directly

```python
from websim.compare import URLComparer

# Create comparer
comparer = URLComparer()

# Compare two URLs
similarity = comparer.compare_urls('https://www.example.com', 'https://www.example.org')
print(f"Similarity score: {similarity:.4f}")
```

## Advanced Configuration

You can configure WebSim by modifying the parameters in `websim/config.py`:

```python
# Example configuration options
FEATURE_INDEX_DIR = "feature_index"  # Directory for storing feature index
TARGET_SIZE = (1024, 1024)  # Target size for screenshot processing
PAGE_LOAD_TIMEOUT = 5  # Page load timeout in seconds
HEADLESS = True  # Run WebDriver in headless mode
```

## Troubleshooting

### Common Issues

**Issue**: WebDriver cannot render page completely
**Solution**: Increase the `PAGE_LOAD_TIMEOUT` in config.py

**Issue**: Feature extraction fails for JavaScript-heavy pages
**Solution**: Try disabling headless mode by setting `HEADLESS = False` in config.py

**Issue**: "ChromeDriver not found" error
**Solution**: The package should automatically download the appropriate ChromeDriver version, but you can manually install it if needed.

### Getting Help

If you encounter any issues or have questions, please:

1. Check the project's GitHub Issues page
2. Review this documentation thoroughly
3. Open a new issue if your problem hasn't been addressed 