# WebSim Technical Documentation

This document provides technical details about how WebSim works, including its architecture, algorithms, and implementation details.

## Architecture Overview

WebSim is organized into several key components:

```
websim/
├── core/                  # Core functionality
│   ├── page_processor.py  # Web page processing
│   ├── feature_extractor.py  # Feature extraction
│   └── similarity.py      # Similarity calculations
├── utils/                 # Utility functions
│   └── spm.py             # Spatial pyramid matching
├── indexer.py             # URL indexing
├── query.py               # URL querying
├── compare.py             # Direct URL comparison
└── config.py              # Configuration
```

## Page Processing

The `PageProcessor` class in `page_processor.py` is responsible for loading web pages and extracting both visual and structural information.

### Key Components:

1. **WebDriver Setup**:
   - Uses Selenium with Chrome/Chromium WebDriver
   - Automatically manages ChromeDriver versions
   - Configurable viewport size and timeouts

2. **Page Rendering**:
   - Renders pages in headless or visible mode
   - Waits for page content to load
   - Handles JavaScript execution

3. **Element Extraction**:
   - Identifies visible leaf elements
   - Records element positions, dimensions, and types
   - Extracts element attributes and text content

4. **Screenshot Capture**:
   - Takes full-page screenshots
   - Processes images to standard dimensions

## Feature Extraction

The `FeatureExtractor` class in `feature_extractor.py` processes page information to generate feature vectors.

### Feature Types:

1. **Visual Features**:
   - Histogram of Oriented Gradients (HOG) from screenshots
   - Color histograms in RGB or HSV space
   - Edge detection and texture analysis

2. **Structural Features**:
   - Element type distribution (text, image, form, etc.)
   - Element position mapping
   - Hierarchical layout structure

3. **Content Features**:
   - Text density and distribution
   - Image-to-text ratio
   - Form element presence

### Feature Vector Creation:

Features are organized using Spatial Pyramid Matching (SPM), which divides the page into increasingly fine regions at multiple levels. This approach captures both global and local structures.

```python
# Example SPM implementation pseudocode
def create_spm_features(features, levels=3):
    pyramid = []
    
    # Level 0: Global features
    pyramid.append(global_features(features))
    
    # Higher levels: increasingly fine grid
    for level in range(1, levels):
        grid_size = 2**level
        grid_features = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                cell_features = extract_cell_features(features, row, col, grid_size)
                grid_features.append(cell_features)
                
        pyramid.append(grid_features)
        
    return pyramid
```

## Similarity Calculation

The `SimilarityCalculator` class in `similarity.py` computes similarity scores between feature vectors.

### Similarity Metrics:

1. **Cosine Similarity**:
   - Measures angle between feature vectors
   - Scale-invariant, focuses on orientation
   - Primary metric for overall similarity

2. **Euclidean Distance**:
   - Measures absolute distance between vectors
   - Used for specific feature comparisons

3. **Histogram Intersection**:
   - Measures overlap between histogram features
   - Useful for visual pattern matching

### Combined Similarity:

The final similarity score is a weighted combination of individual feature similarities:

```python
def calculate_similarity(features1, features2, weights=None):
    if weights is None:
        weights = {
            'visual': 0.5,
            'structural': 0.3,
            'content': 0.2
        }
    
    visual_sim = cosine_similarity(features1['visual'], features2['visual'])
    structural_sim = cosine_similarity(features1['structural'], features2['structural'])
    content_sim = histogram_intersection(features1['content'], features2['content'])
    
    combined_sim = (
        weights['visual'] * visual_sim +
        weights['structural'] * structural_sim +
        weights['content'] * content_sim
    )
    
    return combined_sim
```

## Index Management

The WebSim index stores feature vectors for each processed URL in the numpy format (.npy files) and maintains a JSON index file with metadata.

### Index Structure:

```
feature_index/
├── features/
│   ├── url_hash_1.npy
│   ├── url_hash_2.npy
│   └── ...
└── index_info.json
```

The `index_info.json` file contains:
- URL to hash mapping
- Timestamp of indexing
- Feature vector metadata
- Configuration parameters used

## Query Processing

When querying for similarity, WebSim:

1. Processes the query URL to extract features
2. Loads pre-computed feature vectors from the index
3. Calculates similarity scores between the query and indexed URLs
4. Sorts and returns results in descending order of similarity

## Performance Considerations

- **Memory Usage**: Feature vectors can be large, especially with multiple SPM levels
- **Processing Time**: Page rendering and feature extraction are the most time-consuming operations
- **Storage**: Index size scales linearly with the number of indexed URLs

## Configuration Parameters

Key parameters in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TARGET_SIZE` | Screenshot processing size | (1024, 1024) |
| `SPM_LEVELS` | Spatial pyramid levels | 3 |
| `PAGE_LOAD_TIMEOUT` | Seconds to wait for page load | 5 |
| `FEATURE_WEIGHTS` | Weight for each feature type | Visual: 0.5, Structural: 0.3, Content: 0.2 |
| `HEADLESS` | Run WebDriver in headless mode | True |

## Algorithm Complexity

- **Page Processing**: O(n) where n is the number of DOM elements
- **Feature Extraction**: O(p) where p is the number of pixels in the processed image
- **Similarity Calculation**: O(f) where f is the feature vector dimension
- **Query Processing**: O(m) where m is the number of indexed URLs 