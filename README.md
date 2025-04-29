# Lab Report Extractor

A FastAPI microservice that processes scanned medical lab-report images (PNG/JPEG) and returns structured JSON of all test entries using advanced computer vision and regular expression techniques.

## Architecture

```
                                   ┌─────────────────┐
                                   │     FastAPI      │
                                   │     Server       │
                                   └────────┬────────┘
                                            │
                                            ▼
                       ┌───────────────────────────────────┐
                       │    Lab Report Processing Pipeline  │
                       └───────────────┬───────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    Preprocessing    │   │   Table Detection   │   │  Direct OCR Analysis │
│                     │   │                     │   │                     │
│ - CLAHE Enhancement │   │ - Adaptive Kernels  │   │ - EasyOCR Extraction│
│ - Deskewing         │   │ - Multiple Scales   │   │ - Position Analysis │
│ - Denoising         │   │ - Contour Analysis  │   │ - Row Detection     │
│ - Binarization      │   │ - Grid Partitioning │   │ - Pattern Matching  │
└─────────┬───────────┘   └─────────┬───────────┘   └─────────┬───────────┘
          │                         │                         │
          └─────────────┬───────────────────────┬─────────────┘
                        │                       │
                        ▼                       ▼
           ┌─────────────────────┐   ┌─────────────────────┐
           │   Text Extraction   │   │   Pattern Matching  │
           │   with EasyOCR      │   │   with RegEx        │
           └─────────┬───────────┘   └─────────┬───────────┘
                     │                         │
                     └─────────────┬───────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │     Structured Lab Test     │
                     │     Data Extraction         │
                     └─────────────┬───────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │     Range Analysis and      │
                     │     JSON Response           │
                     └─────────────────────────────┘
```

## Features

- **Advanced Preprocessing**: CLAHE enhancement, deskewing, denoising, adaptive binarization
- **Dual-approach Processing**: Table detection + Direct text analysis approaches, using the best result
- **EasyOCR Integration**: Modern OCR with higher accuracy than Tesseract
- **Pattern Recognition**: Sophisticated regex patterns for complex lab report formats
- **Multiple Layout Support**: Handles various lab report layouts and formats
- **Enhanced Range Analysis**: Supports multiple reference range formats (10-20, <10, >=5, etc.)

## Requirements

- Python 3.8+
- EasyOCR and its dependencies

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```
   python server.py
   ```

2. The API will be available at http://localhost:8000

3. Use the `/get-lab-tests` endpoint to upload and process lab report images.

### API Endpoints

- **POST** `/get-lab-tests`: Upload a lab report image for processing
- **GET** `/health`: Health check endpoint
- **GET** `/docs`: Swagger UI documentation

### Example Response

```json
{
  "is_success": true,
  "data": [
    {
      "test_name": "Hemoglobin",
      "test_value": "14.2",
      "bio_reference_range": "12.0-16.0",
      "test_unit": "g/dL",
      "lab_test_out_of_range": false
    },
    {
      "test_name": "WBC",
      "test_value": "11.5",
      "bio_reference_range": "4.0-10.0",
      "test_unit": "10^3/μL",
      "lab_test_out_of_range": true
    }
  ]
}
```

## Error Handling

If any exception occurs during processing, the API will return:

```json
{
  "is_success": false,
  "data": []
}
```

## Implementation Details

### 1. Enhanced Preprocessing

The system performs advanced image preprocessing to optimize for OCR accuracy:

- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for better contrast
- **Advanced Deskewing**: Uses Hough transform to detect and straighten text lines
- **Non-Local Means Denoising**: Superior noise reduction that preserves edges
- **Otsu's Thresholding**: Adaptive binarization for optimal text/background separation

### 2. Dual-Approach Processing

The system uses two complementary approaches and selects the best result:

#### Approach 1: Table Detection and Cell Analysis

- **Multi-scale Kernel Detection**: Uses multiple kernels to detect lines of different lengths
- **Adaptive Contour Filtering**: Identifies cells based on size, aspect ratio, and position
- **Hierarchical Analysis**: Analyzes containment relationships for better cell detection
- **Grid Fallback**: Applies grid partitioning when explicit table lines cannot be detected

#### Approach 2: Direct OCR with Layout Analysis

- **EasyOCR Text Detection**: Extracts text blocks directly without table detection
- **Adaptive Row Grouping**: Groups text by position using dynamic vertical thresholds
- **Content Relationship Analysis**: Identifies relationships between text elements

### 3. Pattern Recognition with Regular Expressions

The system uses sophisticated regex patterns to extract lab test information:

- **Multiple Layout Parsers**: Different parsers for standard, combined, and complex layouts
- **Common Medical Test Patterns**: Special handling for common lab tests like CBC, lipid panels, etc.
- **Reference Range Recognition**: Parses various formats (10-20, <10, ≤5, "up to 15", etc.)
- **Unit Extraction**: Intelligent separation of values from units

### 4. Range Analysis

The system performs nuanced range analysis to determine if values are outside reference ranges:

- **Multiple Format Support**: Handles ranges with dashes, inequality symbols, and text
- **Scientific Notation Parsing**: Correctly processes values in scientific notation
- **Unit Awareness**: Considers units when comparing values to ranges
- **Proximity Analysis**: For complex formats, evaluates how far values are from reference points

### 5. Output
![image](https://github.com/user-attachments/assets/72816a65-8413-44da-8d7c-09042925f75c)
