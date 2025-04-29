import cv2
import numpy as np
import re
import pytesseract
import easyocr
import logging
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from .models import LabTest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader for text recognition
reader = easyocr.Reader(['en'], gpu=False)

def process_lab_report(image_data: bytes) -> List[LabTest]:
    """
    Process a lab report image and extract test information using advanced techniques.
    
    Args:
        image_data: Raw image data bytes
        
    Returns:
        List of LabTest objects containing extracted information
    """
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Enhanced preprocessing
        preprocessed_img = enhanced_preprocess_image(img)
        
        # Try two methods and use the one that gives better results
        method1_result = process_with_table_detection(preprocessed_img)
        method2_result = process_with_layout_analysis(preprocessed_img, img)
        
        # Return the method that found more lab tests
        if len(method1_result) >= len(method2_result):
            return method1_result
        else:
            return method2_result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Return empty list on error
        return []

def enhanced_preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing with multiple techniques for better OCR results.
    
    Args:
        img: Input image
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Deskew (straighten) the image if needed
    gray = deskew(gray)
    
    # Denoise using Non-Local Means Denoising
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Binarize the image with Otsu's thresholding for better separation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure we have black text on white background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Apply morphological operations to clean up noise and enhance text
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary

def process_with_table_detection(preprocessed_img: np.ndarray) -> List[LabTest]:
    """
    Process lab report using advanced table detection approach.
    
    Args:
        preprocessed_img: Preprocessed image
    
    Returns:
        List of extracted lab tests
    """
    # Detect table structure with improved method
    cells = advanced_table_detection(preprocessed_img)
    
    # Extract text from cells using EasyOCR for better accuracy
    lab_tests = extract_lab_tests_with_easyocr(preprocessed_img, cells)
    
    return lab_tests

def process_with_layout_analysis(preprocessed_img: np.ndarray, original_img: np.ndarray) -> List[LabTest]:
    """
    Process lab report using layout analysis without strict table detection.
    
    Args:
        preprocessed_img: Preprocessed image
        original_img: Original color image
    
    Returns:
        List of extracted lab tests
    """
    # Perform direct text detection and extraction with EasyOCR
    results = reader.readtext(original_img)
    
    # Process EasyOCR results to extract lab tests
    return extract_tests_from_ocr_results(results)

def deskew(img: np.ndarray) -> np.ndarray:
    """
    Deskew (straighten) an image using the Hough transform.
    
    Args:
        img: Grayscale image
        
    Returns:
        Deskewed grayscale image
    """
    try:
        # Find all edges using Canny algorithm
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        
        # Use Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None and len(lines) > 0:
            # Find the dominant angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                if theta < np.pi/4 or theta > 3*np.pi/4:  # Only consider mostly vertical lines
                    angles.append(theta)
            
            if angles:
                # Calculate median angle
                median_angle = np.median(angles)
                
                # Calculate skew angle
                if median_angle > np.pi/2:
                    skew_angle = (np.pi - median_angle) * 180 / np.pi
                else:
                    skew_angle = median_angle * 180 / np.pi
                
                # Only correct if skew is significant
                if abs(skew_angle) > 0.5:
                    # Rotate to correct skew
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Return original if no skew correction was performed
        return img
    except Exception as e:
        return img  # Return original on error

def advanced_table_detection(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Advanced table detection with improved line detection and cell extraction.
    
    Args:
        img: Preprocessed binary image
        
    Returns:
        List of cell coordinates (x, y, w, h)
    """
    # Make a copy of the image
    table_img = img.copy()
    
    # Use multiple kernel sizes to detect lines of different lengths
    horizontal_kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    ]
    
    vertical_kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    ]
    
    # Initialize masks
    horizontal_mask = np.zeros(table_img.shape, dtype=np.uint8)
    vertical_mask = np.zeros(table_img.shape, dtype=np.uint8)
    
    # Apply each kernel for better line detection
    for kernel in horizontal_kernels:
        horizontal_lines = cv2.erode(table_img, kernel, iterations=2)
        horizontal_lines = cv2.dilate(horizontal_lines, kernel, iterations=2)
        horizontal_mask = cv2.bitwise_or(horizontal_mask, horizontal_lines)
    
    for kernel in vertical_kernels:
        vertical_lines = cv2.erode(table_img, kernel, iterations=2)
        vertical_lines = cv2.dilate(vertical_lines, kernel, iterations=2)
        vertical_mask = cv2.bitwise_or(vertical_mask, vertical_lines)
    
    # Combine horizontal and vertical lines
    table_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    
    # Apply morphology to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours of cells
    contours, hierarchy = cv2.findContours(~table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort cells by position
    cells = []
    min_width, min_height = 30, 15  # Minimum dimensions for a valid cell
    
    for i, contour in enumerate(contours):
        # Skip the outermost contour
        if hierarchy[0][i][3] == -1:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small, very large, or very elongated cells
        if (w > min_width and h > min_height and 
            w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9 and
            0.1 < w/h < 10):  # Aspect ratio check
            cells.append((x, y, w, h))
    
    # If no cells found with hierarchy filtering, try without it
    if not cells and contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > min_width and h > min_height and 
                w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9):
                cells.append((x, y, w, h))
    
    # Group cells into rows
    row_threshold = img.shape[0] // 30  # Adaptive threshold based on image height
    cells = sorted(cells, key=lambda c: (c[1] // row_threshold, c[0]))
    
    # If still no cells found, try a more basic approach with grid partitioning
    if not cells:
        cells = grid_partitioning(img)
    
    return cells

def grid_partitioning(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Partition the image into a grid of cells as a fallback method.
    
    Args:
        img: Input image
        
    Returns:
        List of cell coordinates (x, y, w, h)
    """
    height, width = img.shape
    
    # Estimate number of rows and columns based on image size
    num_rows = max(3, height // 50)  # Assume at least 3 rows
    num_cols = max(4, width // 200)  # Assume at least 4 columns
    
    # Calculate cell dimensions
    cell_height = height // num_rows
    cell_width = width // num_cols
    
    # Create cells
    cells = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * cell_width
            y = row * cell_height
            w = cell_width
            h = cell_height
            cells.append((x, y, w, h))
    
    return cells

def extract_lab_tests_with_easyocr(img: np.ndarray, cells: List[Tuple[int, int, int, int]]) -> List[LabTest]:
    """
    Extract lab test information from detected cells using EasyOCR for better text recognition.
    
    Args:
        img: Preprocessed image
        cells: List of cell coordinates
        
    Returns:
        List of LabTest objects
    """
    lab_tests = []
    
    # Convert grayscale to RGB for EasyOCR if needed
    if len(img.shape) == 2:  # If grayscale
        img_for_ocr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_for_ocr = img.copy()
    
    # Group cells into rows
    rows = group_cells_into_rows(cells)
    
    for row in rows:
        if len(row) >= 3:  # We need at least name, value, and range
            # Extract cell content using EasyOCR
            cell_texts = []
            for x, y, w, h in row:
                # Extract cell image
                cell_img = img_for_ocr[y:y+h, x:x+w]
                
                # Skip cells that are too small
                if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                    cell_texts.append("")
                    continue
                    
                # Apply EasyOCR
                result = reader.readtext(cell_img)
                
                # Extract text from results
                text = " ".join([r[1] for r in result]).strip()
                
                # If EasyOCR failed, try pytesseract as fallback
                if not text and len(img.shape) == 2:  # Only try fallback on grayscale images
                    text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()
                    
                cell_texts.append(text)
            
            # Process cell texts to identify test components
            if len(cell_texts) >= 3:
                # Try to identify components by position and content pattern
                test = extract_test_from_row(cell_texts)
                if test:
                    lab_tests.append(test)
    
    return lab_tests

def extract_test_from_row(cell_texts: List[str]) -> Optional[LabTest]:
    """
    Extract lab test details from a row of cell texts using advanced pattern matching.
    
    Args:
        cell_texts: List of texts extracted from cells in a row
        
    Returns:
        LabTest object if a valid test is found, None otherwise
    """
    # Skip empty rows
    if not any(cell_texts) or all(len(text.strip()) == 0 for text in cell_texts):
        return None
    
    # Join all cell texts to check for patterns across multiple cells
    combined_text = ' '.join(cell_texts)
    
    # Try pattern matching on the combined text first for common lab report formats
    # Pattern: Test Name followed by value with unit and reference range
    # Examples: "Hemoglobin 14.5 g/dL (12.0-16.0)", "RBC: 5.2 10^6/μL [4.5-5.9]"
    test_pattern = re.compile(
        r'([\w\s\(\)\.\,\-\+]+?)\s*[:\-]?\s*'
        r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*'
        r'([\w\/%\^\*\·]+)?\s*'
        r'(?:[\(\[\{]?\s*(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|<\s*\d+\.?\d*|>\s*\d+\.?\d*)\s*[\)\]\}]?)?'
    )
    
    match = test_pattern.search(combined_text)
    if match:
        test_name = match.group(1).strip()
        test_value = match.group(2).strip()
        test_unit = match.group(3).strip() if match.group(3) else ''
        bio_reference_range = match.group(4).strip() if match.group(4) else ''
        
        # Check if we have a valid test name and value
        if test_name and test_value and len(test_name) > 2:
            # Check if value is out of range
            out_of_range = check_range(test_value, bio_reference_range)
            
            return LabTest(
                test_name=clean_test_name(test_name),
                test_value=test_value,
                test_unit=test_unit,
                bio_reference_range=bio_reference_range,
                lab_test_out_of_range=out_of_range
            )
    
    # If no match found with combined text, try different layouts based on cell positions
    layouts = [
        # Layout 1: [test_name, test_value, test_unit, reference_range]
        lambda texts: parse_standard_layout(texts) if len(texts) >= 2 else None,
        
        # Layout 2: [test_name, test_value+unit, reference_range]
        lambda texts: parse_combined_value_unit(texts) if len(texts) >= 2 else None,
        
        # Layout 3: [test_name+value, test_unit, reference_range]
        lambda texts: parse_name_with_value(texts) if len(texts) >= 1 else None,
        
        # Layout 4: [test_code, test_name, test_value, test_unit, reference_range]
        lambda texts: parse_with_code(texts) if len(texts) >= 3 else None
    ]
    
    # Try each layout until we find one that works
    for layout_func in layouts:
        result = layout_func(cell_texts)
        if result:
            test_name, test_value, test_unit, bio_reference_range = result
            
            # Clean up and validate extracted data
            if test_name and test_value:
                # Check if value is out of range
                out_of_range = check_range(test_value, bio_reference_range)
                
                return LabTest(
                    test_name=clean_test_name(test_name),
                    test_value=test_value,
                    test_unit=test_unit,
                    bio_reference_range=bio_reference_range,
                    lab_test_out_of_range=out_of_range
                )
    
    return None

def parse_standard_layout(texts: List[str]) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse standard layout with separate columns for name, value, unit, range.
    Format: [test_name, test_value, test_unit, reference_range]
    """
    # Define regex patterns for different components
    value_pattern = re.compile(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)')
    range_pattern = re.compile(r'(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|<\s*\d+\.?\d*|>\s*\d+\.?\d*)')
    
    test_name = texts[0].strip()
    
    # Extract value from second cell
    value_match = value_pattern.search(texts[1])
    if not value_match:
        return None
    
    test_value = value_match.group(1)
    
    # Extract unit and reference range based on available cells
    if len(texts) > 3:
        test_unit = texts[2].strip()
        bio_reference_range = texts[3].strip()
        
        # If reference range doesn't match expected format, try to extract it
        if not range_pattern.search(bio_reference_range):
            range_match = range_pattern.search(bio_reference_range)
            if range_match:
                bio_reference_range = range_match.group(1)
            else:
                bio_reference_range = ''
    elif len(texts) > 2:
        # Try to determine if third cell is unit or reference range
        if range_pattern.search(texts[2]):
            test_unit = ''
            bio_reference_range = texts[2].strip()
        else:
            test_unit = texts[2].strip()
            bio_reference_range = ''
    else:
        test_unit = ''
        bio_reference_range = ''
    
    return test_name, test_value, test_unit, bio_reference_range

def parse_combined_value_unit(texts: List[str]) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse layout where value and unit are combined in one cell.
    Format: [test_name, test_value+unit, reference_range]
    """
    test_name = texts[0].strip()
    
    # Extract value and unit from second cell
    value_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', texts[1])
    if not value_match:
        return None
    
    test_value = value_match.group(1)
    
    # Everything after the value in the second cell is considered the unit
    value_end_pos = texts[1].find(test_value) + len(test_value)
    test_unit = texts[1][value_end_pos:].strip()
    
    # Remove common unit separators
    test_unit = re.sub(r'^[\s\*\·]+', '', test_unit)
    
    # Extract reference range from third cell if available
    bio_reference_range = ''
    if len(texts) > 2:
        range_match = re.search(r'(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|<\s*\d+\.?\d*|>\s*\d+\.?\d*)', texts[2])
        if range_match:
            bio_reference_range = range_match.group(1)
        else:
            bio_reference_range = texts[2].strip()
    
    return test_name, test_value, test_unit, bio_reference_range


def parse_name_with_value(texts: List[str]) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse layout where test name and value are in the same cell.
    Format: [test_name+value, test_unit, reference_range]
    """
    first_cell = texts[0]
    
    # Find numeric value in first cell
    value_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', first_cell)
    if not value_match:
        return None
    
    test_value = value_match.group(1)
    value_pos = first_cell.find(test_value)
    
    # Everything before the value is the test name
    test_name = first_cell[:value_pos].strip()
    
    # Everything after the value in the first cell might be part of the unit
    unit_part = first_cell[value_pos + len(test_value):].strip()
    
    # Determine unit and reference range based on available cells
    if len(texts) > 2:
        # If we have 3+ cells, second is likely unit, third is range
        test_unit = texts[1].strip() if not unit_part else unit_part
        bio_reference_range = texts[2].strip()
    elif len(texts) > 1:
        # With 2 cells, second could be unit or range
        range_match = re.search(r'(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|<\s*\d+\.?\d*|>\s*\d+\.?\d*)', texts[1])
        if range_match:
            test_unit = unit_part
            bio_reference_range = texts[1].strip()
        else:
            test_unit = texts[1].strip()
            bio_reference_range = ''
    else:
        # Only one cell, try to find unit in remaining text
        test_unit = unit_part
        bio_reference_range = ''
    
    return test_name, test_value, test_unit, bio_reference_range


def parse_with_code(texts: List[str]) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse layout that includes a test code before test name.
    Format: [test_code, test_name, test_value, test_unit, reference_range]
    """
    # Check if first cell looks like a code (short, may contain numbers)
    if len(texts[0]) <= 10 and (re.search(r'\d', texts[0]) or texts[0].isupper()):
        # If first cell is likely a code, shift everything
        test_name = texts[1].strip()
        
        if len(texts) > 2:
            value_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', texts[2])
            if value_match:
                test_value = value_match.group(1)
                if len(texts) > 3:
                    test_unit = texts[3].strip()
                    bio_reference_range = texts[4].strip() if len(texts) > 4 else ''
                else:
                    # Extract unit from value cell if needed
                    value_end_pos = texts[2].find(test_value) + len(test_value)
                    test_unit = texts[2][value_end_pos:].strip()
                    bio_reference_range = ''
                
                return test_name, test_value, test_unit, bio_reference_range
    
    return None


def clean_test_name(name: str) -> str:
    """
    Clean and normalize test names.
    
    Args:
        name: Raw test name
        
    Returns:
        Cleaned test name
    """
    # Remove common prefixes and suffixes
    name = re.sub(r'^test[:\s-]*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^examination[:\s-]*', '', name, flags=re.IGNORECASE)
    
    # Remove trailing punctuation
    name = re.sub(r'[:\-,;]*$', '', name)
    
    # Fix common OCR errors in medical terms
    name = name.replace('0', 'O')  # Replace zero with capital O
    
    return name.strip()

def group_cells_into_rows(cells: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
    """
    Group cells into rows based on their y-coordinates.
    
    Args:
        cells: List of cell coordinates
        
    Returns:
        List of rows, where each row is a list of cell coordinates
    """
    if not cells:
        return []
    
    # Define a threshold for considering cells to be in the same row
    row_threshold = 20
    
    # Sort cells by y-coordinate
    sorted_cells = sorted(cells, key=lambda c: c[1])
    
    # Group cells into rows
    rows = []
    current_row = [sorted_cells[0]]
    current_y = sorted_cells[0][1]
    
    for cell in sorted_cells[1:]:
        if abs(cell[1] - current_y) <= row_threshold:
            # Same row
            current_row.append(cell)
        else:
            # New row
            if current_row:
                # Sort cells in the row by x-coordinate
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]
                current_y = cell[1]
    
    # Add the last row if it exists
    if current_row:
        rows.append(sorted(current_row, key=lambda c: c[0]))
    
    return rows

def extract_tests_from_ocr_results(ocr_results) -> List[LabTest]:
    """
    Extract lab tests from direct OCR results without table detection using advanced regex patterns.
    
    Args:
        ocr_results: Results from EasyOCR reader.readtext
        
    Returns:
        List of LabTest objects
    """
    # Extract all text lines from OCR results
    text_lines = []
    for (bbox, text, prob) in ocr_results:
        if prob > 0.5:  # Only consider high confidence detections
            text_lines.append((text.strip(), bbox))
    
    # Analyze vertical positions to identify potential rows
    if not text_lines:
        return []
    
    # Sort by y-coordinate (vertical position)
    text_lines.sort(key=lambda x: x[1][0][1])  # Sort by top-left y-coordinate
    
    # Group lines by vertical proximity with adaptive threshold based on text height
    rows = []
    if text_lines:
        # Calculate average text height to set better row threshold
        avg_height = sum([bbox[2][1] - bbox[0][1] for _, bbox in text_lines]) / len(text_lines)
        row_threshold = max(20, avg_height * 1.2)  # At least 20 pixels or 1.2x text height
        
        current_row = [text_lines[0]]
        current_y = text_lines[0][1][0][1]
        
        for line in text_lines[1:]:
            y = line[1][0][1]  # Top-left y-coordinate
            if abs(y - current_y) <= row_threshold:
                # Same row
                current_row.append(line)
            else:
                # New row
                if current_row:
                    # Sort items in the row by x-coordinate
                    current_row.sort(key=lambda x: x[1][0][0])  # Sort by top-left x-coordinate
                    rows.append([item[0] for item in current_row])  # Extract only text
                    current_row = [line]
                    current_y = y
        
        # Add the last row if it exists
        if current_row:
            current_row.sort(key=lambda x: x[1][0][0])  # Sort by top-left x-coordinate
            rows.append([item[0] for item in current_row])  # Extract only text
    
    # Define common medical lab test names to help with identification
    common_lab_tests = (
        r'(?:hemoglobin|hgb|hb|wbc|rbc|platelets|plt|mch|mchc|hematocrit|hct|' +
        r'glucose|sodium|potassium|chloride|calcium|magnesium|phosphorus|' +
        r'creatinine|bun|alt|ast|alp|ggt|bilirubin|albumin|protein|' +
        r'cholesterol|hdl|ldl|triglycerides|tsh|t3|t4|' +
        r'vitamin|iron|ferritin|transferrin|folate|b12)'
    )
    
    # Process rows to extract lab tests
    lab_tests = []
    
    # First pass - look for complete lab tests in individual rows
    for row in rows:
        # Skip short rows or rows unlikely to contain lab tests
        if len(row) < 2:
            continue
        
        # Join row text to analyze as a whole
        row_text = ' '.join(row)
        
        # Multiple pattern matching for different lab report formats
        patterns = [
            # Pattern 1: Test Name: 12.3 units (10.0-15.0)
            re.compile(
                r'([\w\s\,\.\-\(\)]+?)\s*[:=]\s*' +
                r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*' +
                r'([\w\s\/%\^\*\·\(\)]+?)?\s*' +
                r'(?:[\(\[\{]?\s*(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|[<>]\s*\d+\.?\d*)\s*[\)\]\}]?)?',
                re.IGNORECASE
            ),
            
            # Pattern 2: Test Name 12.3 units 10.0-15.0
            re.compile(
                r'([\w\s\.\,\-\(\)]+?)\s+' +
                r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*' +
                r'([\w\s\/%\^\*\·\(\)]+?)?\s*' +
                r'(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|[<>]\s*\d+\.?\d*)',
                re.IGNORECASE
            ),
            
            # Pattern 3: Common lab test names with values
            re.compile(
                r'(' + common_lab_tests + r'[\w\s\.\,\-\(\)]*?)\s*[:=]?\s*' +
                r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*' +
                r'([\w\s\/%\^\*\·\(\)]+?)?',
                re.IGNORECASE
            )
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = pattern.search(row_text)
            if match:
                test_name = match.group(1).strip()
                test_value = match.group(2).strip()
                
                # Extract unit and reference range if available
                if len(match.groups()) > 2:
                    test_unit = match.group(3).strip() if match.group(3) else ''
                else:
                    test_unit = ''
                    
                if len(match.groups()) > 3 and match.group(4):
                    bio_reference_range = match.group(4).strip()
                else:
                    # Try to find reference range elsewhere in the text
                    range_match = re.search(r'(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|[<>]\s*\d+\.?\d*)', row_text)
                    bio_reference_range = range_match.group(0) if range_match else ''
                
                # Check if we have a valid test name and value
                if test_name and test_value and len(test_name) > 2 and re.search(r'\d', test_value):
                    # Check if value is out of range
                    out_of_range = check_range(test_value, bio_reference_range)
                    
                    lab_test = LabTest(
                        test_name=clean_test_name(test_name),
                        test_value=test_value,
                        test_unit=test_unit,
                        bio_reference_range=bio_reference_range,
                        lab_test_out_of_range=out_of_range
                    )
                    lab_tests.append(lab_test)
                    break  # Found a match with this pattern, move to next row
    
    # Second pass - look for split tests across multiple text elements
    for i in range(len(rows)-1):
        current_row = rows[i]
        next_row = rows[i+1]
        
        # Look for rows that might have test names without values
        for text in current_row:
            # Skip if too short or contains numeric values
            if len(text) < 3 or re.search(r'\d', text):
                continue
                
            # Check if this might be a test name
            test_name = text.strip()
            
            # Look for corresponding values in the next row
            for next_text in next_row:
                value_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', next_text)
                if value_match:
                    test_value = value_match.group(1)
                    
                    # Try to extract unit and reference range
                    value_end_pos = next_text.find(test_value) + len(test_value)
                    remaining_text = next_text[value_end_pos:].strip()
                    
                    # Extract unit
                    unit_match = re.search(r'^([\w\s\/%\^\*\·\(\)]+?)[\s\(\[]', remaining_text)
                    test_unit = unit_match.group(1).strip() if unit_match else ''
                    
                    # Extract reference range
                    range_match = re.search(r'(\d+\.?\d*\s*[-–—]\s*\d+\.?\d*|[<>]\s*\d+\.?\d*)', remaining_text)
                    bio_reference_range = range_match.group(0) if range_match else ''
                    
                    # Check if already have this test (avoid duplicates)
                    if not any(t.test_name.lower() == clean_test_name(test_name).lower() for t in lab_tests):
                        out_of_range = check_range(test_value, bio_reference_range)
                        
                        lab_test = LabTest(
                            test_name=clean_test_name(test_name),
                            test_value=test_value,
                            test_unit=test_unit,
                            bio_reference_range=bio_reference_range,
                            lab_test_out_of_range=out_of_range
                        )
                        lab_tests.append(lab_test)
                        break  # Found a match, move to next potential test name
    
    # Remove any exact duplicates
    unique_lab_tests = []
    seen = set()
    for test in lab_tests:
        key = (test.test_name.lower(), test.test_value)
        if key not in seen:
            seen.add(key)
            unique_lab_tests.append(test)
    
    return unique_lab_tests

def check_range(test_value: str, reference_range: str) -> bool:
    """
    Check if test value is outside the reference range with enhanced pattern matching.
    
    Args:
        test_value: String containing the test value
        reference_range: String containing the reference range
        
    Returns:
        True if out of range, False otherwise or if unable to determine
    """
    try:
        # Skip check if no reference range provided
        if not reference_range or reference_range.strip() == '-':
            return False
        
        # Extract numeric test value, handling scientific notation
        value_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', test_value)
        if not value_match:
            return False
        
        # Convert to float, handling scientific notation
        value_str = value_match.group(1)
        value = float(value_str)
        
        # Patterns for different reference range formats
        
        # Standard range format: "10.0-20.0" or "10.0 - 20.0" or "10.0–20.0" (en dash) or "10.0—20.0" (em dash)
        range_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*[-–—]\s*(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', reference_range)
        if range_match:
            range_min = float(range_match.group(1))
            range_max = float(range_match.group(2))
            
            # Check if value is outside the range
            return value < range_min or value > range_max
        
        # Less than formats: "<10", "< 10", "<=10", "<= 10"
        less_than_match = re.search(r'<\s*(=)?\s*(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', reference_range)
        if less_than_match:
            equals = less_than_match.group(1) is not None  # True if "<="
            upper_limit = float(less_than_match.group(2))
            
            if equals:
                return value > upper_limit  # Out of range if value > limit for "<="
            else:
                return value >= upper_limit  # Out of range if value >= limit for "<"
        
        # Greater than formats: ">10", "> 10", ">=10", ">= 10"
        greater_than_match = re.search(r'>\s*(=)?\s*(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', reference_range)
        if greater_than_match:
            equals = greater_than_match.group(1) is not None  # True if ">="
            lower_limit = float(greater_than_match.group(2))
            
            if equals:
                return value < lower_limit  # Out of range if value < limit for ">="
            else:
                return value <= lower_limit  # Out of range if value <= limit for ">"
        
        # Range with units: "10.0-20.0 mg/dL"
        range_with_units_match = re.search(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*[-–—]\s*(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)\s*[a-zA-Z\/%\^]+', reference_range)
        if range_with_units_match:
            range_min = float(range_with_units_match.group(1))
            range_max = float(range_with_units_match.group(2))
            
            return value < range_min or value > range_max
        
        # Format with up to/up to and including: "up to 10" or "up to and including 10"
        up_to_match = re.search(r'up\s+to(?:\s+and\s+including)?\s+(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', reference_range, re.IGNORECASE)
        if up_to_match:
            upper_limit = float(up_to_match.group(1))
            # Assume "up to" means "<="
            return value > upper_limit
        
        # Multiple values (choose the closest to the test value)
        multi_value_match = re.findall(r'(\d+\.?\d*(?:[Ee][\+\-]?\d+)?)', reference_range)
        if len(multi_value_match) >= 2:
            # Convert all to float
            ref_values = [float(v) for v in multi_value_match]
            # Sort by distance to test value
            ref_values.sort(key=lambda x: abs(x - value))
            
            # If the closest reference value is far from the test value, consider it out of range
            # Use a threshold of 15% difference
            closest = ref_values[0]
            return abs(value - closest) / closest > 0.15
        
        return False  # Default to in-range if format is unrecognized
    
    except (ValueError, AttributeError) as e:
        # f"Error checking range: {str(e)} for value '{test_value}' and range '{reference_range}'")
        return False
