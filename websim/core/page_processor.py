"""
Web page processing module for extracting visual elements and screenshots.

This module handles all interactions with web browsers through Selenium,
extracts relevant page elements, and captures screenshots for analysis.
"""

import time
import logging
import hashlib
import io
from typing import Tuple, List, Dict, Optional, Any, Union

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import cv2
import numpy as np

from ..config import (
    TARGET_SIZE, SELENIUM_WAIT_TIME, MIN_ELEMENT_SIZE, 
    SCREENSHOT_WIDTH
)

# Set up module logger
logger = logging.getLogger(__name__)

def get_url_hash(url: str) -> str:
    """Create a hash of the URL to use as a unique identifier.
    
    Args:
        url: The URL to hash
        
    Returns:
        A hexadecimal hash string for the URL
    """
    return hashlib.md5(url.encode()).hexdigest()

# JavaScript to get relevant leaf element data
GET_LEAF_ELEMENTS_JS = """
function getElementData() {
    const leaves = [];
    const elements = document.body.getElementsByTagName('*');
    const minSize = %d; // Inject MIN_ELEMENT_SIZE
    const pageScrollX = window.scrollX;
    const pageScrollY = window.scrollY;

    for (let i = 0; i < elements.length; i++) {
        const elem = elements[i];
        try {
            const rect = elem.getBoundingClientRect();
            const style = window.getComputedStyle(elem);

            // Basic filtering: visible, minimum size
            if (rect.width >= minSize &&
                rect.height >= minSize &&
                style.display !== 'none' &&
                style.visibility !== 'hidden' &&
                style.opacity > 0)
            {
                 // Simple leaf heuristic: common media/form tags OR no block children OR text node container
                 let isLeafLike = ['IMG', 'SVG', 'CANVAS', 'VIDEO', 'IFRAME', // Media/Embed
                                'INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', // Form
                                'HR'] // Other visual elements
                                .includes(elem.tagName);

                let containsText = elem.innerText && elem.innerText.trim().length > 0;

                 if (!isLeafLike) {
                     let hasBlockChild = false;
                     for(let j=0; j < elem.children.length; j++){
                         const childStyle = window.getComputedStyle(elem.children[j]);
                         if(['block', 'flex', 'grid', 'list-item', 'table'].includes(childStyle.display) &&
                             elem.children[j].getBoundingClientRect().height > 0){ // Check if child actually occupies space
                             hasBlockChild = true;
                             break;
                         }
                     }
                     // If no block children but contains text, consider it a text container
                     if (!hasBlockChild && containsText) {
                        isLeafLike = true;
                     }
                 }

                 if(isLeafLike) {
                    leaves.push({
                        tag: elem.tagName,
                        x: rect.left + pageScrollX,
                        y: rect.top + pageScrollY,
                        width: rect.width,
                        height: rect.height,
                        text: containsText ? elem.innerText.trim() : "" // Basic text content
                        // Could add more: background-image, etc.
                    });
                 }
            }
        } catch (e) {
            // Ignore elements that cause errors (e.g., detached from DOM)
        }
    }
    return leaves;
}
return getElementData();
""" % MIN_ELEMENT_SIZE


class PageProcessor:
    """Processes web pages to extract visual elements and screenshots."""
    
    def __init__(self, headless: bool = True, screenshot_width: int = SCREENSHOT_WIDTH, 
                 screenshot_height: Optional[int] = None):
        """Initialize the page processor.
        
        Args:
            headless: Whether to run the browser in headless mode
            screenshot_width: Width of the screenshot to capture
            screenshot_height: Height of the screenshot (optional, will use window size if not specified)
        """
        self.headless = headless
        self.screenshot_width = screenshot_width
        self.screenshot_height = screenshot_height
        self.driver = self._setup_driver()
        
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up the Selenium WebDriver.
        
        Returns:
            A configured Chrome WebDriver instance
        
        Raises:
            WebDriverException: If the WebDriver setup fails
        """
        logger.info("Setting up WebDriver...")
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Set window size for screenshot capture
        window_size = f"--window-size={self.screenshot_width},{self.screenshot_height or 2000}"
        chrome_options.add_argument(window_size)
        chrome_options.add_argument("--hide-scrollbars")
        chrome_options.add_argument("--disable-gpu")  # For stability in headless mode

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            # Set timeout for page loads
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(15)
            logger.info("WebDriver setup complete.")
            return driver
        except WebDriverException as e:
            logger.error(f"Failed to setup WebDriver: {e}")
            raise

    def process_url(self, url: str) -> Optional[Tuple[List[Dict[str, Any]], np.ndarray, int, int]]:
        """Process a URL to extract visual elements and take a screenshot.
        
        Args:
            url: The URL to process
            
        Returns:
            A tuple containing:
                - List of adjusted element data dictionaries
                - Processed screenshot as a numpy array (BGR)
                - Original page width
                - Original page height
            Or None if processing fails
        """
        if not self.driver:
            logger.error("WebDriver not initialized.")
            return None

        try:
            logger.info(f"Loading URL: {url}")
            self.driver.get(url)
            logger.info(f"Waiting {SELENIUM_WAIT_TIME}s for page to render...")
            time.sleep(SELENIUM_WAIT_TIME)

            # Get actual rendered dimensions
            page_width = self.driver.execute_script("return document.body.scrollWidth")
            page_height = self.driver.execute_script("return document.body.scrollHeight")
            logger.info(f"Page dimensions: {page_width}x{page_height}")

            # Extract leaf elements
            logger.info("Extracting leaf elements via JavaScript...")
            raw_elements = self.driver.execute_script(GET_LEAF_ELEMENTS_JS)
            logger.info(f"Found {len(raw_elements)} potential leaf elements.")
            
            # Fallback for sites with no detected elements
            if len(raw_elements) == 0:
                logger.warning("No leaf elements found. Using fallback approach.")
                raw_elements = [{
                    'tag': 'BODY',
                    'x': 0,
                    'y': 0,
                    'width': page_width,
                    'height': min(page_height, TARGET_SIZE),
                    'text': 'Fallback element'
                }]

            # Take and process screenshot
            logger.info("Taking screenshot...")
            png = self.driver.get_screenshot_as_png()
            pil_image = Image.open(io.BytesIO(png))
            screenshot = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

            # Process screenshot to standard size
            processed_image, adjusted_elements = self._process_screenshot_and_elements(
                screenshot, raw_elements, page_width, page_height
            )

            return adjusted_elements, processed_image, page_width, page_height

        except TimeoutException:
            logger.error(f"Timeout loading URL: {url}")
            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error processing {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            return None

    def _process_screenshot_and_elements(
        self, screenshot: np.ndarray, elements: List[Dict[str, Any]], 
        page_width: int, page_height: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process the screenshot and adjust element coordinates.
        
        Args:
            screenshot: Original screenshot as numpy array
            elements: List of raw element data
            page_width: Original page width
            page_height: Original page height
            
        Returns:
            Tuple of:
                - Processed image (TARGET_SIZE x TARGET_SIZE)
                - Adjusted element data with coordinates scaled to TARGET_SIZE
        """
        # Crop top part of the page
        cropped_screenshot = screenshot[:min(page_height, TARGET_SIZE), :page_width]

        # Resize width to TARGET_SIZE, maintaining aspect ratio for height temporarily
        scale_factor = TARGET_SIZE / page_width if page_width > 0 else 1
        new_height = int(cropped_screenshot.shape[0] * scale_factor)
        resized_image = cv2.resize(cropped_screenshot, (TARGET_SIZE, new_height), 
                                  interpolation=cv2.INTER_AREA)

        # Create final TARGET_SIZE x TARGET_SIZE image (padding if necessary)
        final_image = np.full((TARGET_SIZE, TARGET_SIZE, 3), (255, 255, 255), 
                             dtype=np.uint8)  # White background
        copy_height = min(new_height, TARGET_SIZE)
        final_image[:copy_height, :] = resized_image[:copy_height, :]

        logger.info(f"Screenshot processed to {TARGET_SIZE}x{TARGET_SIZE}.")

        # Adjust element coordinates for resizing
        adjusted_elements = []
        for elem in elements:
            adj_x = elem['x'] * scale_factor
            adj_y = elem['y'] * scale_factor
            adj_w = elem['width'] * scale_factor
            adj_h = elem['height'] * scale_factor

            # Only keep elements that start within the TARGET_SIZE vertical boundary
            if adj_y < TARGET_SIZE:
                elem['adj_x'] = adj_x
                elem['adj_y'] = adj_y
                elem['adj_w'] = adj_w
                elem['adj_h'] = adj_h
                adjusted_elements.append(elem)

        logger.info(f"Adjusted coordinates for {len(adjusted_elements)} elements.")
        return final_image, adjusted_elements

    def close(self) -> None:
        """Close the WebDriver and release resources."""
        if self.driver:
            logger.info("Closing WebDriver.")
            self.driver.quit()
            self.driver = None 