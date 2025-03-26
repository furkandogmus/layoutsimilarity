# web_similarity/page_processor.py
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import io
import cv2
import numpy as np
import hashlib

from config import TARGET_SIZE, SELENIUM_WAIT_TIME, MIN_ELEMENT_SIZE

def get_url_hash(url):
    """Create a hash of the URL to use as a unique identifier."""
    return hashlib.md5(url.encode()).hexdigest()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# JavaScript to get relevant leaf element data
# Needs refinement for robustness (e.g., handling iframes, shadow DOM)
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
            // console.warn("Error processing element:", elem, e);
        }
    }
    // Add document body itself if it has direct text/background? Maybe not needed if leaves are captured.
    return leaves;
}
return getElementData();
""" % MIN_ELEMENT_SIZE


class PageProcessor:
    def __init__(self, headless=True):
        self.driver = self._setup_driver(headless)

    def _setup_driver(self, headless):
        logging.info("Setting up WebDriver...")
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Set a large enough window size initially to capture layout
        chrome_options.add_argument("--window-size=1920,2000")
        chrome_options.add_argument("--hide-scrollbars")
        # Disable GPU for stability in headless mode?
        chrome_options.add_argument("--disable-gpu")

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            # Set timeout for page loads
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(15)
            logging.info("WebDriver setup complete.")
            return driver
        except WebDriverException as e:
            logging.error(f"Failed to setup WebDriver: {e}")
            raise

    def process_url(self, url):
        """
        Loads a URL, takes a screenshot, extracts leaf elements, and prepares the image.

        Returns:
            tuple: (processed_image, raw_elements_data, original_width, original_height) or None if failed.
                   processed_image is a TARGET_SIZExTARGET_SIZE numpy array (BGR).
                   raw_elements_data is a list of dictionaries from JS.
        """
        if not self.driver:
            logging.error("WebDriver not initialized.")
            return None

        try:
            logging.info(f"Loading URL: {url}")
            self.driver.get(url)
            logging.info(f"Waiting {SELENIUM_WAIT_TIME}s for page to render...")
            time.sleep(SELENIUM_WAIT_TIME) # Simple wait, consider WebDriverWait for specific elements

            # Get actual rendered dimensions (might be different from window size)
            page_width = self.driver.execute_script("return document.body.scrollWidth")
            page_height = self.driver.execute_script("return document.body.scrollHeight")
            logging.info(f"Page dimensions: {page_width}x{page_height}")

            # --- Get Leaf Elements Data ---
            logging.info("Extracting leaf elements via JavaScript...")
            raw_elements = self.driver.execute_script(GET_LEAF_ELEMENTS_JS)
            logging.info(f"Found {len(raw_elements)} potential leaf elements.")
            
            # Fallback for sites where our JS extraction doesn't find elements
            if len(raw_elements) == 0:
                logging.warning("No leaf elements found. Using fallback approach.")
                # Add a single element representing the entire viewport as a text element
                raw_elements = [{
                    'tag': 'BODY',
                    'x': 0,
                    'y': 0,
                    'width': page_width,
                    'height': min(page_height, TARGET_SIZE),
                    'text': 'Fallback element'
                }]

            # --- Take and Process Screenshot ---
            logging.info("Taking screenshot...")
            png = self.driver.get_screenshot_as_png()
            pil_image = Image.open(io.BytesIO(png))
            screenshot = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

            # Crop top part
            cropped_screenshot = screenshot[:min(page_height, TARGET_SIZE), :page_width]

            # Resize width to TARGET_SIZE, maintaining aspect ratio for height temporarily
            scale_factor = TARGET_SIZE / page_width if page_width > 0 else 1
            new_height = int(cropped_screenshot.shape[0] * scale_factor)
            resized_image = cv2.resize(cropped_screenshot, (TARGET_SIZE, new_height), interpolation=cv2.INTER_AREA)

            # Create final TARGET_SIZExTARGET_SIZE image (padding if necessary)
            final_image = np.full((TARGET_SIZE, TARGET_SIZE, 3), (255, 255, 255), dtype=np.uint8) # White background
            copy_height = min(new_height, TARGET_SIZE)
            final_image[:copy_height, :] = resized_image[:copy_height, :]

            logging.info(f"Screenshot processed to {TARGET_SIZE}x{TARGET_SIZE}.")

            # --- Adjust Element Coordinates for Resizing ---
            adjusted_elements = []
            for elem in raw_elements:
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

            logging.info(f"Adjusted coordinates for {len(adjusted_elements)} elements.")

            return final_image, adjusted_elements, page_width, page_height

        except TimeoutException:
            logging.error(f"Timeout loading URL: {url}")
            return None
        except WebDriverException as e:
            logging.error(f"WebDriver error processing {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error processing {url}: {e}")
            return None

    def close(self):
        if self.driver:
            logging.info("Closing WebDriver.")
            self.driver.quit()
            self.driver = None

# Example usage (for testing)
if __name__ == "__main__":
    processor = PageProcessor(headless=True)
    test_url = "http://example.com"
    result = processor.process_url(test_url)
    if result:
        img, elements, _, _ = result
        cv2.imwrite("processed_example.png", img)
        print(f"Saved processed image for {test_url}")
        print(f"First 5 elements: {elements[:5]}")
    else:
        print(f"Failed to process {test_url}")
    processor.close()