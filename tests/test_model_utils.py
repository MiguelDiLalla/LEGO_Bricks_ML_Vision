import unittest
import os
import shutil
import cv2
import json
import logging
from utils import model_utils
import random

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestModelUtilsLocal(unittest.TestCase):
    def setUp(self):
        # Set up test image path
        folder = os.path.join(os.getcwd(), "presentation", "Test_images", "BricksPics")
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.test_image_path = os.path.join(folder, random.choice(files))
        if not os.path.exists(self.test_image_path):
            self.skipTest(f"Test image not found: {self.test_image_path}")
        logging.info(f"[SETUP] Test image located at: {self.test_image_path}")
        
        # Read image as a numpy array for later testing
        self.test_image = cv2.imread(self.test_image_path)
        if self.test_image is None:
            self.skipTest("Failed to load test image as numpy array.")
        
        # Set output directory for test results
        self.output_dir = os.path.join(os.getcwd(), "tests", "test_results")
        # Remove previous test results if they exist
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            logging.info(f"[SETUP] Removed existing output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"[SETUP] Created new output directory: {self.output_dir}")

    def test_detect_bricks_via_path(self):
        logging.info("[TEST] Running detect_bricks() using image path input.")
        # Call detect_bricks by passing the image path
        result = model_utils.detect_bricks(
            self.test_image_path, 
            model=None,             # Should default to loaded brick model from local mode
            conf=0.25, 
            save_json=True, 
            save_annotated=True, 
            output_folder=self.output_dir
        )
        # Check that a composite annotated image was created and metadata JSON exists
        meta = result.get("metadata", {})
        composite_path = meta.get("annotated_image_path", "")
        json_path = meta.get("json_results_path", "")
        logging.info(f"[TEST] Composite image path: {composite_path}")
        logging.info(f"[TEST] Metadata JSON path: {json_path}")
        # NEW: Print full metadata dictionary for inspection
        print("\n[DEBUG] Metadata (from image path detection):")
        # print(json.dumps(meta, indent=4))
        self.assertTrue(os.path.exists(composite_path), "Composite annotated image not created.")
        self.assertTrue(os.path.exists(json_path), "Metadata JSON file not created.")
    
    # def test_detect_bricks_via_numpy(self):
    #     logging.info("[TEST] Running detect_bricks() using image as NumPy array.")
    #     # Call detect_bricks by passing the numpy array (not a file path)
    #     result = model_utils.detect_bricks(
    #         self.test_image, 
    #         model=None, 
    #         conf=0.25, 
    #         save_json=True, 
    #         save_annotated=True, 
    #         output_folder=self.output_dir
    #     )
    #     meta = result.get("metadata", {})
    #     composite_path = meta.get("annotated_image_path", "")
    #     json_path = meta.get("json_results_path", "")
    #     logging.info(f"[TEST] Composite image path (numpy input): {composite_path}")
    #     logging.info(f"[TEST] Metadata JSON path (numpy input): {json_path}")
    #     # NEW: Print full metadata dictionary for inspection
    #     print("\n[DEBUG] Metadata (from NumPy array detection):")
    #     # print(json.dumps(meta, indent=4))
    #     self.assertTrue(os.path.exists(composite_path), "Composite annotated image not created (via NumPy).")
    #     self.assertTrue(os.path.exists(json_path), "Metadata JSON file not created (via NumPy).")
    
    # def test_render_metadata(self):
    #     logging.info("[TEST] Testing render_metadata() function directly.")
    #     # Render metadata panel on the test image using dummy metadata
    #     dummy_meta = {"dummy": True, "info": "Render test"}
    #     rendered_img = model_utils.render_metadata(self.test_image, dummy_meta)
    #     self.assertIsNotNone(rendered_img, "render_metadata() returned None.")
    #     # Save rendered metadata image in the output folder
    #     render_output_path = os.path.join(self.output_dir, "rendered_metadata.jpg")
    #     cv2.imwrite(render_output_path, rendered_img)
    #     logging.info(f"[TEST] Rendered metadata image saved at: {render_output_path}")
    #     self.assertTrue(os.path.exists(render_output_path), "Rendered metadata image file was not created successfully.")

if __name__ == '__main__':
    unittest.main()
