import os
import unittest
from PIL import Image
from segment.background_replacer import BackgroundReplacer

class TestBackgroundReplacer(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.bg_replacer = BackgroundReplacer(openai_api_key=self.api_key)
        # Basit bir siyah-beyaz test görüntüsü oluştur
        self.test_image = Image.new("RGB", (64, 64), color="white")

    def test_create_mask(self):
        binary_mask, mask = self.bg_replacer.create_mask(self.test_image, "object", 0.4)
        self.assertEqual(binary_mask.shape, (64, 64))
        self.assertEqual(mask.shape, (64, 64))

if __name__ == "__main__":
    unittest.main() 