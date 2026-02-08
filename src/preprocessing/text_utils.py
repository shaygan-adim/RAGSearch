import hazm
import pandas as pd

class TextCleaner:
    def __init__(self, use_hazm=True):
        """
        Initializes the text cleaner.
        :param use_hazm: Boolean flag to enable/disable Hazm normalization.
        """
        self.use_hazm = use_hazm
        if self.use_hazm:
            self.normalizer = hazm.Normalizer()

    def normalize(self, text):
        """
        Cleans and normalizes the input text.
        1. Handles None/NaN values.
        2. Applies Hazm normalization (character unification, zero-width spaces).
        3. Strips whitespace.
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        
        if not text:
            return ""

        if self.use_hazm:
            try:
                text = self.normalizer.normalize(text)
            except Exception as e:
                # Fallback in case of unexpected hazm errors
                pass
        
        return text

    def filter_length(self, text, min_len):
        """
        Returns True if text length >= min_len, else False.
        """
        if not isinstance(text, str):
            return False
        return len(text) >= min_len