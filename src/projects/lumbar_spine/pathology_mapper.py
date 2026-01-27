# coding: utf-8

import logging
from typing import List, Optional, Dict

class PathologyMapper:
    """
    Bidirectional mapper between pathology names strings (so-called "Condition")
    and indices (integers).
    This class handles the conversion of categorical string labels into integer formats
    suitable from machine learning models and provides the reverse lookup functionality.
    """

    def __init__(self, categories: List[str], logger: Optional[logging.logger]=None):
        """
        Initialize the mapper with a list of category strings

        Args:
            - categories (List[str]: list of pathology names (so-called conditions).
                The order in the list determines the integer index.
            - logger (Optional[logging.logger]: logger instance fro status report)
        """
        self.logger = logger

        # Use dict.fromkeys to remove duplicate while preserving insertion order
        unique_categories = list(dict.fromkeys(categories))

        # Create the forward mapping: String -> Integer.
        self.mapping: Dict[str, int] = {
            name: idx for idx, name in enumerate(unique_categories)
        }

        # Create the reverse mapping: Integer -> String
        self.reverse_mapping: Dict[int, str] = {
            idx: name for idx, name in enumerate(unique_categories)
        }

        if self.logger:
            self.logger.info(
                f"PathologyMapper initialized with {len(self.mapping)} unique categories."
            )
    
    def to_int(self, name:str) -> int:
        """
        Convert a pathology name to its corresponding integer index.

        Args: 
            - name (str): the pathology label to convert
        
        Returns: 
            int: the index of the pathology or -1 if the name is not recognized.
        """
        return self.mapping.get(name, -1)

    def to_str(self, index: int) -> str:
        """
        Convert an integer index back to its original pathology name.

        Args:
            - index (int): the integer index to look up

        Returns: 
            str: The pathology name, or "Unknown" if the index is out of bounds.
        """
        return self.reverse_mapping.get(index, "Unknown")

    def __call__(self, name: str) -> int:
        """
        Allows the object to be called like a function

        Example:
            mapper = PathologyMapper(["Normal", "Stenosis"])
            index = mapper("Normal")  # Returns 0

        Args: 
            - name (str): the pathology label to convert

        Return:
            int: The corresponding integer index
        """
        return self.to_int(name)

    def get_all_categories(self) -> List[str]:
        """
        Returns the list of all the registered pathology names
        """
        return list(self.mapping.keys())
