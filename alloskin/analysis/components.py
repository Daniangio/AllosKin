"""
Defines the Abstract Base Class (ABC) for all analysis components.
This provides a common interface for `run()`.
"""

from abc import ABC, abstractmethod
from typing import Any

class AnalysisComponent(ABC):
    """Abstract interface for an analysis goal."""
    
    @abstractmethod
    def run(self, data: Any, **kwargs) -> Any:
        """
        Run the analysis.
        
        The 'data' argument will be specific to the component.
        """
        pass