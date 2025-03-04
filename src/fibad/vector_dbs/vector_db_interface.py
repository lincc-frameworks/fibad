from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np


class VectorDB(ABC):
    """Interface for a vector database"""

    def __init__(self, config: Optional[dict] = None, context: Optional[dict] = None):
        """Create a new instance of a `VectorDB` object.

        Parameters
        ----------
        config : dict, optional
            An instance of the runtime configuration, by default None
        context : dict, optional
            An instance of the context object, by default None
        """
        self.config = config
        self.context = context

    @abstractmethod
    def connect(self):
        """Connect to an existing database"""
        pass

    @abstractmethod
    def create(self):
        """Create a new database"""
        pass

    @abstractmethod
    def insert(self, ids: list[Union[str | int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the database.

        Parameters
        ----------
        ids : list[Union[str | int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """
        pass

    @abstractmethod
    def search_by_id(self, id: Union[str | int], k: int = 1) -> list[Union[str, int]]:
        """Get the ids of the k nearest neighbors for a given id in the database.
        Should use the provided id to look up the vector, then call search_by_vector.

        Parameters
        ----------
        id : Union[str | int]
            The id of the vector in the database for which we want to find the
            k nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        list[Union[str,int]]
            The ids of the k nearest neighbors
        """
        pass

    @abstractmethod
    def search_by_vector(self, vector: np.ndarray, k: int = 1) -> list[Union[str, int]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vector : np.ndarray
            The vector to use when searching for nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        list[Union[str,int]]
            The ids of the k nearest neighbors
        """
        pass
