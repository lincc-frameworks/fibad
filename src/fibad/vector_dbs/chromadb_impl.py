from typing import Union

import chromadb
import numpy as np
from fibad.vector_dbs.vector_db_interface import VectorDB


class ChromaDB(VectorDB):
    """Implementation of the VectorDB interface using ChromaDB as the backend."""

    def __init__(self, config, context):
        super().__init__(config, context)
        self.chromadb_client = None
        self.collection = None

        self.shard_index = 0  # The current shard id for insertion
        self.shard_size = 0  # The number of vectors in the current shard

        # The approximate maximum size of a shard before a new one is created
        self.shard_size_limit = 41_666

    def connect(self):
        """Create a database connection"""
        results_dir = self.context["results_dir"]
        self.chromadb_client = chromadb.PersistentClient(path=str(results_dir))
        return self.chromadb_client

    def create(self):
        """Create a new database"""

        if self.chromadb_client is None:
            self.connect()

        # Create a chromadb shard (a.k.a. "collection")
        self.collection = self.chromadb_client.create_collection(
            name=f"shard_{self.shard_index}",
            metadata={
                # These are chromadb defaults, may want to make them configurable
                "hsnw:space": "l2",
                "hsnw:construction_ef": 100,
                "hsnw:search_ef": 100,
            },
        )

        return self.collection

    def insert(self, ids: list[Union[str | int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the database.

        Parameters
        ----------
        ids : list[Union[str | int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """

        # increment counter, if exceeds shard limit, create a new collection
        self.shard_size += len(ids)
        if self.shard_size > self.shard_size_limit:
            self.shard_index += 1
            self.shard_size = 0
            self.collection = self.create()

        self.collection.add(ids=ids, embeddings=vectors)

    def search_by_id(self, id: Union[str | int], k: int = 1) -> dict[int, list[Union[str | int]]]:
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
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the k
            nearest neighbors as the value.

        Raises
        ------
        ValueError
            If more than one vector is found for the given id
        """

        #! Fix the definition of the return type here!!!

        if k < 1:
            raise ValueError("k must be greater than 0")

        # create the database connection
        if self.chromadb_client is None:
            self.connect()

        # get all the shards
        shards = self.chromadb_client.list_collections()

        vectors = []
        for shard in shards:
            # Get the vector for the id
            collection = self.chromadb_client.get_collection(id=shard.id)
            results = collection.get(id, include=["embeddings"])

            vectors.extend(results["embeddings"])

        query_results: dict[int, list[Union[str | int]]] = {}
        # no matching id found in database
        if len(vectors) == 0:
            query_results = {}

        # multiple matching ids found in database
        elif len(vectors) > 1:
            raise ValueError(f"More than one vector found for id: {id}")

        # single matching id found in database
        else:
            query_results = self.search_by_vector(vectors, k=k)

        return query_results

    def search_by_vector(self, vectors: list[np.ndarray], k: int = 1) -> dict[int, list[Union[str | int]]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vectors : np.ndarray
            The vector to use when searching for nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the k
            nearest neighbors as the value.
        """

        if k < 1:
            raise ValueError("k must be greater than 0")

        # create the database connection
        if self.chromadb_client is None:
            self.connect()

        # get all the shards
        shards = self.chromadb_client.list_collections()

        # This dictionary will hold the k nearest neighbors ids for each input vector
        result_dict: dict[int, list[Union[str | int]]] = {i: [] for i in range(len(vectors))}

        # Intermediate results holds all of the query results from all shards.
        # These results will be sorted and trimmed to the appropriate length before
        # being added to `result_dict`.
        intermediate_results: dict[int, dict[str, list[Union[str | int]]]] = {
            i: {"ids": [], "distances": []} for i in range(len(vectors))
        }

        # Query each shard, return the k nearest neighbors from each shard.
        for shard in shards:
            collection = self.chromadb_client.get_collection(name=shard.name)
            results = collection.query(query_embeddings=vectors, n_results=k)
            for i in range(len(results["ids"])):
                intermediate_results[i]["ids"].extend(results["ids"][i])
                intermediate_results[i]["distances"].extend(results["distances"][i])

        # Sort the distances ascending
        for i in range(len(intermediate_results)):
            sorted_indicies = np.argsort(intermediate_results[i]["distances"], stable=True)
            result_dict[i] = [intermediate_results[i]["ids"][j] for j in sorted_indicies][:k]

        return result_dict
