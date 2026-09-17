import logging
import os
import pickle
import sys

import numpy as np
import yaml

from src.criminalNetwork.utils.exception import CriminalNetworkException

logger = logging.getLogger(__name__)


def normalize_neo4j_uri(uri: str, trust_self_signed_certificate: bool) -> str:
    """Return a Neo4j URI that matches the target certificate trust mode.

    Some Neo4j cloud or self-hosted deployments present a self-signed certificate chain,
    even when the hostname is a `.neo4j.io` domain. In that case the driver must be
    told to trust the self-signed certificate using the `+ssc` scheme.
    """
    if not uri:
        return uri

    normalized_uri = uri.strip()
    if trust_self_signed_certificate:
        return (
            normalized_uri.replace("neo4j+s://", "neo4j+ssc://", 1)
            .replace("bolt+s://", "bolt+ssc://", 1)
            .replace("neo4j://", "neo4j+ssc://", 1)
            .replace("bolt://", "bolt+ssc://", 1)
        )

    return normalized_uri


def create_directories(path_to_directories: str) -> None:
    os.makedirs(path_to_directories, exist_ok=True)


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file) or {}
    except Exception as e:
        raise CriminalNetworkException(e, sys) from e


read_yaml_file = read_yaml

    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        directory = os.path.dirname(file_path)
        if directory:
            create_directories(directory)
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CriminalNetworkException(e, sys) from e
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            create_directories(dir_path)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CriminalNetworkException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info("Entered the save_object method")
        directory = os.path.dirname(file_path)
        if directory:
            create_directories(directory)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info("Exited the save_object method")
    except Exception as e:
        raise CriminalNetworkException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file does not exist: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CriminalNetworkException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CriminalNetworkException(e, sys) from e
    


def load_graph_schema(path) -> dict:
    with open(path, "r") as f:
        schema = yaml.safe_load(f)
    return schema