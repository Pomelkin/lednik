from uuid import uuid4

from kostyl.utils import setup_logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import QueryRequest
from qdrant_client.http.models import QueryResponse
from qdrant_client.http.models import VectorParams
from torch import Tensor

from lednik.distill.validation.structs import MRRConfig


logger = setup_logger()

client: QdrantClient | None = None

_QDRANT_UPLOAD_BATCH_SIZE = 64
_QDRANT_QUERY_BATCH_SIZE = 64


def _batched(items: list, batch_size: int) -> list[list]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _calculate_mrr_for_query(response_uuids: list[str], target_uuid: str) -> float:
    for rank, uuid_ in enumerate(response_uuids, start=1):
        if uuid_ == target_uuid:
            return 1 / rank
    return 0.0


def _extract_uuids_from_points(
    responses: list[QueryResponse],
) -> list[list[str]]:
    results = []
    for response in responses:
        response_uuids = []
        for point in response.points:
            uuid_ = str(point.id).replace("-", "")
            response_uuids.append(uuid_)
        results.append(response_uuids)
    return results


def calculate_mrr(  # noqa: C901
    queries: Tensor,
    positives: Tensor,
    config: MRRConfig,
) -> dict[str, float]:
    """
    Calculate bidirectional Mean Reciprocal Rank using Qdrant collections.

    The function builds two temporary collections in Qdrant:
    one for query embeddings and one for positive embeddings. It then computes
    reciprocal rank in both directions (query->positive and positive->query)
    and returns the average value.

    Args:
        queries: Tensor of query embeddings with shape (batch_size, dim).
        positives: Tensor of positive embeddings with shape (batch_size, dim).
        config: MRR configuration including Qdrant connection and top-k.

    Returns:
        Dictionary with a single key "MRR" and float value in range [0.0, 1.0].
        If Qdrant is unavailable, returns {"MRR": 0.0}.
    """
    queries_list: list[list[float]] = queries.round(decimals=4).tolist()
    positives_list: list[list[float]] = positives.round(decimals=4).tolist()

    if len(queries_list) == 0 or len(positives_list) == 0:
        return {"MRR": 0.0}

    id2query = {}
    id2pos = {}
    for query_emb, pos_emb in zip(queries_list, positives_list, strict=True):
        uuid_ = uuid4().hex
        id2query[uuid_] = query_emb
        id2pos[uuid_] = pos_emb

    global client
    if client is None:
        temp_client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
        )
        try:
            temp_client.get_collections()
            client = temp_client
        except Exception as e:
            logger.warning(
                f"Failed to get collections from Qdrant:\n{e}\nMake sure Qdrant is running and accessible at {config.qdrant_host}:{config.qdrant_port}.\nReturning MRR=0.0."
            )
            return {"MRR": 0.0}

    VECTOR_TYPES = {"query", "pos"}
    VECTOR_SIMILARITY_MAPPING = {"query": "pos", "pos": "query"}
    VECTOR_TYPE_TO_COLLECTION_NAME = {
        "query": "lendik-queries",
        "pos": "lednik-positives",
    }
    VECTOR_TYPE_TO_VECTOR_MAPPING = {
        "query": id2query,
        "pos": id2pos,
    }

    ### Create collections and upload data ###
    for emb_type in VECTOR_TYPES:
        collection_name = VECTOR_TYPE_TO_COLLECTION_NAME[emb_type]

        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)

        resp = client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=queries.shape[-1], distance=Distance.COSINE
            ),
        )
        if not resp:
            raise RuntimeError(f"Failed to create collection {collection_name}")

        points = []

        for emb_id, emb in VECTOR_TYPE_TO_VECTOR_MAPPING[emb_type].items():
            point = PointStruct(id=emb_id, vector=emb)
            points.append(point)

        for points_batch in _batched(points, _QDRANT_UPLOAD_BATCH_SIZE):
            client.upload_points(
                collection_name=collection_name,
                points=points_batch,
                wait=True,
                parallel=1,
            )

    ### Calculate MRR ###
    mrr = 0.0
    for vec_type in VECTOR_TYPES:
        uuid_to_vectors = VECTOR_TYPE_TO_VECTOR_MAPPING[vec_type]

        query_requests = []
        target_ids = []
        for emb_id, emb in uuid_to_vectors.items():
            request = QueryRequest(query=emb, limit=config.mrr_top_k)
            query_requests.append(request)
            target_ids.append(emb_id)

        mrr_sum = 0.0

        for request_batch, target_ids_batch in zip(
            _batched(query_requests, _QDRANT_QUERY_BATCH_SIZE),
            _batched(target_ids, _QDRANT_QUERY_BATCH_SIZE),
            strict=True,
        ):
            query_responses = client.query_batch_points(
                collection_name=VECTOR_TYPE_TO_COLLECTION_NAME[
                    VECTOR_SIMILARITY_MAPPING[vec_type]
                ],
                requests=request_batch,
            )
            uuids = _extract_uuids_from_points(query_responses)
            for response_uuids, target_id in zip(uuids, target_ids_batch, strict=True):
                sample_mrr = _calculate_mrr_for_query(response_uuids, target_id)
                mrr_sum += sample_mrr

        mrr += mrr_sum / len(target_ids) / len(VECTOR_TYPES)
    return {"MRR": mrr}
