from src.criminalNetwork.utils.common import normalize_neo4j_uri


def test_aura_uri_rewrites_when_self_signed_cert_is_required():
    uri = "neo4j+s://c4f82e4b.databases.neo4j.io"
    assert normalize_neo4j_uri(uri, True) == "neo4j+ssc://c4f82e4b.databases.neo4j.io"


def test_plain_uri_is_kept_when_not_using_self_signed_mode():
    uri = "neo4j+s://localhost:7687"
    assert normalize_neo4j_uri(uri, False) == uri
