from mlflow_project.config import DatabaseConfig


def test_database_url_encodes_credentials() -> None:
    config = DatabaseConfig(
        host="db.example",
        port=5432,
        database="models",
        user="user@company",
        password="p@ss/word:42",
    )

    rendered = config.url.render_as_string(hide_password=False)

    assert "user%40company" in rendered
    assert "p%40ss%2Fword%3A42" in rendered
    assert "p@ss/word:42" not in rendered
