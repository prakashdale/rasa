from typing import Text

import pytest

import rasa.shared.utils.io
from rasa.engine.graph import GraphSchema
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.recipes.recipe import Recipe


def test_recipe_for_name():
    recipe = Recipe.recipe_for_name("default.v1")
    assert isinstance(recipe, DefaultV1Recipe)


@pytest.mark.parametrize(
    "config_path, expected_train_schema_path, expected_predict_schema_path",
    [
        # The default config is the config which most users run
        (
            "rasa/shared/importers/default_config.yml",
            "data/graph_schemas/default_config_train_schema.yml",
            "data/graph_schemas/default_config_predict_schema.yml",
        ),
        # A config which uses Spacy and Duckling does not have Core model
        (
            "data/test_config/config_pretrained_embeddings_spacy_duckling.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_spacy_duckling_train_schema.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_spacy_duckling_predict_schema.yml",
        ),
        # A minimal NLU config without Core model
        (
            "data/test_config/keyword_classifier_config.yml",
            "data/graph_schemas/keyword_classifier_config_train_schema.yml",
            "data/graph_schemas/keyword_classifier_config_predict_schema.yml",
        ),
        # A config which uses Mitie and does not have Core model
        (
            "data/test_config/config_pretrained_embeddings_mitie.yml",
            "data/graph_schemas/config_pretrained_embeddings_mitie_train_schema.yml",
            "data/graph_schemas/"
            "config_pretrained_embeddings_mitie_predict_schema.yml",
        ),
        # A core only model
        (
            "data/test_config/max_hist_config.yml",
            "data/graph_schemas/max_hist_config_train_schema.yml",
            "data/graph_schemas/max_hist_config_predict_schema.yml",
        ),
    ],
)
def test_generate_predict_graph(
    config_path: Text,
    expected_train_schema_path: Text,
    expected_predict_schema_path: Text,
):
    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        expected_train_schema_path
    )
    expected_train_schema = GraphSchema.from_dict(expected_schema_as_dict)

    expected_schema_as_dict = rasa.shared.utils.io.read_yaml_file(
        expected_predict_schema_path
    )
    expected_predict_schema = GraphSchema.from_dict(expected_schema_as_dict)

    config = rasa.shared.utils.io.read_yaml_file(config_path)

    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    train_schema, predict_schema = recipe.schemas_for_config(config, {})

    rasa.shared.utils.io.write_yaml(train_schema.as_dict(), "train_schema.yml")
    rasa.shared.utils.io.write_yaml(predict_schema.as_dict(), "predict_schema.yml")

    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert train_schema == expected_train_schema

    for node_name, node in expected_predict_schema.nodes.items():
        assert predict_schema.nodes[node_name] == node

    assert predict_schema == expected_predict_schema
