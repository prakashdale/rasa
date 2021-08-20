from typing import Text

import pytest

import rasa.shared.utils.io
from rasa.core.policies import SimplePolicyEnsemble
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.unexpected_intent_policy import UnexpecTEDIntentPolicy
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.recipes.default_recipe import (
    DefaultV1Recipe,
    ProjectProvider,
    SchemaValidator,
    FinetuningValidator,
    NLUTrainingDataProvider,
    DomainProvider,
    DomainWithoutResponsesProvider,
    StoryGraphProvider,
    TrainingTrackerProvider,
    StoryToNLUTrainingDataConverter,
    EndToEndFeaturesProvider,
    NLUMessageConverter,
    RegexClassifier,
    NLUPredictionToHistoryAdder,
    TrackerToMessageConverter,
)
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.storage.resource import Resource
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


def test_recipe_for_name():
    recipe = Recipe.recipe_for_name("default.v1")
    assert isinstance(recipe, DefaultV1Recipe)


@pytest.mark.parametrize(
    "config_path, expected_train_schema_path, expected_predict_schema_path",
    [
        (
            "rasa/shared/importers/default_config.yml",
            "data/graph_schemas/default_config_train_schema.yml",
            "data/graph_schemas/default_config_predict_schema.yml",
        )
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

    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert predict_schema == expected_predict_schema

    for node_name, node in expected_predict_schema.nodes.items():
        assert predict_schema.nodes[node_name] == node

    assert predict_schema == expected_predict_schema
