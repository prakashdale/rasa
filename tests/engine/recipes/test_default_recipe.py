import rasa.shared.utils.io
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.recipes.default_recipe import (
    DefaultV1Recipe,
    ProjectProvider,
    SchemaValidator,
    FinetuningValidator,
    NLUTrainingDataProvider,
)
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.storage.resource import Resource
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
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


def test_generate_train_graph():
    expected_train_schema = GraphSchema(
        {
            "project_provider": SchemaNode(
                needs={},
                uses=ProjectProvider,
                constructor_name="create",
                fn="provide",
                config={},
                eager=False,
                is_target=False,
                is_input=True,
                resource=None,
            ),
            "schema_validator": SchemaNode(
                needs={"importer": "project_provider"},
                uses=SchemaValidator,
                constructor_name="create",
                fn="validate",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "finetuning_validator": SchemaNode(
                needs={"importer": "schema_validator"},
                uses=FinetuningValidator,
                constructor_name="create",
                fn="validate",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "nlu_training_data_provider": SchemaNode(
                needs={"importer": "finetuning_validator"},
                uses=NLUTrainingDataProvider,
                constructor_name="create",
                fn="provide",
                config={},
                eager=False,
                # TODO: not always
                is_target=True,
                is_input=True,
                resource=None,
            ),
            "run_WhitespaceTokenizer0": SchemaNode(
                needs={"training_data": "nlu_training_data_provider"},
                uses=WhitespaceTokenizer,
                constructor_name="create",
                fn="process_training_data",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "train_RegexFeaturizer1": SchemaNode(
                needs={"training_data": "run_WhitespaceTokenizer0"},
                uses=RegexFeaturizer,
                constructor_name="create",
                fn="train",
                config={},
                eager=False,
                is_target=True,
                is_input=False,
                resource=None,
            ),
            "run_RegexFeaturizer1": SchemaNode(
                needs={
                    "training_data": "run_WhitespaceTokenizer0",
                    "resource": "train_RegexFeaturizer1",
                },
                uses=RegexFeaturizer,
                constructor_name="load",
                fn="process_training_data",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "run_LexicalSyntacticFeaturizer2": SchemaNode(
                needs={"training_data": "run_RegexFeaturizer1"},
                uses=LexicalSyntacticFeaturizer,
                constructor_name="create",
                fn="process_training_data",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "train_CountVectorsFeaturizer3": SchemaNode(
                needs={"training_data": "run_LexicalSyntacticFeaturizer2"},
                uses=CountVectorsFeaturizer,
                constructor_name="create",
                fn="train",
                config={},
                eager=False,
                is_target=True,
                is_input=False,
                resource=None,
            ),
            "run_CountVectorsFeaturizer3": SchemaNode(
                needs={
                    "training_data": "run_LexicalSyntacticFeaturizer2",
                    "resource": "train_CountVectorsFeaturizer3",
                },
                uses=CountVectorsFeaturizer,
                constructor_name="load",
                fn="process_training_data",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "train_CountVectorsFeaturizer4": SchemaNode(
                needs={"training_data": "run_CountVectorsFeaturizer3"},
                uses=CountVectorsFeaturizer,
                constructor_name="create",
                fn="train",
                config={"analyzer": "char_wb", "min_ngram": 1, "max_ngram": 4},
                eager=False,
                is_target=True,
                is_input=False,
                resource=None,
            ),
            "run_CountVectorsFeaturizer4": SchemaNode(
                needs={
                    "training_data": "run_CountVectorsFeaturizer3",
                    "resource": "train_CountVectorsFeaturizer4",
                },
                uses=CountVectorsFeaturizer,
                constructor_name="load",
                fn="process_training_data",
                config={"analyzer": "char_wb", "min_ngram": 1, "max_ngram": 4},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "train_DIETClassifier5": SchemaNode(
                needs={"training_data": "run_CountVectorsFeaturizer4",},
                uses=DIETClassifier,
                constructor_name="create",
                fn="train",
                config={"epochs": 100, "constrain_similarities": True},
                eager=False,
                is_target=True,
                is_input=False,
                resource=None,
            ),
            "run_EntitySynonymMapper6": SchemaNode(
                needs={"training_data": "run_CountVectorsFeaturizer4",},
                uses=EntitySynonymMapper,
                constructor_name="load",
                fn="process_training_data",
                config={},
                eager=False,
                is_target=False,
                is_input=False,
                resource=None,
            ),
            "train_ResponseSelector7": SchemaNode(
                needs={"training_data": "run_CountVectorsFeaturizer4",},
                uses=ResponseSelector,
                constructor_name="create",
                fn="train",
                config={"epochs": 100, "constrain_similarities": True},
                eager=False,
                is_target=True,
                is_input=False,
                resource=None,
            ),
        }
    )

    config = rasa.shared.utils.io.read_yaml_file(
        "rasa/shared/importers/default_config.yml"
    )
    recipe = Recipe.recipe_for_name(DefaultV1Recipe.name)
    train_schema, _ = recipe.schemas_for_config(config, {})

    for node_name, node in expected_train_schema.nodes.items():
        assert train_schema.nodes[node_name] == node

    assert train_schema == expected_train_schema
