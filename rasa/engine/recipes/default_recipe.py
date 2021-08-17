from __future__ import annotations

from typing import Dict, Text, Any, Tuple, Type, Optional

from rasa.engine.graph import GraphSchema, GraphComponent, SchemaNode
from rasa.engine.recipes.recipe import Recipe

from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.tokenizer import Tokenizer

# TODO: Remove once they are implemented
class ProjectProvider(GraphComponent):
    pass


class SchemaValidator(GraphComponent):
    pass


class FinetuningValidator(GraphComponent):
    pass


class NLUTrainingDataProvider(GraphComponent):
    pass


def _create_train(
    component_class: Type,
    config: Dict[Text, Any],
    idx: int,
    previous_node: Text,
    last_feature_node: Optional[Text],
) -> Tuple[Dict[Text, SchemaNode], Text, Optional[Text]]:
    # TODO: Idea for refactoring stuff
    pass


class DefaultV1Recipe(Recipe):
    name = "default.v1"

    def schemas_for_config(
        self, config: Dict, cli_parameters: Dict[Text, Any]
    ) -> Tuple[GraphSchema, GraphSchema]:
        nodes = {
            "project_provider": SchemaNode(
                needs={},
                uses=ProjectProvider,
                constructor_name="create",
                fn="provide",
                config={},
                is_input=True,
            ),
            "schema_validator": SchemaNode(
                needs={"importer": "project_provider"},
                uses=SchemaValidator,
                constructor_name="create",
                fn="validate",
                config={},
            ),
            "finetuning_validator": SchemaNode(
                needs={"importer": "schema_validator"},
                uses=FinetuningValidator,
                constructor_name="create",
                fn="validate",
                config={},
            ),
            # This starts the NLU part of the graph
            "nlu_training_data_provider": SchemaNode(
                needs={"importer": "finetuning_validator"},
                uses=NLUTrainingDataProvider,
                constructor_name="create",
                fn="provide",
                config={},
                # TODO: not always
                is_target=True,
                is_input=True,
            ),
        }

        import rasa.nlu.registry

        last_run_node = "nlu_training_data_provider"
        last_feature_node = None

        idx = 0
        for item in config["pipeline"]:
            component_name = item.pop("name")
            component = rasa.nlu.registry.get_component_class(component_name)
            if issubclass(component, Tokenizer):
                node_name = f"run_{component.__name__}{idx}"
                nodes[node_name] = SchemaNode(
                    needs={"training_data": last_run_node},
                    uses=component,
                    constructor_name="create",
                    fn="process_training_data",
                    config=item,
                )
                last_run_node = node_name
            elif issubclass(component, Featurizer):
                pretrained_featurizer = [
                    LexicalSyntacticFeaturizer,
                    SpacyFeaturizer,
                    MitieFeaturizer,
                    LanguageModelFeaturizer,
                    ConveRTFeaturizer,
                ]
                if component in pretrained_featurizer:
                    node_name = f"run_{component.__name__}{idx}"
                    nodes[node_name] = SchemaNode(
                        needs={"training_data": last_run_node},
                        uses=component,
                        constructor_name="create",
                        fn="process_training_data",
                        config=item,
                    )
                    last_run_node = node_name
                else:
                    train_node_name = f"train_{component.__name__}{idx}"
                    nodes[train_node_name] = SchemaNode(
                        needs={"training_data": last_run_node},
                        uses=component,
                        constructor_name="create",
                        fn="train",
                        config=item,
                        is_target=True,
                    )

                    node_name = f"run_{component.__name__}{idx}"
                    nodes[node_name] = SchemaNode(
                        needs={
                            "training_data": last_run_node,
                            "resource": train_node_name,
                        },
                        uses=component,
                        constructor_name="load",
                        fn="process_training_data",
                        config=item,
                    )
                    last_run_node = last_feature_node = node_name
            elif issubclass(component, IntentClassifier):
                trainable_classifiers = [
                    DIETClassifier,
                    SklearnIntentClassifier,
                    ResponseSelector,
                ]
                if component in trainable_classifiers:
                    node_name = f"train_{component.__name__}{idx}"
                    nodes[node_name] = SchemaNode(
                        needs={"training_data": last_feature_node},
                        uses=component,
                        constructor_name="create",
                        fn="train",
                        config=item,
                        is_target=True,
                    )
                else:
                    # We don't need non trainable classifiers
                    continue
            elif issubclass(component, EntityExtractor):
                trainable_extractors = [CRFEntityExtractor, RegexEntityExtractor]
                if component in trainable_extractors:
                    pass
                else:
                    node_name = f"run_{component.__name__}{idx}"
                    nodes[node_name] = SchemaNode(
                        needs={"training_data": last_run_node},
                        uses=component,
                        constructor_name="load",
                        fn="process_training_data",
                        config=item,
                    )
                    last_run_node = node_name

            idx += 1

        return GraphSchema(nodes), GraphSchema({})
