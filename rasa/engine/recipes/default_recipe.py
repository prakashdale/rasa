from __future__ import annotations

import copy
from typing import Dict, Text, Any, Tuple, Type, Optional, List

import dataclasses

from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.unexpected_intent_policy import UnexpecTEDIntentPolicy
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


class DomainProvider(GraphComponent):
    pass


class DomainWithoutResponsesProvider(GraphComponent):
    pass


class StoryGraphProvider(GraphComponent):
    pass


class TrainingTrackerProvider(GraphComponent):
    pass


class StoryToNLUTrainingDataConverter(GraphComponent):
    pass


class EndToEndFeaturesProvider(GraphComponent):
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

        tokenizer: Optional[Text] = None
        featurizers: List[Text] = []

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
                last_run_node = tokenizer = node_name
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
                # Remember for End-to-End-Featurization
                featurizers.append(last_run_node)
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

        # This starts the Core part of the graph
        nodes["domain_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=DomainProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_target=True,
            is_input=True,
        )
        nodes["domain_without_responses_provider"] = SchemaNode(
            needs={"domain": "domain_provider"},
            uses=DomainWithoutResponsesProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        nodes["story_graph_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=StoryGraphProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        nodes["training_tracker_provider"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_without_responses_provider",
            },
            uses=TrainingTrackerProvider,
            constructor_name="create",
            fn="provide",
            config={},
        )

        # End-to-End feature creation
        nodes["story_to_nlu_training_data_converter"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_without_responses_provider",
            },
            uses=StoryToNLUTrainingDataConverter,
            constructor_name="create",
            fn="convert",
            config={},
            is_input=True,
            resource=None,
        )
        if tokenizer is None:
            raise ValueError("should not happen")

        nodes[f"e2e_{tokenizer}"] = dataclasses.replace(
            nodes[tokenizer],
            needs={"training_data": "story_to_nlu_training_data_converter"},
        )
        last_node_name = f"e2e_{tokenizer}"
        for featurizer in featurizers:
            node = copy.deepcopy(nodes[featurizer])
            node.needs["training_data"] = last_node_name

            node_name = f"e2e_{featurizer}"
            nodes[node_name] = node
            last_node_name = node_name

        node_with_e2e_features = "end_to_end_features_provider"
        nodes[node_with_e2e_features] = SchemaNode(
            needs={"training_data": last_node_name,},
            uses=EndToEndFeaturesProvider,
            constructor_name="create",
            fn="provide",
            config={},
            resource=None,
        )

        # Policies
        import rasa.core.registry

        # last_run_node = "nlu_training_data_provider"
        # last_feature_node = None

        idx = 0
        policies_with_e2e_support = [TEDPolicy, UnexpecTEDIntentPolicy]
        for item in config["policies"]:
            component_name = item.pop("name")
            component = rasa.core.registry.policy_from_module_path(component_name)

            node_name = f"train_{component.__name__}{idx}"

            e2e_needs = {}
            if component in policies_with_e2e_support:
                e2e_needs = {"end_to_end_features": node_with_e2e_features}
            nodes[node_name] = SchemaNode(
                needs={
                    "training_trackers": "training_tracker_provider",
                    "domain": "domain_without_responses_provider",
                    **e2e_needs,
                },
                uses=component,
                constructor_name="create",
                fn="train",
                is_target=True,
                config=item,
            )
            idx += 1

        return GraphSchema(nodes), GraphSchema({})
