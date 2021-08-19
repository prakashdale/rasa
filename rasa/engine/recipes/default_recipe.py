from __future__ import annotations

import copy
from typing import Dict, Text, Any, Tuple, Type, Optional, List

import dataclasses

from rasa.core.policies import SimplePolicyEnsemble
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.unexpected_intent_policy import UnexpecTEDIntentPolicy
from rasa.engine.graph import GraphSchema, GraphComponent, SchemaNode
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.storage.resource import Resource

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


class NLUMessageConverter(GraphComponent):
    pass


class RegexClassifier(GraphComponent):
    pass


class NLUPredictionToHistoryAdder(GraphComponent):
    pass


class TrackerToMessageConverter(GraphComponent):
    pass


pretrained_featurizers = [
    LexicalSyntacticFeaturizer,
    SpacyFeaturizer,
    MitieFeaturizer,
    LanguageModelFeaturizer,
    ConveRTFeaturizer,
]

policies_with_e2e_support = [TEDPolicy, UnexpecTEDIntentPolicy]

trainable_classifiers = [
    DIETClassifier,
    SklearnIntentClassifier,
    ResponseSelector,
]


class DefaultV1Recipe(Recipe):
    name = "default.v1"

    def schemas_for_config(
        self, config: Dict, cli_parameters: Dict[Text, Any]
    ) -> Tuple[GraphSchema, GraphSchema]:
        train_nodes, featurizers, tokenizer = self._create_train_nodes(config)
        predict_nodes = self._create_predict_nodes(
            config, featurizers, tokenizer, train_nodes
        )

        return GraphSchema(train_nodes), GraphSchema(predict_nodes)

    def _create_train_nodes(
        self, config: Dict[Text, Any]
    ) -> Tuple[Dict[Text, SchemaNode], List[Text], Optional[Text]]:
        train_config = copy.deepcopy(config)
        train_nodes = {
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
        }
        featurizers, tokenizer = self._add_nlu_train_nodes(train_config, train_nodes)
        self._add_core_train_nodes(train_config, train_nodes, tokenizer, featurizers)

        return train_nodes, featurizers, tokenizer

    def _add_nlu_train_nodes(
        self, train_config: Dict[Text, Any], train_nodes: Dict[Text, SchemaNode]
    ) -> Tuple[List[Text], Optional[Text]]:
        import rasa.nlu.registry

        train_nodes["nlu_training_data_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=NLUTrainingDataProvider,
            constructor_name="create",
            fn="provide",
            config={},
            # TODO: not always
            is_target=True,
            is_input=True,
        )

        last_run_node = "nlu_training_data_provider"
        last_feature_node = None
        tokenizer: Optional[Text] = None
        featurizers: List[Text] = []
        idx = 0

        # TODO: Trainable vs non trainable NLU node?
        for item in train_config["pipeline"]:
            component_name = item.pop("name")
            component = rasa.nlu.registry.get_component_class(component_name)
            component_name = f"{component.__name__}{idx}"
            if issubclass(component, Tokenizer):
                last_run_node = tokenizer = self._add_nlu_process_node(
                    train_nodes, component, component_name, last_run_node, item
                )
            elif issubclass(component, Featurizer):
                from_resource = None
                if component not in pretrained_featurizers:
                    from_resource = self._add_nlu_train_node(
                        train_nodes, component, component_name, last_run_node, item
                    )

                last_run_node = last_feature_node = self._add_nlu_process_node(
                    train_nodes,
                    component,
                    component_name,
                    last_run_node,
                    item,
                    from_resource=from_resource,
                )
                # Remember for End-to-End-Featurization
                featurizers.append(last_run_node)
            elif issubclass(component, IntentClassifier):
                if component in trainable_classifiers:
                    _ = self._add_nlu_train_node(
                        train_nodes, component, component_name, last_feature_node, item
                    )
                else:
                    # We don't need non trainable classifiers
                    continue
            elif issubclass(component, EntityExtractor):
                trainable_extractors = [CRFEntityExtractor, RegexEntityExtractor]
                if component in trainable_extractors:
                    # TODO: implement
                    pass
                else:
                    last_run_node = self._add_nlu_process_node(
                        train_nodes, component, component_name, last_run_node, item
                    )

            idx += 1

        return featurizers, tokenizer

    def _add_nlu_train_node(
        self,
        train_nodes: Dict[Text, SchemaNode],
        component: Type[GraphComponent],
        component_name: Text,
        last_run_node: Text,
        config: Dict[Text, Any],
    ) -> Text:
        train_node_name = f"train_{component_name}"
        train_nodes[train_node_name] = SchemaNode(
            needs={"training_data": last_run_node},
            uses=component,
            constructor_name="create",
            fn="train",
            config=config,
            is_target=True,
        )
        return train_node_name

    def _add_nlu_process_node(
        self,
        train_nodes: Dict[Text, SchemaNode],
        component_class: Type[GraphComponent],
        component_name: Text,
        last_run_node: Text,
        component_config: Dict[Text, Any],
        from_resource: Optional[Text] = None,
    ) -> Text:
        resource_needs = {}
        if from_resource:
            resource_needs = {"resource": from_resource}

        node_name = f"run_{component_name}"
        train_nodes[node_name] = SchemaNode(
            needs={"training_data": last_run_node, **resource_needs},
            uses=component_class,
            constructor_name="load",
            fn="process_training_data",
            config=component_config,
        )
        return node_name

    def _add_core_train_nodes(
        self,
        train_config: Dict[Text, Any],
        train_nodes: Dict[Text, SchemaNode],
        tokenizer: Optional[Text],
        featurizers: List[Text],
    ) -> None:
        train_nodes["domain_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=DomainProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_target=True,
            is_input=True,
        )
        train_nodes["domain_without_responses_provider"] = SchemaNode(
            needs={"domain": "domain_provider"},
            uses=DomainWithoutResponsesProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        train_nodes["story_graph_provider"] = SchemaNode(
            needs={"importer": "finetuning_validator"},
            uses=StoryGraphProvider,
            constructor_name="create",
            fn="provide",
            config={},
            is_input=True,
        )
        train_nodes["training_tracker_provider"] = SchemaNode(
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
        train_nodes["story_to_nlu_training_data_converter"] = SchemaNode(
            needs={
                "story_graph": "story_graph_provider",
                "domain": "domain_without_responses_provider",
            },
            uses=StoryToNLUTrainingDataConverter,
            constructor_name="create",
            fn="convert",
            config={},
            is_input=True,
        )
        if tokenizer is None:
            raise ValueError("should not happen")
        train_nodes[f"e2e_{tokenizer}"] = dataclasses.replace(
            train_nodes[tokenizer],
            needs={"training_data": "story_to_nlu_training_data_converter"},
        )
        last_node_name = f"e2e_{tokenizer}"
        for featurizer in featurizers:
            node = copy.deepcopy(train_nodes[featurizer])
            node.needs["training_data"] = last_node_name

            node_name = f"e2e_{featurizer}"
            train_nodes[node_name] = node
            last_node_name = node_name
        node_with_e2e_features = "end_to_end_features_provider"
        train_nodes[node_with_e2e_features] = SchemaNode(
            needs={"training_data": last_node_name,},
            uses=EndToEndFeaturesProvider,
            constructor_name="create",
            fn="provide",
            config={},
        )
        # Policies
        import rasa.core.registry

        idx = 0
        for item in train_config["policies"]:
            component_name = item.pop("name")
            component = rasa.core.registry.policy_from_module_path(component_name)

            node_name = f"train_{component.__name__}{idx}"

            e2e_needs = {}
            if component in policies_with_e2e_support:
                e2e_needs = {"end_to_end_features": node_with_e2e_features}
            train_nodes[node_name] = SchemaNode(
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

    def _create_predict_nodes(
        self,
        config: Dict[Text, SchemaNode],
        featurizers: List[Text],
        tokenizer: Optional[Text],
        train_nodes: Dict[Text, SchemaNode],
    ) -> Dict[Text, SchemaNode]:
        predict_config = copy.deepcopy(config)
        predict_nodes = {}
        last_run_node = self._add_nlu_predict_nodes(
            predict_config, predict_nodes, train_nodes
        )
        self._add_core_predict_nodes(
            predict_config,
            predict_nodes,
            last_run_node,
            train_nodes,
            tokenizer,
            featurizers,
        )
        return predict_nodes

    def _add_nlu_predict_nodes(
        self,
        predict_config: Dict[Text, Any],
        predict_nodes: Dict[Text, SchemaNode],
        train_nodes: Dict[Text, SchemaNode],
    ) -> Text:
        import rasa.nlu.registry

        predict_nodes["nlu_message_converter"] = SchemaNode(
            needs={},
            uses=NLUMessageConverter,
            constructor_name="create",
            fn="convert",
            config={},
            eager=True,
        )
        last_run_node = "nlu_message_converter"
        idx = 0
        for item in predict_config["pipeline"]:
            component_name = item.pop("name")
            component = rasa.nlu.registry.get_component_class(component_name)
            if issubclass(component, Tokenizer):
                node_name = f"run_{component.__name__}{idx}"
                predict_nodes[node_name] = dataclasses.replace(
                    train_nodes[node_name],
                    needs={"messages": last_run_node},
                    fn="process",
                    eager=True,
                )
                last_run_node = node_name
            elif issubclass(component, Featurizer):
                node_name = f"run_{component.__name__}{idx}"

                if component in pretrained_featurizers:
                    predict_nodes[node_name] = dataclasses.replace(
                        train_nodes[node_name],
                        needs={"messages": last_run_node},
                        fn="process",
                        eager=True,
                        is_target=False,
                    )
                else:
                    predict_nodes[node_name] = dataclasses.replace(
                        train_nodes[node_name],
                        needs={"messages": last_run_node},
                        fn="process",
                        eager=True,
                        resource=Resource(f"train_{component.__name__}{idx}"),
                    )
                last_run_node = node_name
            elif issubclass(component, IntentClassifier):
                if component in trainable_classifiers:
                    train_node_name = f"train_{component.__name__}{idx}"
                    node_name = f"run_{component.__name__}{idx}"
                    predict_nodes[node_name] = dataclasses.replace(
                        train_nodes[train_node_name],
                        needs={"messages": last_run_node},
                        constructor_name="load",
                        fn="process",
                        eager=True,
                        is_target=False,
                        resource=Resource(train_node_name),
                    )
                else:
                    node_name = f"run_{component.__name__}{idx}"
                    predict_nodes[node_name] = SchemaNode(
                        needs={"messages": last_run_node},
                        uses=component,
                        constructor_name="create",
                        fn="process",
                        config=item,
                        eager=True,
                    )
                last_run_node = node_name

            elif issubclass(component, EntityExtractor) and not issubclass(
                component, IntentClassifier
            ):
                trainable_extractors = [CRFEntityExtractor, RegexEntityExtractor]
                if component in trainable_extractors:
                    # TODO: implement
                    pass
                else:
                    node_name = f"run_{component.__name__}{idx}"
                    predict_nodes[node_name] = dataclasses.replace(
                        train_nodes[node_name],
                        needs={"messages": last_run_node},
                        uses=component,
                        constructor_name="load",
                        eager=True,
                        fn="process",
                    )
                    last_run_node = node_name

            idx += 1
        node_name = f"run_{RegexClassifier.__name__}"
        predict_nodes[node_name] = SchemaNode(
            needs={"messages": last_run_node},
            uses=RegexClassifier,
            constructor_name="create",
            fn="process",
            config={},
            eager=True,
        )
        return node_name

    def _add_core_predict_nodes(
        self,
        predict_config: Dict[Text, Any],
        predict_nodes: Dict[Text, SchemaNode],
        last_run_node: Text,
        train_nodes: Dict[Text, SchemaNode],
        tokenizer: Optional[Text],
        featurizers: List[Text],
    ) -> None:
        import rasa.core.registry

        predict_nodes["nlu_prediction_to_history_adder"] = SchemaNode(
            # TODO: I think there is a bug in our Dask Runner for this case as
            # the input will override `messages`
            needs={"messages": last_run_node},
            uses=NLUPredictionToHistoryAdder,
            constructor_name="create",
            fn="process",
            config={},
            eager=True,
        )
        predict_nodes["domain_provider"] = SchemaNode(
            needs={},
            uses=DomainProvider,
            constructor_name="load",
            fn="provide_persisted",
            config={},
            eager=True,
            resource=Resource("domain_provider"),
        )
        # End-to-end feature creation
        predict_nodes["tracker_to_message_converter"] = SchemaNode(
            needs={"tracker": "nlu_prediction_to_history_adder"},
            uses=TrackerToMessageConverter,
            constructor_name="create",
            fn="convert",
            config={},
            eager=True,
        )
        if tokenizer is None:
            raise ValueError("should not happen")
        predict_nodes[f"e2e_{tokenizer}"] = dataclasses.replace(
            predict_nodes[tokenizer],
            needs={"messages": "tracker_to_message_converter"},
        )
        last_node_name = f"e2e_{tokenizer}"
        for featurizer in featurizers:
            node = dataclasses.replace(
                predict_nodes[featurizer], needs={"messages": last_node_name}
            )

            node_name = f"e2e_{featurizer}"
            predict_nodes[node_name] = node
            last_node_name = node_name
        node_with_e2e_features = "end_to_end_features_provider"
        predict_nodes[node_with_e2e_features] = SchemaNode(
            needs={"messages": last_node_name,},
            uses=EndToEndFeaturesProvider,
            constructor_name="create",
            fn="provide_inference",
            config={},
            eager=True,
        )
        # policies
        idx = 0
        policies: List[Text] = []
        for item in predict_config["policies"]:
            component_name = item.pop("name")
            component = rasa.core.registry.policy_from_module_path(component_name)

            train_node_name = f"train_{component.__name__}{idx}"
            node_name = f"run_{component.__name__}{idx}"
            e2e_needs = {}
            if component in policies_with_e2e_support:
                e2e_needs = {"end_to_end_features": node_with_e2e_features}
            predict_nodes[node_name] = dataclasses.replace(
                train_nodes[train_node_name],
                needs={
                    "tracker": "nlu_prediction_to_history_adder",
                    "domain": "domain_provider",
                    **e2e_needs,
                },
                constructor_name="load",
                fn="predict_action_probabilities",
                eager=True,
                is_target=False,
                resource=Resource(train_node_name),
            )
            policies.append(node_name)
            idx += 1
        predict_nodes["select_prediction"] = SchemaNode(
            needs={f"policy{idx}": name for idx, name in enumerate(policies)},
            uses=SimplePolicyEnsemble,
            constructor_name="create",
            fn="select",
            config={},
            eager=True,
        )
