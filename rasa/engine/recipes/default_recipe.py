from __future__ import annotations

from typing import Dict, Text, Any, Tuple

from rasa.engine.graph import GraphSchema
from rasa.engine.recipes.recipe import Recipe


class DefaultV1Recipe(Recipe):
    name = "default.v1"

    def schemas_for_config(
        self, config: Dict, cli_parameters: Dict[Text, Any]
    ) -> Tuple[GraphSchema, GraphSchema]:
        pass
