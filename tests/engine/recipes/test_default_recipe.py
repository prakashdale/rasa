from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.recipes.recipe import Recipe


def test_recipe_for_name():
    recipe = Recipe.recipe_for_name("default.v1")
    assert isinstance(recipe, DefaultV1Recipe)
