"""Smart Fertilizer Advisor — precision nutrient management package."""


def create_app(*args, **kwargs):
    """Lazily import and construct the Flask application."""
    from smart_fertilizer.app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
