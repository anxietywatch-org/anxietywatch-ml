"""Recursive key normalization for backend Mongo documents.

Backend documents carry the same field in three spellings:

- ``EventId``      PascalCase (real .NET/Mongo serialization)
- ``eventId``      camelCase  (synthetic docs / transport contract)
- ``event_id``     snake_case (internal ML schemas)

:func:`normalize_keys` resolves any of these to the snake_case field name via
a case-insensitive alphanumeric canonical form, recursing into nested models
(``Features``, ``Baseline``, ``Samples``, ``Quality``, ...) driven by the
target pydantic model's declared fields. Unknown keys are dropped.

It also enforces the identity rule: the Mongo ``userId`` (lowercase, injected
by the backend repository) is the canonical authenticated identity. If several
``userId``/``UserId``/``user_id`` spellings coexist with different non-null
values, :class:`IdentityMismatchError` is raised so the caller can exclude the
document instead of trusting stored/imported data blindly.
"""

from typing import Any, get_args, get_origin


class IdentityMismatchError(ValueError):
    """Auth ``userId`` and DTO ``UserId`` coexist with different values."""


def canon_key(key: str) -> str:
    """Lowercase alphanumeric canonical form: EventId/eventId/event_id -> eventid."""
    return "".join(ch for ch in key.lower() if ch.isalnum())


def resolve_model_type(annotation: Any):
    """Return the pydantic model referenced by a field annotation, if any."""
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        resolved = resolve_model_type(arg)
        if resolved is not None:
            return resolved
    return None


def normalize_keys(data: dict, model_type) -> dict:
    """Recursively map the keys of ``data`` to the snake_case fields of ``model_type``."""
    field_canons = {canon_key(name): name for name in model_type.model_fields}
    normalized = {}
    for key, value in data.items():
        field_name = field_canons.get(canon_key(key))
        if field_name is None:
            continue
        sub_model = resolve_model_type(model_type.model_fields[field_name].annotation)
        if sub_model is not None and isinstance(value, dict):
            normalized[field_name] = normalize_keys(value, sub_model)
        elif sub_model is not None and isinstance(value, list):
            normalized[field_name] = [
                normalize_keys(item, sub_model) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            normalized[field_name] = value
    return normalized


def resolve_identity(data: dict):
    """Return the canonical authenticated ``userId`` or raise IdentityMismatchError."""
    values = set()
    for key, value in data.items():
        if canon_key(key) == "userid" and value is not None:
            values.add(str(value))
    if len(values) > 1:
        raise IdentityMismatchError(f"identity mismatch: {sorted(values)}")
    return values.pop() if values else None