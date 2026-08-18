"""Label policy for the ground-truth dataset builder.

The derived binary target answers: "did the user request support after a
heuristic detector event?" It is NOT "did the user have anxiety".

The original user response is ALWAYS preserved verbatim. ``target_support_requested``
is a derived 0/1 view for training only and never replaces the stored response.

Label policy:
    SUPPORT_REQUESTED  -> target_support_requested=1, response_category=SUPPORT_REQUESTED
    ACTIVITY_CONFIRMED -> target_support_requested=0, response_category=PHYSICAL_ACTIVITY
    USER_OK            -> target_support_requested=0, response_category=SELF_REPORTED_OK
"""

from dataclasses import dataclass
from typing import Final

PRIMARY_RESPONSES: Final[tuple[str, ...]] = (
    "ACTIVITY_CONFIRMED",
    "USER_OK",
    "SUPPORT_REQUESTED",
)

SUPPORT_REQUESTED: Final[str] = "SUPPORT_REQUESTED"
PHYSICAL_ACTIVITY: Final[str] = "PHYSICAL_ACTIVITY"
SELF_REPORTED_OK: Final[str] = "SELF_REPORTED_OK"

_RESPONSE_CATEGORY: Final[dict[str, str]] = {
    SUPPORT_REQUESTED: SUPPORT_REQUESTED,
    "ACTIVITY_CONFIRMED": PHYSICAL_ACTIVITY,
    "USER_OK": SELF_REPORTED_OK,
}


@dataclass(frozen=True)
class LabelPolicyResult:
    """Result of applying the label policy to a primary decision response."""

    response: str  # original response, preserved verbatim
    target_support_requested: int  # derived: 1 if SUPPORT_REQUESTED else 0
    response_category: str  # derived category


def apply_label_policy(response: str) -> LabelPolicyResult:
    """Map a primary user response to the derived label view."""
    normalized = response.strip().upper()
    if normalized not in PRIMARY_RESPONSES:
        raise ValueError(
            f"Unsupported response for label policy: {response!r}. "
            f"Allowed primary responses: {', '.join(PRIMARY_RESPONSES)}."
        )
    return LabelPolicyResult(
        response=response,
        target_support_requested=1 if normalized == SUPPORT_REQUESTED else 0,
        response_category=_RESPONSE_CATEGORY[normalized],
    )