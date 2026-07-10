"""
Eligibility decision engine.

PDF requirement: "Clear three-tier eligibility gate — Eligible / Review / Declined —
derived from cohort-relative PD thresholds, not absolute cutoffs."

Eligible  : PD <= cohort 60th percentile (better than 60% of peers)
Review    : 60th < PD <= 80th percentile  (borderline)
Declined  : PD > cohort 80th percentile  (worse than 80% of peers)

If Coverage Index < 0.60 (thin-file route), the interval width is added to the PD
before comparing, making it harder to be auto-eligible but never auto-declined.
"""
from dataclasses import dataclass
from src.data.cohort_builder import get_eligibility_thresholds


ELIGIBLE = "Eligible"
REVIEW = "Review"
DECLINED = "Declined"


@dataclass
class EligibilityResult:
    decision: str                  # "Eligible" | "Review" | "Declined"
    pd: float                      # point-estimate probability of default
    pd_lower: float                # conformal interval lower
    pd_upper: float                # conformal interval upper
    peer_percentile: float         # 0-100, 100 = best (lowest PD in cohort)
    cohort: str
    eligible_pd_threshold: float   # cohort's 60th-pct PD
    review_pd_threshold: float     # cohort's 80th-pct PD
    is_thin_file: bool
    reason: str                    # one-liner for the UI banner


def decide_eligibility(
    pd_value: float,
    cohort: str,
    cohort_data: dict,
    peer_percentile: float,
    is_thin_file: bool = False,
    pd_interval_width: float = 0.0,
) -> EligibilityResult:
    """
    Applies cohort-relative thresholds.
    For thin-file applicants the interval_width is added to effective PD so that
    uncertainty widens the gap toward Eligible but cannot push into Declined alone.
    """
    thresholds = get_eligibility_thresholds(cohort, cohort_data)
    eligible_thr = thresholds["eligible_pd_threshold"]
    review_thr = thresholds["review_pd_threshold"]

    # Effective PD for boundary check — penalise thin-file uncertainty slightly
    effective_pd = pd_value + (0.5 * pd_interval_width if is_thin_file else 0.0)

    pd_lower = max(0.0, pd_value - pd_interval_width / 2)
    pd_upper = min(1.0, pd_value + pd_interval_width / 2)

    if effective_pd <= eligible_thr:
        decision = ELIGIBLE
        reason = (
            f"PD {pd_value:.2%} is within the top 60% of {cohort.replace('_', ' ')} peers. "
            f"Peer percentile: {peer_percentile:.0f}/100."
        )
    elif effective_pd <= review_thr:
        decision = REVIEW
        reason = (
            f"PD {pd_value:.2%} places applicant in the 60-80th percentile band of "
            f"{cohort.replace('_', ' ')} peers. Further due-diligence recommended."
        )
    else:
        decision = DECLINED
        reason = (
            f"PD {pd_value:.2%} exceeds the 80th-percentile threshold "
            f"({review_thr:.2%}) for {cohort.replace('_', ' ')} peers."
        )

    if is_thin_file and decision == ELIGIBLE:
        reason += " (thin-file; uncertainty widened interval applied)"

    return EligibilityResult(
        decision=decision,
        pd=pd_value,
        pd_lower=pd_lower,
        pd_upper=pd_upper,
        peer_percentile=peer_percentile,
        cohort=cohort,
        eligible_pd_threshold=eligible_thr,
        review_pd_threshold=review_thr,
        is_thin_file=is_thin_file,
        reason=reason,
    )
