"""
LLM Agent: the reasoning layer of the pipeline.

Given the complaint text plus the fused model output (category, severity,
confidence), this module:
  1. Retrieves the relevant company policy for that category.
  2. Calls an LLM (Claude, via the Anthropic API) with the complaint,
     the model's classification, and the policy text, asking it to reason
     step by step and produce a final structured recommendation.
  3. Falls back to a deterministic rule-based recommender if no API key
     is configured, so the pipeline still runs end-to-end offline/for demos.

Set the ANTHROPIC_API_KEY environment variable to enable real LLM reasoning.
"""
import os
import json

POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies", "company_policy.json")

VALID_ACTIONS = {"Refund", "Replacement", "Escalation"}


def load_policies():
    with open(POLICY_PATH, "r") as f:
        return json.load(f)


def retrieve_policy(category: str) -> dict:
    """Retrieval step: fetch the policy entry for a given complaint category."""
    policies = load_policies()
    return policies.get(category, {
        "policy": "No specific policy found for this category. Escalate for manual review.",
        "default_actions": {"Low": "Escalation", "Medium": "Escalation", "High": "Escalation"},
    })


def _rule_based_recommendation(category, severity, confidence, policy_entry, image_authenticity=None):
    """Deterministic fallback used when no LLM API key is available."""
    action_text = policy_entry.get("default_actions", {}).get(severity, "Escalation")

    if "refund" in action_text.lower():
        action = "Refund"
    elif "replacement" in action_text.lower():
        action = "Replacement"
    else:
        action = "Escalation"

    # An image flagged as digitally manipulated overrides everything else -
    # this always goes to a human for fraud review.
    if image_authenticity and not image_authenticity.get("is_real", True):
        action = "Escalation"
        reason = (
            f"The attached photo was flagged as potentially manipulated by the image "
            f"authenticity CNN (fake probability: {image_authenticity.get('fake_probability', 0):.2f}). "
            f"This case is escalated for manual fraud review regardless of the text classification."
        )
        return {
            "action": action,
            "reasoning": reason,
            "policy_used": policy_entry.get("policy", ""),
            "source": "rule_based_fallback",
        }

    # Low classifier confidence should always be routed to a human.
    if confidence < 0.5:
        action = "Escalation"
        reason = (
            f"Model confidence ({confidence:.2f}) is below the 0.50 threshold, so this "
            f"case is routed to a human agent for manual review rather than auto-resolved."
        )
    else:
        reason = (
            f"Category '{category}' with severity '{severity}' maps to company policy "
            f"guidance: \"{action_text}\". Classifier confidence: {confidence:.2f}."
        )
        if image_authenticity:
            reason += (
                f" Attached photo passed authenticity check "
                f"(real probability: {image_authenticity.get('real_probability', 0):.2f})."
            )

    return {
        "action": action,
        "reasoning": reason,
        "policy_used": policy_entry.get("policy", ""),
        "source": "rule_based_fallback",
    }


def analyze_with_llm(complaint_text: str, category: str, severity: str, confidence: float,
                      image_authenticity: dict = None,
                      model: str = "claude-sonnet-4-6") -> dict:
    """
    Reasoning step: ask Claude to weigh the complaint, the model's classification,
    the retrieved policy, and (if provided) the image-authenticity CNN's verdict on
    any attached photo, then produce a final recommendation.
    Falls back to rule-based logic if ANTHROPIC_API_KEY is not set or the call fails.
    """
    policy_entry = retrieve_policy(category)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return _rule_based_recommendation(category, severity, confidence, policy_entry, image_authenticity)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (
            "You are a customer support decision agent for an e-commerce company. "
            "You are given a customer complaint, an automated classification "
            "(category, severity, confidence) produced by a CNN/RNN/LSTM ensemble, "
            "the relevant company policy text, and (optionally) a verdict from an "
            "image-authenticity CNN on whether an attached photo appears digitally "
            "manipulated. If the photo is flagged as likely fake, you should almost "
            "always recommend Escalation for manual fraud review, regardless of the "
            "text classification. Reason step by step, then output ONLY a JSON object "
            "(no markdown, no preamble) with exactly these keys: "
            '"action" (one of "Refund", "Replacement", "Escalation"), '
            '"reasoning" (2-4 sentences explaining the decision, referencing the policy '
            "and the image check if one was provided)."
        )

        image_section = "No photo was attached to this complaint."
        if image_authenticity:
            image_section = (
                f"Attached photo authenticity check: {image_authenticity.get('label', 'Unknown')} "
                f"(real probability: {image_authenticity.get('real_probability', 0):.2f}, "
                f"fake probability: {image_authenticity.get('fake_probability', 0):.2f})."
            )

        user_prompt = f"""Complaint: "{complaint_text}"

Automated classification:
- Category: {category}
- Severity: {severity}
- Confidence: {confidence:.2f}

{image_section}

Company policy for this category:
{policy_entry.get('policy', '')}

Decide the final recommendation."""

        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_text)

        action = parsed.get("action", "Escalation")
        if action not in VALID_ACTIONS:
            action = "Escalation"

        return {
            "action": action,
            "reasoning": parsed.get("reasoning", ""),
            "policy_used": policy_entry.get("policy", ""),
            "source": "llm_agent",
        }

    except Exception as e:
        fallback = _rule_based_recommendation(category, severity, confidence, policy_entry, image_authenticity)
        fallback["llm_error"] = str(e)
        return fallback
