"""Reward Language Model: process-level reward using a strong LLM as judge.

Scores reasoning chains on three dimensions:
  - perception_score: does the model correctly read/interpret the visual input?
  - reasoning_score: is the logic chain coherent and valid?
  - groundedness_score: are there hallucinated objects/attributes?

Supports:
  - Online scoring via API (for offline DPO/KTO or small-scale evaluation)
  - Batch scoring for generating training data for a local PRM
"""

import json
import os
import re

SCORING_PROMPT = """\
You are an expert evaluator of visual reasoning chains. You will be given:
1. A visual reasoning problem (the question)
2. The ground truth answer
3. A model's response containing its reasoning and answer

Score the response on three dimensions (each 0-5):

**perception_score** (0-5): Does the model correctly perceive and describe the visual elements?
- 0: Completely wrong perception, describes objects/features not in the image
- 1: Major perception errors (wrong shapes, wrong counts, wrong spatial relations)
- 2: Some correct elements but significant misperceptions
- 3: Mostly correct perception with minor errors
- 4: Accurate perception with very minor issues
- 5: Perfect perception of all relevant visual elements

**reasoning_score** (0-5): Is the logical reasoning chain coherent and valid?
- 0: No reasoning or completely incoherent
- 1: Reasoning present but fundamentally flawed logic
- 2: Some valid steps but major logical gaps or errors
- 3: Mostly sound reasoning with minor logical issues
- 4: Strong reasoning with very minor issues
- 5: Perfect logical chain from observations to conclusion

**groundedness_score** (0-5): Is the response free of hallucinations?
- 0: Heavily hallucinated (invents objects, numbers, relationships)
- 1: Multiple hallucinated elements
- 2: Some hallucinated details mixed with real ones
- 3: Mostly grounded with minor unsupported claims
- 4: Nearly fully grounded, very minor issues
- 5: Completely grounded in the visual input

**diagnosis**: What is the primary error type?
- "correct": Answer is correct with sound reasoning
- "perception_error": Wrong answer due to misreading the visual input
- "reasoning_error": Correct perception but flawed reasoning
- "both": Both perception and reasoning errors
- "hallucination": Answer based on hallucinated content
- "format_error": Could not extract answer due to formatting issues

Respond ONLY with a JSON object:
```json
{
  "perception_score": <int 0-5>,
  "reasoning_score": <int 0-5>,
  "groundedness_score": <int 0-5>,
  "diagnosis": "<string>",
  "brief_explanation": "<1-2 sentences>"
}
```

---

**Question:** {question}

**Ground truth answer:** {ground_truth}

**Model response:**
{response}
"""


def build_scoring_prompt(question: str, ground_truth: str, response: str) -> str:
    """Build the scoring prompt for the Reward LM."""
    return SCORING_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        response=response,
    )


def parse_reward_lm_output(output: str) -> dict:
    """Parse the JSON output from the Reward LM.

    Returns dict with scores and diagnosis, or defaults on parse failure.
    """
    # Try to extract JSON from the output
    json_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            # Validate and clamp scores
            for key in ["perception_score", "reasoning_score", "groundedness_score"]:
                if key in result:
                    result[key] = max(0, min(5, int(result[key])))
                else:
                    result[key] = 0
            if result.get("diagnosis") not in {
                "correct", "perception_error", "reasoning_error", "both",
                "hallucination", "format_error",
            }:
                result["diagnosis"] = "unknown"
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Default on parse failure
    return {
        "perception_score": 0,
        "reasoning_score": 0,
        "groundedness_score": 0,
        "diagnosis": "parse_error",
        "brief_explanation": "Failed to parse Reward LM output",
    }


def scores_to_reward(scores: dict, weights: dict | None = None) -> float:
    """Convert structured scores to a single reward value in [0, 1].

    Default weights emphasize reasoning and groundedness over perception.
    """
    if weights is None:
        weights = {
            "perception_score": 0.3,
            "reasoning_score": 0.4,
            "groundedness_score": 0.3,
        }

    total = 0.0
    for key, weight in weights.items():
        total += (scores.get(key, 0) / 5.0) * weight

    return total


class RewardLMScorer:
    """Score reasoning chains using an LLM API.

    Supports Anthropic (Claude) and OpenAI-compatible APIs.
    """

    def __init__(self, provider: str = "anthropic", model: str | None = None):
        self.provider = provider
        if provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            self.model = model or "gpt-4o"
        elif provider == "qwen":
            # Qwen2.5-VL-72B via DashScope or compatible API
            self.model = model or "qwen2.5-vl-72b-instruct"
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.provider in ("openai", "qwen"):
            import openai
            if self.provider == "qwen":
                self._client = openai.OpenAI(
                    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            else:
                self._client = openai.OpenAI()
        return self._client

    def score(self, question: str, ground_truth: str, response: str) -> dict:
        """Score a single reasoning chain.

        Returns dict with perception_score, reasoning_score, groundedness_score,
        diagnosis, and brief_explanation.
        """
        prompt = build_scoring_prompt(question, ground_truth, response)
        client = self._get_client()

        try:
            if self.provider == "anthropic":
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = msg.content[0].text
            else:
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = resp.choices[0].message.content

            return parse_reward_lm_output(output)
        except Exception as e:
            return {
                "perception_score": 0,
                "reasoning_score": 0,
                "groundedness_score": 0,
                "diagnosis": "api_error",
                "brief_explanation": str(e),
            }

    def score_batch(
        self,
        questions: list[str],
        ground_truths: list[str],
        responses: list[str],
    ) -> list[dict]:
        """Score a batch of reasoning chains sequentially.

        For large batches, consider using async or the offline pipeline instead.
        """
        results = []
        for q, gt, r in zip(questions, ground_truths, responses):
            results.append(self.score(q, gt, r))
        return results


def process_reward(completions: list[str], solution: list[str], **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward function using Reward LM.

    NOTE: This calls an API for each completion — expensive for online GRPO.
    Use offline scoring (score_batch + DPO) or a distilled local PRM instead.
    """
    scorer = RewardLMScorer(
        provider=os.environ.get("REWARD_LM_PROVIDER", "anthropic"),
        model=os.environ.get("REWARD_LM_MODEL"),
    )

    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []

    for completion, gt, prompt in zip(completions, solution, prompts):
        if isinstance(completion, dict):
            completion = completion.get("content", "")
        elif isinstance(completion, list):
            completion = completion[-1].get("content", "") if completion else ""

        if isinstance(prompt, list):
            # Extract question text from chat messages
            question = ""
            for msg in prompt:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        question = content
                    elif isinstance(content, list):
                        question = " ".join(
                            p.get("text", "") for p in content if p.get("type") == "text"
                        )
        else:
            question = str(prompt)

        scores = scorer.score(question, gt, completion)
        rewards.append(scores_to_reward(scores))

    return rewards
