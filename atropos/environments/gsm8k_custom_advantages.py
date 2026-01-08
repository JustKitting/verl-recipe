# Copyright 2025 Nous Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from typing import List, Optional, Union

from .gsm8k import GSM8kEnv, ScoredDataGroup

logger = logging.getLogger(__name__)


class GSM8KCustomAdvantagesEnvironment(GSM8kEnv):
    """Boosts tokens matching answer formatting patterns."""
    BOOST_PATTERNS = [r"\\boxed\{", r"\\boxed", r"\}", r"####", r"The answer is", r"Answer:"]
    ADVANTAGE_BOOST = 0.1
    BASE_ADVANTAGE = 0.0

    async def score(self, rollout_group_data) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = await super().score(rollout_group_data)
        if scores is None:
            return None

        scores["advantages"] = [
            self._compute_token_advantages(tokens, masks)
            for tokens, masks in zip(scores["tokens"], scores["masks"])
        ]
        return scores

    def _compute_token_advantages(self, tokens: List[int], masks: List[int]) -> List[float]:
        advantages = []
        try:
            decoded = self.tokenizer.decode(tokens)
        except Exception:
            return [self.BASE_ADVANTAGE if m != -100 else 0.0 for m in masks]

        boost_positions = set()
        for pattern in self.BOOST_PATTERNS:
            for match in re.finditer(pattern, decoded, re.IGNORECASE):
                boost_positions.update(range(match.start(), match.end()))

        char_pos = 0
        for token_id, mask in zip(tokens, masks):
            if mask == -100:
                advantages.append(0.0)
            else:
                try:
                    token_len = len(self.tokenizer.decode([token_id]))
                except Exception:
                    token_len = 1

                should_boost = any(pos in boost_positions for pos in range(char_pos, char_pos + token_len))
                advantages.append(self.BASE_ADVANTAGE + self.ADVANTAGE_BOOST if should_boost else self.BASE_ADVANTAGE)
                char_pos += token_len

        return advantages


Environment = GSM8KCustomAdvantagesEnvironment

if __name__ == "__main__":
    GSM8KCustomAdvantagesEnvironment.cli()
