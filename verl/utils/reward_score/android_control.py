# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Modified for the CGL release: safe literal parsing and strict action matching.
import ast
import math
import re
from difflib import SequenceMatcher


def similarity(x0, y0, x1, y1):
    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if d > 0.1:
        return 0.0
    elif d == 0:
        return 1.0
    else:
        return math.cos((math.pi / 2) * (d / 0.1))


def parse_action(action_str):
    """解析action字符串，返回(action_type, parameters)"""
    action_str = action_str.strip()

    no_param_actions = ["Back()", "Home()", "Wait()", "Finish()"]
    if action_str in no_param_actions:
        return action_str[:-2], []

    scroll_match = re.fullmatch(r"Scroll\((.*?)\)", action_str)
    if scroll_match:
        dir_str = scroll_match.group(1).strip()
        return "Scroll", [dir_str]

    coord_match = re.fullmatch(r"(Click|Long_press)\(([^,]+?),([^,]+?)\)", action_str)
    if coord_match:
        action_type = coord_match.group(1)
        try:
            x = float(coord_match.group(2).strip())
            y = float(coord_match.group(3).strip())
            return action_type, [x, y]
        except ValueError:
            return None

    str_param_match = re.fullmatch(r"(Open|Input_text)\((.*)\)", action_str)
    if str_param_match:
        action_type = str_param_match.group(1)
        param_str = str_param_match.group(2).strip()
        if (param_str.startswith('"') and param_str.endswith('"')) or (
            param_str.startswith("'") and param_str.endswith("'")
        ):
            param_str = param_str[1:-1]
        return action_type, [param_str]

    return None


def score_action(predict_str: str, ground_truth: str, extra_info) -> float:

    content = predict_str
    typee = extra_info["action_type"]
    arg = ground_truth
    type_matches = re.findall(r"<action>(.*?)</action>", content)
    if len(type_matches) == 0:
        reward = 0
        return reward
    else:
        pred_action = type_matches[0].strip()
        pred_parsed = parse_action(pred_action)
        if pred_parsed is None:
            reward = 0
            return reward

        pred_type, pred_params = pred_parsed
        if pred_type != typee:
            reward = 0
            return reward
        else:
            reward = 1
            if pred_type in ["Click", "Long_press"]:
                if len(pred_params) != 2:
                    return reward
                x, y = pred_params
                try:
                    arg = ast.literal_eval(arg) if isinstance(arg, str) else arg
                except (ValueError, SyntaxError, TypeError):
                    return 0.0
                if not isinstance(arg, (list, tuple)) or len(arg) not in (2, 4):
                    return 0.0
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in arg):
                    return 0.0
                if not (math.isfinite(x) and math.isfinite(y)):
                    return 0.0

                if len(arg) == 4:
                    if x > arg[0] and x < arg[2] and y > arg[1] and y < arg[3]:
                        reward += 1
                    else:
                        reward += 0
                elif len(arg) == 2:
                    dis = similarity(x, y, arg[0], arg[1])
                    reward += dis
            elif pred_type in ["Scroll", "Open"]:
                if pred_params[0] == arg:
                    reward += 1
            elif pred_type in ["Input_text"]:
                if SequenceMatcher(None, pred_params[0], arg).ratio() > 0.5:
                    reward += 1
            elif pred_type in ["Back", "Home", "Wait", "Finish"]:
                reward = 2
            return reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Adapter for verl's custom reward function calling convention."""
    return score_action(solution_str, ground_truth, extra_info)
