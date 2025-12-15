"""
This module contains utility functions for sanitizing and adapting LLM message histories to comply with the strict
conversation template requirements of certain models, such as Mistral. It ensures adherence to constraints like the
single, initial system message rule and the alternation of user/assistant roles by squashing consecutive messages.
"""

from typing import List, Dict, Any, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache
from litellm.types.utils import CallTypesLiteral


class MistralSanitizerHandler(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ):
        """
        Hook to sanitize messages for Mistral models before the API call.
        """
        # 1. Apply only to chat completion endpoints (streaming or standard)
        if call_type in {"completion", "acompletion"}:
            if "messages" in data and isinstance(data["messages"], list):
                # 2. Run the sanitization logic
                data["messages"] = self._fix_mistral_messages(data["messages"])

        return data

    def _fix_mistral_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enforces Mistral constraints:
        1. Max one system message at the start.
        2. Strictly alternating User/Assistant roles (squashing duplicates).
        """
        if not messages:
            return messages

        # --- Phase 1: Handle System Messages ---
        sanitized_prefix = []
        pending_messages = []

        # Check the very first message
        first_msg = messages[0]

        if first_msg["role"] == "system":
            # Start with the first system message
            current_system_msg = first_msg.copy()
            idx = 1

            # Squash subsequent system messages immediately following the first one
            while idx < len(messages) and messages[idx]["role"] == "system":
                current_system_msg["content"] = self._merge_contents(
                    current_system_msg.get("content"), messages[idx].get("content")
                )
                idx += 1

            sanitized_prefix.append(current_system_msg)
            # The rest of the messages to process
            pending_messages = messages[idx:]
        else:
            # First message is not system, process all
            pending_messages = messages[:]

        # Change any remaining 'system' roles in the body to 'user'
        for msg in pending_messages:
            if msg["role"] == "system":
                msg["role"] = "user"

        # --- Phase 2: Ensure Alternating Roles (Squashing) ---
        final_messages = sanitized_prefix

        for msg in pending_messages:
            # If final_messages is empty (no system msg), just add the first one
            if not final_messages:
                final_messages.append(msg)
                continue

            last_msg = final_messages[-1]

            # Logic to check if we should squash
            # We squash if roles match.
            # Note: We usually avoid squashing if tool_calls are involved to be safe,
            # but for strict text alternation errors, we must squash User-User or Assistant-Assistant.
            roles_match = msg["role"] == last_msg["role"]

            # Exception: Do not squash consecutive 'tool' messages (results from parallel calls)
            # as these are usually expected to be distinct items in the list.
            is_tool_result = msg["role"] == "tool"

            if roles_match and not is_tool_result:
                # Merge content into the last message
                last_msg["content"] = self._merge_contents(
                    last_msg.get("content"), msg.get("content")
                )

                # If the current message has tool_calls and the last didn't, we inherit them
                # (This handles: Assistant(Text) + Assistant(ToolCall) -> Assistant(Text+ToolCall))
                if "tool_calls" in msg and "tool_calls" not in last_msg:
                    last_msg["tool_calls"] = msg["tool_calls"]
                elif "tool_calls" in msg and "tool_calls" in last_msg:
                    # If both have tool calls, we extend the list
                    last_msg["tool_calls"].extend(msg["tool_calls"])
            else:
                final_messages.append(msg)

        return final_messages

    def _merge_contents(
        self, content_a: Union[str, list, None], content_b: Union[str, list, None]
    ) -> Union[str, list]:
        """
        Safely merges two content fields.
        Handles: str+str, list+list, mixed types, and None.
        """
        # Normalization: Treat None as empty string
        if content_a is None:
            content_a = ""
        if content_b is None:
            content_b = ""

        # Case 1: Both are strings
        if isinstance(content_a, str) and isinstance(content_b, str):
            if not content_a:
                return content_b
            if not content_b:
                return content_a
            return f"{content_a}\n\n{content_b}"

        # Case 2: At least one is a list (multimodal or structured content)
        # Helper to force content to list format
        def to_list(c):
            if isinstance(c, list):
                return c
            if isinstance(c, str):
                return [{"type": "text", "text": c}] if c else []
            return []  # Fallback for unknown types

        list_a = to_list(content_a)
        list_b = to_list(content_b)

        return list_a + list_b


# Instantiate the handler
proxy_handler_instance = MistralSanitizerHandler()
