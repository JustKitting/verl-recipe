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

_patches_applied = False


def patch_sglang_template_manager():
    try:
        import sglang.srt.managers.template_manager as tm

        original_init = tm.TemplateManager.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self, 'has_reasoning'):
                template = getattr(self, 'chat_template', None)
                if template is not None and not isinstance(template, str):
                    self.has_reasoning = False

        tm.TemplateManager.__init__ = patched_init
        print("[patches] Applied SGLang template_manager patch")
    except Exception as e:
        print(f"[patches] Could not patch SGLang template_manager: {e}")


def apply_all_patches():
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_sglang_template_manager()
