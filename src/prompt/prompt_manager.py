import os
import yaml
from typing import Dict, Any


class PromptManager:
    """
    PromptManager：统一管理不同数据集的多模板 Prompt。
    支持：
      - 每个数据集一个 YAML；
      - 一个 YAML 文件中有多个模板（如 build_treatment_reasoning_prompt / build_diagnosis_prompt）；
      - 动态选择模板构建完整 prompt。
    """

    def __init__(self, prompt_dir: str):
        self.prompt_dir = prompt_dir
        self.prompts: Dict[str, Any] = {}
        self._load_all_prompts()

    # =====================================
    # 📦 1. 加载所有 YAML 模板文件
    # =====================================
    def _load_all_prompts(self):
        """加载指定目录下的所有 .yaml 文件"""
        for filename in os.listdir(self.prompt_dir):
            if not filename.endswith((".yaml", ".yml")):
                continue

            path = os.path.join(self.prompt_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            dataset_name = data.get("dataset") or filename.replace(".yaml", "")
            self.prompts[dataset_name] = data

    # =====================================
    # 🧩 2. 构建指定模板的完整 Prompt
    # =====================================
    def build_prompt_for_question(
        self,
        dataset_name: str,
        template_name: str,
        role: Any,
        role_descriptions: Dict[Any, str],
        question_state: Any,
    ) -> Dict[str, str]:
        """
        根据 dataset_name + template_name 构造完整 Prompt。
        模板来源：YAML 文件中对应的模板配置。
        """

        # === 1️⃣ 获取数据集模板 ===
        dataset_prompts = self.prompts.get(dataset_name)
        if not dataset_prompts:
            raise ValueError(f"❌ 未找到数据集 '{dataset_name}' 的模板配置")

        # === 2️⃣ 获取模板节点 ===
        template = dataset_prompts.get(template_name)
        if not template:
            raise ValueError(
                f"❌ 模板 '{template_name}' 不存在于 '{dataset_name}' 的配置中。\n"
                f"可选模板包括: {', '.join([k for k in dataset_prompts.keys() if k.startswith('build_')])}"
            )

        system_text = template.get("system", "")
        prompt_text = template.get("prompt", "")

        # === 3️⃣ 构造替换内容 ===
        meta_info = getattr(question_state, "meta_info", "") or "无特殊背景"
        question = getattr(question_state, "question", "")
        options = getattr(question_state, "options", {})
        options_list = "\n".join([f"{k}: {v}" for k, v in options.items()])
        role_value = getattr(role, "value", str(role))

        # === 4️⃣ 替换模板占位符 ===
        try:
            filled_prompt = prompt_text.format(
                role_value=role_value,
                question=question,
                meta_info=meta_info,
                options_list=options_list,
            )
        except KeyError as e:
            raise KeyError(f"⚠️ 模板缺少占位符: {e}")

        # === 5️⃣ 返回完整结构 ===
        return {"system": system_text, "prompt": filled_prompt}

    # =====================================
    # 🧾 3. 查看当前加载的模板结构
    # =====================================
    def list_templates(self, dataset_name: str):
        """列出某个数据集下所有可用模板"""
        if dataset_name not in self.prompts:
            raise ValueError(f"❌ 数据集 '{dataset_name}' 未加载")
        dataset = self.prompts[dataset_name]
        return [k for k in dataset.keys() if k.startswith("build_")]
