from typing import List, Dict, Union


def create_letter_options(items: List[str]) -> Dict[str, str]:
    """
    将列表元素转换为以字母(A, B, C...)为键的字典

    参数:
        items: 原始字符串列表（如诊断列表）

    返回:
        键为字母(A, B, C...)、值为列表元素的字典
    """
    # 生成与列表长度匹配的字母键（A对应65，依次递增）
    letters = [chr(65 + i) for i in range(len(items))]
    # 构造键值对字典
    return {letter: item for letter, item in zip(letters, items)}


def format_as_letter_options(items: List[str], separator: str = "\n") -> str:
    """
    将列表元素格式化为带字母前缀的字符串（如"A. 选项1\nB. 选项2"）

    参数:
        items: 原始字符串列表
        separator: 选项之间的分隔符（默认换行符"\n"）

    返回:
        格式化后的字符串
    """
    letter_dict = create_letter_options(items)
    # 拼接成 "字母. 内容" 的格式
    return separator.join([f"{letter}. {item}" for letter, item in letter_dict.items()])
