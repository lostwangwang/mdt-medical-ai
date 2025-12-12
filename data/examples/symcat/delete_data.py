import json
import os


def filter_age_zero_data(input_file, output_file=None):
    """
    读取JSON文件，删除Age等于0的记录

    参数:
    input_file: 输入文件路径
    output_file: 输出文件路径，如果为None则覆盖原文件
    """
    # 如果输出文件未指定，默认覆盖原文件
    if output_file is None:
        output_file = input_file

    try:
        # 读取文件
        with open(input_file, 'r', encoding='utf-8') as f:
            # 处理可能的多行JSON或单行JSON
            data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # 解析单行JSON
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError:
                        # 如果是整个文件一个JSON对象
                        f.seek(0)
                        data = json.load(f)
                        break

        # 过滤掉Age等于0的记录
        # 处理两种情况：单个字典或字典列表
        if isinstance(data, dict):
            filtered_data = data if data.get('Age', 1) != 0 else {}
        else:
            filtered_data = [record for record in data if record.get('Age', 1) != 0]

        # 写入过滤后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(filtered_data, list) and len(filtered_data) > 0:
                # 每行一个JSON对象（JSON Lines格式）
                for record in filtered_data:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
            elif isinstance(filtered_data, dict) and filtered_data:
                # 单个JSON对象
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"处理完成！")
        print(f"原始数据条数: {len(data) if isinstance(data, list) else 1}")
        print(
            f"过滤后数据条数: {len(filtered_data) if isinstance(filtered_data, list) else (1 if filtered_data else 0)}")

        return filtered_data

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return None
    except Exception as e:
        print(f"处理出错：{str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    input_file_path = "symcat_style_dataset.jsonl"

    # 方法1：覆盖原文件
    # filter_age_zero_data(input_file_path)

    # 方法2：保存到新文件
    output_file_path = "symcat_style_dataset_new.jsonl"
    filtered_data = filter_age_zero_data(input_file_path, output_file_path)

    # 打印过滤后的数据（可选）
    if filtered_data:
        print("\n过滤后的数据：")
        if isinstance(filtered_data, list):
            for record in filtered_data:
                print(json.dumps(record, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(filtered_data, ensure_ascii=False, indent=2))