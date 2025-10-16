#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试PKL文件结构
检查index.pkl的实际数据结构
"""

import pickle
import json
from pathlib import Path

def debug_pkl_structure(pkl_file="clinical_memory_db/index.pkl"):
    """调试PKL文件结构"""
    pkl_path = Path(pkl_file)
    
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        return
    
    try:
        print(f"📥 读取 {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 成功加载数据")
        print(f"🔍 数据类型: {type(data)}")
        print(f"📏 数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        # 如果是元组，检查每个元素
        if isinstance(data, tuple):
            print(f"\n📦 元组内容分析:")
            for i, item in enumerate(data):
                print(f"  元素 {i}: {type(item)}")
                
                # 检查是否有docstore属性
                if hasattr(item, 'docstore'):
                    print(f"    ✅ 有docstore属性")
                    docstore = item.docstore
                    print(f"    docstore类型: {type(docstore)}")
                    
                    if hasattr(docstore, '_dict'):
                        docs = docstore._dict
                        print(f"    📄 文档数量: {len(docs)}")
                        
                        # 显示前几个文档
                        for j, (doc_id, doc) in enumerate(docs.items()):
                            if j >= 2:  # 只显示前2个
                                break
                            print(f"    文档 {j+1}: ID={doc_id}")
                            print(f"      类型: {type(doc)}")
                            
                            if hasattr(doc, 'page_content'):
                                content = doc.page_content
                                print(f"      内容长度: {len(content)}")
                                
                                # 尝试解析JSON
                                try:
                                    patient_data = json.loads(content)
                                    subject_id = patient_data.get('subject_id', 'N/A')
                                    print(f"      患者ID: {subject_id}")
                                except:
                                    print(f"      内容预览: {content[:100]}...")
                            
                            if hasattr(doc, 'metadata'):
                                print(f"      元数据: {doc.metadata}")
                    else:
                        print(f"    ❌ docstore没有_dict属性")
                        print(f"    docstore属性: {dir(docstore)}")
                else:
                    print(f"    ❌ 没有docstore属性")
                    print(f"    对象属性: {dir(item)[:10]}...")  # 只显示前10个属性
        
        # 如果是其他类型，显示属性
        elif hasattr(data, '__dict__'):
            print(f"\n📋 对象属性:")
            for attr in dir(data):
                if not attr.startswith('_'):
                    try:
                        value = getattr(data, attr)
                        print(f"  {attr}: {type(value)}")
                    except:
                        print(f"  {attr}: <无法访问>")
        
        return data
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_pkl_structure()