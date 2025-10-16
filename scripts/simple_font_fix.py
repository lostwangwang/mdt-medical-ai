#!/usr/bin/env python3
"""
简单字体修复脚本
使用matplotlib内置方法解决中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from pathlib import Path

def force_chinese_font():
    """强制设置中文字体"""
    print("🔧 强制设置中文字体...")
    
    # 方法1: 使用matplotlib的字体查找
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name.lower()
        if any(keyword in font_name for keyword in ['hei', 'kai', 'song', 'yuan', 'noto', 'wenquanyi', 'source']):
            chinese_fonts.append(font.name)
    
    if chinese_fonts:
        font_to_use = chinese_fonts[0]
        print(f"✅ 找到中文字体: {font_to_use}")
    else:
        # 方法2: 使用系统默认字体
        font_to_use = 'sans-serif'
        print(f"⚠️  使用默认字体: {font_to_use}")
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = [font_to_use, 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 方法3: 如果还是不行，使用unicode转义
    plt.rcParams['font.family'] = 'sans-serif'
    
    return font_to_use

def create_unicode_labels():
    """创建unicode标签（备用方案）"""
    treatments = [
        '\u624b\u672f',  # 手术
        '\u5316\u7597',  # 化疗  
        '\u653e\u7597',  # 放疗
        '\u514d\u75ab\u6cbb\u7597',  # 免疫治疗
        '\u59d1\u606f\u6cbb\u7597',  # 姑息治疗
        '\u89c2\u5bdf\u7b49\u5f85'   # 观察等待
    ]
    
    roles = [
        '\u80bf\u7624\u79d1\u533b\u751f',  # 肿瘤科医生
        '\u62a4\u58eb',  # 护士
        '\u5fc3\u7406\u5e08',  # 心理师
        '\u653e\u5c04\u79d1\u533b\u751f'   # 放射科医生
    ]
    
    return treatments, roles

def create_enhanced_plot():
    """创建增强的测试图表"""
    print("🎨 创建增强测试图表...")
    
    # 设置字体
    current_font = force_chinese_font()
    
    # 创建数据
    treatments_cn = ['手术', '化疗', '放疗', '免疫治疗', '姑息治疗', '观察等待']
    roles_cn = ['肿瘤科医生', '护士', '心理师', '放射科医生']
    
    # 备用unicode标签
    treatments_unicode, roles_unicode = create_unicode_labels()
    
    np.random.seed(42)
    data = np.random.uniform(-1, 1, (len(treatments_cn), len(roles_cn)))
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图表1: 使用中文标签
    sns.heatmap(
        data, 
        xticklabels=roles_cn,
        yticklabels=treatments_cn,
        annot=True, 
        cmap='RdYlGn', 
        center=0, 
        fmt='.2f',
        ax=ax1,
        cbar_kws={'label': '支持度'}
    )
    ax1.set_title('方案1: 直接中文标签', fontsize=14, fontweight='bold')
    ax1.set_xlabel('医疗团队角色', fontsize=12)
    ax1.set_ylabel('治疗方案', fontsize=12)
    
    # 图表2: 使用unicode标签
    sns.heatmap(
        data, 
        xticklabels=roles_unicode,
        yticklabels=treatments_unicode,
        annot=True, 
        cmap='RdYlGn', 
        center=0, 
        fmt='.2f',
        ax=ax2,
        cbar_kws={'label': '\u652f\u6301\u5ea6'}  # 支持度
    )
    ax2.set_title('\u65b9\u68481: Unicode\u6807\u7b7e', fontsize=14, fontweight='bold')  # 方案2: Unicode标签
    ax2.set_xlabel('\u533b\u7597\u56e2\u961f\u89d2\u8272', fontsize=12)  # 医疗团队角色
    ax2.set_ylabel('\u6cbb\u7597\u65b9\u6848', fontsize=12)  # 治疗方案
    
    plt.tight_layout()
    
    # 保存图片
    output_path = "enhanced_chinese_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 增强测试图表已保存: {output_path}")
    
    plt.show()
    
    return output_path

def create_simple_plot():
    """创建简单的英文+中文对照图表"""
    print("📊 创建英文+中文对照图表...")
    
    # 英文标签
    treatments_en = ['Surgery', 'Chemotherapy', 'Radiotherapy', 'Immunotherapy', 'Palliative', 'Watchful']
    roles_en = ['Oncologist', 'Nurse', 'Psychologist', 'Radiologist']
    
    # 中文标签（作为注释）
    treatments_cn = ['手术', '化疗', '放疗', '免疫治疗', '姑息治疗', '观察等待']
    roles_cn = ['肿瘤科医生', '护士', '心理师', '放射科医生']
    
    # 组合标签
    treatments_combined = [f"{en}\n({cn})" for en, cn in zip(treatments_en, treatments_cn)]
    roles_combined = [f"{en}\n({cn})" for en, cn in zip(roles_en, roles_cn)]
    
    np.random.seed(42)
    data = np.random.uniform(-1, 1, (len(treatments_en), len(roles_en)))
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        data, 
        xticklabels=roles_combined,
        yticklabels=treatments_combined,
        annot=True, 
        cmap='RdYlGn', 
        center=0, 
        fmt='.2f',
        cbar_kws={'label': 'Support Level (-1: Strongly Against, +1: Strongly Support)'}
    )
    
    plt.title('Medical Team Treatment Consensus Matrix\n医疗团队治疗方案共识矩阵', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Medical Team Roles / 医疗团队角色', fontsize=12)
    plt.ylabel('Treatment Options / 治疗方案', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    output_path = "bilingual_consensus_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 双语对照图表已保存: {output_path}")
    
    plt.show()
    
    return output_path

def main():
    """主函数"""
    print("🔧 简单字体修复工具")
    print("=" * 50)
    
    # 显示当前字体信息
    print(f"📋 当前matplotlib配置:")
    print(f"   font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"   font.family: {plt.rcParams['font.family']}")
    
    # 1. 创建增强图表
    enhanced_plot = create_enhanced_plot()
    
    # 2. 创建双语图表
    bilingual_plot = create_simple_plot()
    
    print(f"\n✅ 测试完成!")
    print(f"🖼️  生成的图片:")
    print(f"   - {enhanced_plot}")
    print(f"   - {bilingual_plot}")
    
    print(f"\n💡 如果中文仍显示为方框，建议:")
    print(f"   1. 使用双语版本图表")
    print(f"   2. 安装中文字体: sudo apt-get install fonts-wqy-microhei")
    print(f"   3. 重启Jupyter/Python环境")

if __name__ == "__main__":
    main()