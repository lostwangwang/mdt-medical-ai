#!/usr/bin/env python3
"""
中文字体修复脚本
下载并配置中文字体以解决matplotlib显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
import zipfile
from pathlib import Path
import platform
import subprocess
import sys

def download_chinese_font():
    """下载中文字体文件"""
    print("📥 正在下载中文字体...")
    
    # 创建字体目录
    font_dir = Path.home() / ".matplotlib" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载思源黑体（开源中文字体）
    font_url = "https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip"
    font_zip_path = font_dir / "SourceHanSansSC.zip"
    font_ttf_path = font_dir / "SourceHanSansSC-Regular.otf"
    
    try:
        if not font_ttf_path.exists():
            print(f"   正在从 {font_url} 下载字体...")
            urllib.request.urlretrieve(font_url, font_zip_path)
            
            # 解压字体文件
            with zipfile.ZipFile(font_zip_path, 'r') as zip_ref:
                # 只提取Regular字体
                for file in zip_ref.namelist():
                    if "Regular.otf" in file and "SC" in file:
                        zip_ref.extract(file, font_dir)
                        # 重命名为简单名称
                        extracted_path = font_dir / file
                        extracted_path.rename(font_ttf_path)
                        break
            
            # 清理zip文件
            font_zip_path.unlink()
            print(f"✅ 字体下载完成: {font_ttf_path}")
        else:
            print(f"✅ 字体已存在: {font_ttf_path}")
            
        return str(font_ttf_path)
        
    except Exception as e:
        print(f"❌ 字体下载失败: {e}")
        return None

def install_system_fonts():
    """尝试安装系统中文字体包"""
    system = platform.system()
    
    if system == "Linux":
        try:
            print("🔧 尝试安装系统中文字体包...")
            
            # 检查包管理器
            if os.system("which apt-get > /dev/null 2>&1") == 0:
                # Ubuntu/Debian
                commands = [
                    "sudo apt-get update",
                    "sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk"
                ]
            elif os.system("which yum > /dev/null 2>&1") == 0:
                # CentOS/RHEL
                commands = [
                    "sudo yum install -y wqy-microhei-fonts wqy-zenhei-fonts google-noto-cjk-fonts"
                ]
            else:
                print("⚠️  未识别的包管理器，跳过系统字体安装")
                return False
            
            for cmd in commands:
                print(f"   执行: {cmd}")
                result = os.system(cmd)
                if result != 0:
                    print(f"   命令执行失败: {cmd}")
                    
            print("✅ 系统字体安装完成")
            return True
            
        except Exception as e:
            print(f"❌ 系统字体安装失败: {e}")
            return False
    else:
        print(f"⚠️  非Linux系统，跳过系统字体安装")
        return False

def setup_matplotlib_font(font_path=None):
    """配置matplotlib字体"""
    print("⚙️  配置matplotlib字体...")
    
    # 清除matplotlib字体缓存
    try:
        cache_dir = Path(fm.get_cachedir())
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("🗑️  已清除matplotlib字体缓存")
    except Exception as e:
        print(f"⚠️  清除缓存失败: {e}")
    
    # 重新构建字体缓存
    fm._rebuild()
    
    # 尝试设置字体
    fonts_to_try = []
    
    # 如果有下载的字体，优先使用
    if font_path and os.path.exists(font_path):
        fonts_to_try.append(font_path)
    
    # 添加系统字体
    system_fonts = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans'
    ]
    fonts_to_try.extend(system_fonts)
    
    for font in fonts_to_try:
        try:
            if font.endswith('.otf') or font.endswith('.ttf'):
                # 字体文件路径
                prop = fm.FontProperties(fname=font)
                plt.rcParams['font.sans-serif'] = [prop.get_name()]
            else:
                # 字体名称
                plt.rcParams['font.sans-serif'] = [font]
            
            plt.rcParams['axes.unicode_minus'] = False
            
            # 测试字体是否工作
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=12, ha='center', va='center')
            plt.close(fig)
            
            print(f"✅ 成功设置字体: {font}")
            return font
            
        except Exception as e:
            print(f"❌ 字体设置失败 {font}: {e}")
            continue
    
    print("❌ 所有字体设置都失败了")
    return None

def create_test_plot():
    """创建测试图表"""
    print("🧪 创建测试图表...")
    
    import numpy as np
    import seaborn as sns
    
    # 创建测试数据
    treatments = ['手术', '化疗', '放疗', '免疫治疗', '姑息治疗', '观察等待']
    roles = ['肿瘤科医生', '护士', '心理师', '放射科医生']
    
    np.random.seed(42)
    data = np.random.uniform(-1, 1, (len(treatments), len(roles)))
    
    # 创建热力图
    plt.figure(figsize=(10, 6))
    
    sns.heatmap(
        data, 
        xticklabels=roles,
        yticklabels=treatments,
        annot=True, 
        cmap='RdYlGn', 
        center=0, 
        fmt='.2f',
        cbar_kws={'label': '支持度 (-1: 强烈反对, +1: 强烈支持)'}
    )
    
    plt.title('医疗团队治疗方案共识矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('医疗团队角色', fontsize=12)
    plt.ylabel('治疗方案', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    output_path = "fixed_chinese_font_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 测试图表已保存: {output_path}")
    
    plt.show()
    
    return output_path

def list_available_fonts():
    """列出可用字体"""
    print("\n📋 系统可用字体:")
    
    fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_keywords = ['han', 'hei', 'kai', 'song', 'yuan', 'micro', 'noto', 'source', 'wenquanyi', 'sim']
    
    chinese_fonts = []
    for font in fonts:
        if any(keyword in font.lower() for keyword in chinese_keywords):
            chinese_fonts.append(font)
    
    if chinese_fonts:
        print("🈶 可能支持中文的字体:")
        for font in sorted(set(chinese_fonts))[:15]:
            print(f"   - {font}")
    else:
        print("❌ 未找到明确支持中文的字体")
    
    return chinese_fonts

def main():
    """主函数"""
    print("🔧 中文字体修复工具")
    print("=" * 50)
    
    # 1. 列出当前可用字体
    available_fonts = list_available_fonts()
    
    # 2. 尝试安装系统字体
    install_system_fonts()
    
    # 3. 下载字体文件
    downloaded_font = download_chinese_font()
    
    # 4. 配置matplotlib字体
    current_font = setup_matplotlib_font(downloaded_font)
    
    # 5. 创建测试图表
    if current_font:
        test_image = create_test_plot()
        
        print(f"\n✅ 字体修复完成!")
        print(f"📊 当前使用字体: {current_font}")
        print(f"🖼️  测试图片: {test_image}")
        
        # 显示当前配置
        print(f"\n⚙️  matplotlib配置:")
        print(f"   font.sans-serif: {plt.rcParams['font.sans-serif']}")
        print(f"   axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
        
    else:
        print(f"\n❌ 字体修复失败!")
        print(f"💡 建议手动操作:")
        print(f"   1. sudo apt-get install fonts-wqy-microhei")
        print(f"   2. fc-cache -fv")
        print(f"   3. 重启Python环境")

if __name__ == "__main__":
    main()