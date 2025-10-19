#!/usr/bin/env python3
"""
MDT医疗智能体系统 - 数据库初始化脚本
文件路径: scripts/init_database.py
作者: 系统维护
功能: 初始化SQLite数据库和基本表结构
"""

import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_database_schema(db_path: str):
    """创建数据库表结构"""
    
    # 确保目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 连接数据库（如果不存在会自动创建）
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 创建系统配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建患者信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                age INTEGER,
                diagnosis TEXT,
                stage TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建MDT会诊记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mdt_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                patient_id TEXT NOT NULL,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                consensus_score REAL,
                final_recommendation TEXT,
                participants TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # 创建对话记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dialogue_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                round_number INTEGER,
                role_type TEXT,
                message_content TEXT,
                confidence_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES mdt_sessions (session_id)
            )
        ''')
        
        # 创建系统健康监控表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_name TEXT NOT NULL,
                component_type TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                response_time_ms REAL,
                metadata TEXT,
                check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建性能指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                component TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 插入初始配置数据
        initial_configs = [
            ('system_version', '1.0.0', 'MDT系统版本'),
            ('database_version', '1.0.0', '数据库版本'),
            ('initialized_at', datetime.now().isoformat(), '数据库初始化时间'),
            ('environment', 'development', '运行环境'),
            ('debug_mode', 'true', '调试模式')
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO system_config (config_key, config_value, description)
            VALUES (?, ?, ?)
        ''', initial_configs)
        
        # 提交事务
        conn.commit()
        
        print(f"✅ 数据库初始化成功: {db_path}")
        print("📋 创建的表:")
        print("  - system_config (系统配置)")
        print("  - patients (患者信息)")
        print("  - mdt_sessions (MDT会诊记录)")
        print("  - dialogue_records (对话记录)")
        print("  - health_monitoring (健康监控)")
        print("  - performance_metrics (性能指标)")
        
        # 验证表创建
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📊 数据库包含 {len(tables)} 个表")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()


def verify_database(db_path: str):
    """验证数据库连接和基本功能"""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # 测试基本查询
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        if result and result[0] == 1:
            print("✅ 数据库连接测试成功")
            
            # 获取表数量
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"📊 数据库包含 {table_count} 个表")
            
            # 获取数据库大小
            db_size = os.path.getsize(db_path) / 1024  # KB
            print(f"💾 数据库大小: {db_size:.2f} KB")
            
            return True
        else:
            print("❌ 数据库查询测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据库验证失败: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()


def main():
    """主函数"""
    print("🏥 MDT医疗智能体系统 - 数据库初始化")
    print("=" * 50)
    
    # 数据库文件路径
    db_path = "data/mdt_system.db"
    
    # 检查是否已存在数据库
    if os.path.exists(db_path):
        response = input(f"⚠️  数据库文件已存在: {db_path}\n是否要重新初始化? (y/N): ")
        if response.lower() != 'y':
            print("🚫 取消初始化")
            return
        else:
            # 备份现有数据库
            backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(db_path, backup_path)
            print(f"📦 已备份现有数据库到: {backup_path}")
    
    # 创建数据库
    print(f"🔧 正在初始化数据库: {db_path}")
    
    if create_database_schema(db_path):
        print("\n🔍 验证数据库...")
        if verify_database(db_path):
            print("\n🎉 数据库初始化完成!")
            print(f"📍 数据库位置: {os.path.abspath(db_path)}")
        else:
            print("\n❌ 数据库验证失败")
            sys.exit(1)
    else:
        print("\n❌ 数据库初始化失败")
        sys.exit(1)


if __name__ == "__main__":
    main()