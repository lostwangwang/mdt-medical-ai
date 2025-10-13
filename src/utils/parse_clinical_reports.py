import re
import json
from pathlib import Path
from datetime import datetime

LAB_PATTERN = re.compile(r'\*\*(?P<key>[^:]+):\s*(?P<val>[^\*]+)\*\*')
META_PATTERNS = {
    'patient_id': re.compile(r'患者ID[:：]\s*(\d+)'),
    'encounter_id': re.compile(r'住院ID[:：]\s*(\d+)'),
    'sex_age': re.compile(r'性别[:：]\s*([FM男女])\s*\|\s*年龄[:：]\s*(\d+)'),
    'admit_discharge': re.compile(r'入院[:：]\s*([^|]+)\s*\|\s*出院[:：]\s*([^\n]+)')
}
DATE_BLOCK_RE = re.compile(r'📅\s*(\d{4}-\d{2}-\d{2})')

def parse_meta(text):
    meta = {}
    m = META_PATTERNS['patient_id'].search(text)
    if m: meta['patient_id'] = m.group(1)
    m = META_PATTERNS['encounter_id'].search(text)
    if m: meta['encounter_id'] = m.group(1)
    m = META_PATTERNS['sex_age'].search(text)
    if m:
        sex = m.group(1)
        if sex == 'M': sex = 'M'
        meta['sex'] = 'F' if sex in ('F','女') else 'M'
        meta['age'] = int(m.group(2))
    m = META_PATTERNS['admit_discharge'].search(text)
    if m:
        try:
            meta['admission_date'] = m.group(1).strip()
            meta['discharge_date'] = m.group(2).strip()
        except:
            pass
    return meta

def parse_lab_pairs(block_text):
    labs = {}
    for m in LAB_PATTERN.finditer(block_text):
        key = m.group('key').strip()
        raw = m.group('val').strip()
        # 尝试分离数值与单位与注释
        num_match = re.search(r'([+-]?\d+(\.\d+)?)', raw)
        unit_match = re.search(r'([a-zA-Z%/]+(?:\s*\S+)?)', raw)
        value = float(num_match.group(1)) if num_match else None
        unit = None
        # 简单提取单位（若在 raw 中）
        unit_parts = re.findall(r'[a-zA-Z%/]+', raw)
        if unit_parts:
            unit = ' '.join(unit_parts[:2])
        labs[key] = {'raw': raw, 'value': value, 'unit': unit}
    return labs

def split_date_blocks(text):
    parts = DATE_BLOCK_RE.split(text)
    # split returns [prefix, date1, block1, date2, block2,...]
    prefix = parts[0]
    blocks = []
    for i in range(1, len(parts), 2):
        date = parts[i]
        block = parts[i+1]
        blocks.append({'date': date, 'text': block.strip()})
    return prefix, blocks

def parse_report_file(path: str):
    text = Path(path).read_text(encoding='utf-8')
    meta = parse_meta(text)
    prefix, blocks = split_date_blocks(text)
    reports = []
    for b in blocks:
        labs = parse_lab_pairs(b['text'])
        # 抓诊断摘要（简单 heuristics）
        diag_match = re.search(r'核心诊断摘要[:：]\s*([^\n]+)', b['text'])
        diagnoses = []
        if diag_match:
            diagnoses = [d.strip() for d in re.split(r'[，,;；]', diag_match.group(1)) if d.strip()]
        # 若块内提到 vitals_json 为空，则设为空标志
        vitals_missing = 'vitals_json为空' in b['text'] or '无有效记录' in b['text']
        reports.append({
            'date': b['date'],
            'diagnoses': diagnoses,
            'labs': labs,
            'vitals_missing': vitals_missing,
            'raw_text': b['text'][:1000]
        })
    out = {'meta': meta, 'reports': reports, 'source': str(path)}
    return out

def save_parsed(out_obj, out_path):
    Path(out_path).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    src = Path(__file__).parents[2] / 'data' / 'patient_daily_condition_updated.txt'
    parsed = parse_report_file(str(src))
    dest = Path(__file__).parents[2] / 'data' / 'processed' / f"patient_{parsed['meta'].get('patient_id','unknown')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_parsed(parsed, str(dest))
    print("saved:", dest)