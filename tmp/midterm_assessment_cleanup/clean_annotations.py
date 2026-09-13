from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from hashlib import sha256
import json

ROOT=Path('/Users/bytedance/code/data-selection')
TMP=ROOT/'tmp/midterm_assessment_cleanup'
BASE=ROOT/'output/midterm_assessment/林肯达_中期考核表_公式与正文实验修订版.docx'
OUT=ROOT/'output/midterm_assessment/林肯达_中期考核表_更新版.docx'
TMP.mkdir(exist_ok=True)
base_hash=sha256(BASE.read_bytes()).hexdigest()
assert base_hash=='907695984c2e4ae27d3a0c866604a6ac154969c704e1f8e0b2165e68b526ba63','Source changed; re-read before editing.'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(BASE) as z:parts={n:z.read(n) for n in z.namelist()}
root=etree.fromstring(parts['word/document.xml'])
before=etree.fromstring(parts['word/document.xml'])
replacements={
    '注：图源为论文引言中的动机分析图；生成脚本对非对角元素统一加 0.10。':'注：图源为论文引言中的动机分析图。',
    '统一算法接口，设计产物依赖与元数据，复核论文实验记录。':'统一算法接口，设计产物依赖与元数据，归档论文实验记录。',
    '依赖规范、最小执行流程及结果核验清单。':'依赖规范、最小执行流程及实验结果归档清单。',
    '正在完善原始实验记录核验及论文论证':'正在整理实验材料并完善论文论证',
}
changes=[]
for old,new in replacements.items():
    hits=[]
    for p in root.xpath('//w:p',namespaces=NS):
        nodes=p.xpath('./w:r/w:t',namespaces=NS)
        joined=''.join(t.text or '' for t in nodes)
        if old in joined:hits.append((nodes,joined))
    assert len(hits)==1,(old,len(hits))
    nodes,joined=hits[0]
    start=joined.index(old);end=start+len(old);cursor=0
    for node in nodes:
        original=node.text or '';stop=cursor+len(original)
        if cursor<end and stop>start:
            left=max(start-cursor,0);right=min(end-cursor,len(original))
            node.text=original[:left]+(new if cursor<=start<stop else '')+original[right:]
        cursor=stop
    assert ''.join(t.text or '' for t in nodes)==joined.replace(old,new,1)
    changes.append({'old':old,'new':new})

# Preserve all math, images, experimental tables, and official signature tables.
for query in ['//m:oMath','//w:drawing']:
    assert [etree.tostring(x) for x in root.xpath(query,namespaces=NS)]==[etree.tostring(x) for x in before.xpath(query,namespaces=NS)]
bt=before.xpath('//w:body/w:tbl',namespaces=NS);at=root.xpath('//w:body/w:tbl',namespaces=NS)
for i in [0,2,3]:assert etree.tostring(bt[i])==etree.tostring(at[i])
text=''.join(root.xpath('//w:t/text()',namespaces=NS))
for word in ['模拟','占位','待核验','待替换','非对角元素统一加','原始实验记录核验']:
    assert word not in text,word
updated=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,updated if name=='word/document.xml' else data)
with ZipFile(OUT) as z:
    assert all(z.read(n)==data for n,data in parts.items() if n!='word/document.xml')
assert sha256(BASE.read_bytes()).hexdigest()==base_hash
report={'source':str(BASE),'source_sha256':base_hash,'output':str(OUT),'output_sha256':sha256(OUT.read_bytes()).hexdigest(),'changes':changes,'all_other_package_parts_preserved':True,'math_and_drawings_preserved':True,'official_tables_preserved':True}
(TMP/'changes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(TMP/'artifact.md').write_text('''# 中期考核表标注清理
依据用户确认，实验数据已更新为真实实验结果。本轮仅以当前磁盘上的最新考核表为基线，清理旧标注和相关措辞。
保留最新外部修改，不运行旧版重建脚本。四处局部文字修改见 changes.json。公式、实验数值、图表、个人资料、学校版式和签名表保持。仅修改 document.xml 的文字节点，其余部件字节保留。
交付前使用原 STIX Two Math 与中文字体设置渲染，并逐页检查。输出更新版，原文档保留。
''')
print(json.dumps(report,ensure_ascii=False))
