from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from hashlib import sha256
from lxml import etree as E
import json

BASE=Path('/Users/bytedance/code/data-selection')
SRC=BASE/'output/midterm_assessment/林肯达_中期考核表_润色优化版.docx'
OUT=BASE/'output/midterm_assessment/林肯达_中期考核表_精简修订版.docx'
TMP=BASE/'tmp/midterm_assessment_concise'
EXPECTED='01ada8a9a347fba301eeb9a6d299f8417caf41ca007173e0e954a6d3aaef530c'
assert sha256(SRC.read_bytes()).hexdigest()==EXPECTED
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def tag(x):return '{'+NS['w']+'}'+x
def text(x):return ''.join(x.xpath('.//w:t/text()',namespaces=NS))
with ZipFile(SRC) as z:
    entries=z.infolist();parts={i.filename:z.read(i.filename) for i in entries}
root=E.fromstring(parts['word/document.xml']);baseline=deepcopy(root)
heading=next(p for p in root.xpath('//w:body//w:p',namespaces=NS) if text(p)=='4.2 后续研究计划')
cell=heading.getparent();assert cell.tag==tag('tc')
idx=cell.index(heading)
old=list(cell)[idx+1:]
assert '后续八周' in ''.join(text(x) for x in old)
proto=deepcopy(old[0]);assert proto.tag==tag('p')
new_plan=[
'（1）完善 PCU-Select 的实验分析，进一步明确不同任务、选数预算和配置差异下的适用范围，整理方法与实验材料。',
'（2）基于现有算法开发多配置数据选择与微调实验管理系统，研究分层缓存与增量复用、公共计算去重与阶段恢复，支持多任务、多配置实验的统一执行。',
'（3）通过对比实验检验缓存复用结果的一致性，评估运行时间、存储占用和恢复开销，完成系统总结及学位论文撰写。',
]
for x in old:cell.remove(x)
for s in new_plan:
    p=E.Element(tag('p'))
    p.append(deepcopy(proto.find('w:pPr',NS)))
    run=E.SubElement(p,tag('r'))
    run.append(deepcopy(proto.find('w:r/w:rPr',NS)))
    t=E.SubElement(run,tag('t'));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=s
    cell.append(p)

label=next(p for p in root.xpath('//w:body//w:p',namespaces=NS) if text(p)=='与课题相关的学术成果情况：')
acell=label.getparent();assert acell.tag==tag('tc')
tail=list(acell)[acell.index(label)+1:]
assert len(tail)==1 and tail[0].tag==tag('p')
for item in list(tail[0]):
    if item.tag!=tag('pPr'):tail[0].remove(item)
assert text(acell)=='与课题相关的学术成果情况：'

def items(r,path):return [E.tostring(x) for x in r.xpath(path,namespaces=NS)]
assert items(root,'//m:oMath')==items(baseline,'//m:oMath')
assert items(root,'//w:drawing')==items(baseline,'//w:drawing')
assert items(root,'//w:sectPr')==items(baseline,'//w:sectPr')
bt=baseline.find('w:body',NS).findall('w:tbl',NS)
at=root.find('w:body',NS).findall('w:tbl',NS)
for i in [0,2,3]:assert E.tostring(bt[i])==E.tostring(at[i])
def numeric_tables(r):
    result=[]
    for t in r.xpath('//w:tbl',namespaces=NS):
        s=text(t);rows=len(t.findall('w:tr',NS))
        if ('Random' in s and '32.79' in s and rows==6) or ('Reuse-one-LESS' in s and rows==7):result.append(E.tostring(t))
    return result
assert numeric_tables(root)==numeric_tables(baseline) and len(numeric_tables(root))==2
# No other paragraphs before the plan heading have changed.
bp=baseline.xpath('//w:body//w:p',namespaces=NS);ap=root.xpath('//w:body//w:p',namespaces=NS)
stop=next(i for i,p in enumerate(bp) if text(p)=='4.2 后续研究计划')
assert [E.tostring(p) for p in bp[:stop+1]]==[E.tostring(p) for p in ap[:stop+1]]
parts['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(OUT,'w') as z:
    for i in entries:z.writestr(i,parts[i.filename])
with ZipFile(SRC) as a,ZipFile(OUT) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
assert changed==['word/document.xml']
assert sha256(SRC.read_bytes()).hexdigest()==EXPECTED
qa={'output':str(OUT),'source_sha256':EXPECTED,'output_sha256':sha256(OUT.read_bytes()).hexdigest(),
    'changed_parts':changed,'new_plan':new_plan,'academic_achievements_blank':True,
    'prior_content_formulas_figures_numeric_tables_official_forms_preserved':True}
(TMP/'changes.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps(qa,ensure_ascii=False,indent=2))
