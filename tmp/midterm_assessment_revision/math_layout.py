import re

def E(tag,*children,**attrs):
    el=OxmlElement('m:'+tag)
    for k,v in attrs.items():el.set(qn('m:'+k),str(v))
    for c in children:
        if isinstance(c,list):
            for cc in c:el.append(cc)
        else:el.append(c)
    return el

def R(text,normal=False):
    pr=E('rPr')
    if normal and any(c.isascii() and c.isalpha() for c in text):pr.append(E('nor'))
    pr.append(E('sty',val='p' if normal else 'i'))
    r=E('r',pr)
    wp=OxmlElement('w:rPr')
    f=OxmlElement('w:rFonts')
    for name in ['ascii','hAnsi','eastAsia','cs']:f.set(qn('w:'+name),'STIX Two Math')
    wp.append(f)
    for name in ['i','iCs']:
        item=OxmlElement('w:'+name);item.set(qn('w:val'),'0' if normal else '1');wp.append(item)
    for name in ['sz','szCs']:
        item=OxmlElement('w:'+name);item.set(qn('w:val'),'22');wp.append(item)
    r.append(wp)
    t=E('t');t.text=text;t.set(qn('xml:space'),'preserve');r.append(t)
    return r

def V(s):return R(s)
def N(s):
    if s in ['=','+','−','≈']:s='\u2009'+s+'\u2009'
    if s=='−0.2':s='\u2009−\u20090.2'
    return R(s,True)
def seq(x):return x if isinstance(x,list) else [x if not isinstance(x,str) else V(x)]
def sub(b,s):return E('sSub',E('e',*seq(b)),E('sub',*seq(s)))
def sup(b,s):return E('sSup',E('e',*seq(b)),E('sup',*seq(s)))
def ss(b,s,u):return E('sSubSup',E('e',*seq(b)),E('sub',*seq(s)),E('sup',*seq(u)))
def frac(n,d):return E('f',E('num',*seq(n)),E('den',*seq(d)))
def group(xs,left='(',right=')'):
    return E('d',E('dPr',E('begChr',val=left),E('endChr',val=right)),E('e',*seq(xs)))
def nsum(low,term):
    return E('nary',E('naryPr',E('chr',val='∑'),E('limLoc',val='subSup'),E('supHide',val='1')),E('sub',*seq(low)),E('sup'),E('e',*seq(term)))
def hat(x):return E('acc',E('accPr',E('chr',val='̂')),E('e',*seq(x)))
def tilde(x):return E('acc',E('accPr',E('chr',val='̃')),E('e',*seq(x)))
def fun(name,args):return E('func',E('fName',N(name)),E('e',group(args)))
def comma():return N(',')

def eqnodes(key):
    if key=='selection_objective':
        constraint=[V('S'),N('⊆'),V('D'),N(', '),group('S','|','|'),N('≤'),V('B')]
        arg=E('limLow',E('e',N('arg max')),E('lim',*constraint))
        value=fun('Perf',[fun('Tune',[sub('M',N('0')),comma(),V('p'),comma(),V('S')]),comma(),ss('V','t',N('test'))])
        return [sup('S',N('*')),N('='),E('func',E('fName',arg),E('e',value))]
    if key=='proxy_label':
        return [sup('u',N('lo')),group([V('x'),comma(),V('p'),comma(),V('t')]),N('='),nsum([V('ω'),N('∈'),V('Ω')],[ss(tilde('w'),'p','ω'),fun('cos',[ss('g','x','ω'),comma(),ss('g','t','ω')])])]
    if key=='short_update':
        theta=sub('θ','a');lv=sub('L',sub('V','t'))
        adapt=E('func',E('fName',ss(N('Adapt'),'p','h')),E('e',group([V('B'),group('x'),N(';'),deepcopy(theta)])))
        return [V('Δ'),group([V('x'),comma(),V('p'),comma(),V('t'),comma(),V('a'),comma(),V('h')]),N('='),deepcopy(lv),group(deepcopy(theta)),N('−'),deepcopy(lv),group(adapt)]
    if key=='conservative_score':
        args=group([V('x'),comma(),sup('p',N('*')),comma(),sup('t',N('*'))])
        return [V('q'),group('x'),N('='),sub(hat('μ'),'φ'),deepcopy(args),N('−0.2'),sub(hat('σ'),'φ'),deepcopy(args)]
    if key=='cluster_quota':
        def value(i):return [sup(group(ss('v',i,N('+'))),N('0.6')),sup(group(sub('C',i),'|','|'),N('0.4'))]
        return [sub('b','k'),N('='),E('func',E('fName',N('round')),E('e',group([V('B'),frac(value('k'),nsum('j',value('j')))],'[',']'))),N(',   '),ss('v','k',N('+')),N('='),fun('max',[sub('v','k'),comma(),N('0')])]
    if key=='cost_general':
        lines=[
            [sub('C',N('PCU')),group([V('P'),comma(),V('Q')]),N('='),sub('C',N('offline')),N('+'),V('PQ'),ss('C',N('PCU'),N('pair'))],
            [sub('C',N('LESS')),group([V('P'),comma(),V('Q')]),N('='),V('P'),ss('C',N('LESS'),N('data')),N('+'),V('PQ'),ss('C',N('LESS'),N('rank'))]]
        return [E('eqArr',E('eqArrPr',E('baseJc',val='center')), *[E('e',*line) for line in lines])]
    if key=='break_even':return [sup('P',N('*')),N('='),frac(N('144'),N('63.2−1.51')),N('≈2.33')]
    raise KeyError(key)

def hidden_borders(t):
    for b in list(t._tbl.tblPr.findall(qn('w:tblBorders'))):t._tbl.tblPr.remove(b)
    b=OxmlElement('w:tblBorders');t._tbl.tblPr.append(b)
    for name in ['top','left','bottom','right','insideH','insideV']:
        edge=OxmlElement('w:'+name);edge.set(qn('w:val'),'nil');b.append(edge)

eqcount=0
def equation(key):
    global eqcount
    eqcount+=1
    t=wc.add_table(rows=1,cols=3);t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for col,c,w in zip(t.columns,t.rows[0].cells,[0.75,13.5,0.75]):
        col.width=Cm(w);c.width=Cm(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    margins(t,60,0,60,0);hidden_borders(t)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    for c in t.rows[0].cells:
        p=setcell(c,'',indent=False,align=WD_ALIGN_PARAGRAPH.CENTER,after=0)
        p.paragraph_format.line_spacing=1.15;p.paragraph_format.keep_together=True
    p=t.cell(0,1).paragraphs[0]
    p._p.append(E('oMathPara',E('oMathParaPr',E('jc',val='center')),E('oMath',*eqnodes(key))))
    p=t.cell(0,2).paragraphs[0];p.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    font(p.add_run('('+str(eqcount)+')'),10.5)
    para(wc.paragraphs[-1],'',size=1,line=2,indent=False,after=2)
    return t

def heading2(c,text):return addp(c,text,bold=True,indent=False,before=7,after=4,line=17,keep=True)
def major(c,text):return addp(c,text,bold=True,indent=False,before=9,after=5,line=18,keep=True)

def fig(path,caption,width,cell=None):
    target=wc if cell is None else cell
    t=target.add_table(rows=1,cols=1);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    t.columns[0].width=Cm(15.0);t.cell(0,0).width=Cm(15.0)
    margins(t,30,0,30,0);hidden_borders(t)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    c=t.cell(0,0);p=setcell(c,'',indent=False,align=WD_ALIGN_PARAGRAPH.CENTER,after=3)
    p.paragraph_format.line_spacing=1.0;p.paragraph_format.keep_together=True;p.paragraph_format.keep_with_next=True
    pic=p.add_run().add_picture(str(path),width=Cm(width));pic._inline.docPr.set('descr',caption)
    para(c.add_paragraph(),caption,size=9.5,indent=False,align=WD_ALIGN_PARAGRAPH.CENTER,line=14,after=0)
    para(target.paragraphs[-1],'',size=1,line=2,indent=False,after=0)
    return t

PPR_ORDER='pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange'.split()
def normalize_body_math_and_properties():
    mp=doc.settings.element.find(qn('m:mathPr'))
    if mp is None:mp=E('mathPr');doc.settings.element.append(mp)
    mf=mp.find(qn('m:mathFont'))
    if mf is None:mf=E('mathFont');mp.insert(0,mf)
    mf.set(qn('m:val'),'STIX Two Math')
    names={s:i for i,s in enumerate(PPR_ORDER)}
    # Only the newly authored main form changes; cover and signatures retain exact XML.
    for pr in main._tbl.iter(qn('w:pPr')):
        children=list(pr)
        for e in children:pr.remove(e)
        for e in sorted(children,key=lambda e:names.get(etree.QName(e).localname,100)):pr.append(e)
    table_order='tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription tblPrChange'.split()
    order={s:i for i,s in enumerate(table_order)}
    for pr in main._tbl.iter(qn('w:tblPr')):
        if pr.getparent() is main._tbl:continue
        children=list(pr)
        for e in children:pr.remove(e)
        for e in sorted(children,key=lambda e:order.get(etree.QName(e).localname,100)):pr.append(e)
    # Keep the achievements label and its response on the same page.
    tr=main.rows[4]._tr.get_or_add_trPr()
    for item in list(tr.findall(qn('w:cantSplit'))):tr.remove(item)
    tr.append(OxmlElement('w:cantSplit'))
    ac.paragraphs[0].paragraph_format.keep_with_next=True
    symbols={'M₀':lambda:sub('M',N('0')),'Vₜ':lambda:sub('V','t'),'θₐ':lambda:sub('θ','a'),'μ̂':lambda:hat('μ'),'σ̂':lambda:hat('σ'),'Cₖ':lambda:sub('C','k'),'vₖ⁺':lambda:ss('v','k',N('+')),'vₖ':lambda:sub('v','k')}
    pat=re.compile('('+'|'.join(re.escape(s) for s in symbols)+')')
    for run in list(main._tbl.iter(qn('w:r'))):
        txt=''.join(run.xpath('./w:t/text()'))
        if not pat.search(txt):continue
        parent=run.getparent();where=parent.index(run);rp=run.find(qn('w:rPr'))
        for part in pat.split(txt):
            if not part:continue
            if part in symbols:node=E('oMath',symbols[part]())
            else:
                node=OxmlElement('w:r')
                if rp is not None:node.append(deepcopy(rp))
                text=OxmlElement('w:t');text.text=part;text.set(qn('xml:space'),'preserve');node.append(text)
            parent.insert(where,node);where+=1
        parent.remove(run)


def keep_caption_with_table(table):
    tbl = table._tbl
    caption = tbl.getprevious()
    parent = tbl.getparent()
    width = sum(int(col.get(qn('w:w'))) for col in tbl.find(qn('w:tblGrid')))
    wrapper = OxmlElement('w:tbl')
    props = OxmlElement('w:tblPr')
    wrapper.append(props)
    w = OxmlElement('w:tblW'); w.set(qn('w:w'), str(width)); w.set(qn('w:type'), 'dxa'); props.append(w)
    align = OxmlElement('w:jc'); align.set(qn('w:val'), 'center'); props.append(align)
    borders = OxmlElement('w:tblBorders'); props.append(borders)
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:' + side); border.set(qn('w:val'), 'nil'); borders.append(border)
    margins = OxmlElement('w:tblCellMar'); props.append(margins)
    for side in ('top', 'left', 'bottom', 'right'):
        margin = OxmlElement('w:' + side); margin.set(qn('w:w'), '0'); margin.set(qn('w:type'), 'dxa'); margins.append(margin)
    grid = OxmlElement('w:tblGrid'); col = OxmlElement('w:gridCol'); col.set(qn('w:w'), str(width)); grid.append(col); wrapper.append(grid)
    row = OxmlElement('w:tr'); row_props = OxmlElement('w:trPr'); row_props.append(OxmlElement('w:cantSplit')); row.append(row_props); wrapper.append(row)
    cell = OxmlElement('w:tc'); cell_props = OxmlElement('w:tcPr'); cell_width = OxmlElement('w:tcW'); cell_width.set(qn('w:w'), str(width)); cell_width.set(qn('w:type'), 'dxa'); cell_props.append(cell_width); cell.append(cell_props); row.append(cell)
    for page_break in list(caption.iter(qn('w:pageBreakBefore'))):
        page_break.set(qn('w:val'), '0')
    parent.insert(parent.index(caption), wrapper)
    cell.append(caption); cell.append(tbl)
    end = OxmlElement('w:p'); end_props = OxmlElement('w:pPr'); spacing = OxmlElement('w:spacing'); spacing.set(qn('w:line'), '20'); spacing.set(qn('w:lineRule'), 'exact'); end_props.append(spacing); end.append(end_props); cell.append(end)
