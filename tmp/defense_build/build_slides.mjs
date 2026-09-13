import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { finalizePresentation, applyPresentationChartFont } from '/Users/bytedance/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';

const root='/Users/bytedance/code/data-selection';
const build=path.join(root,'tmp/defense_build');
const SKILL='/Users/bytedance/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const data=JSON.parse(await fs.readFile(path.join(build,'defense_content.json'),'utf8'));
const FONT='Heiti TC';
const C={bg:'#FAFBFD',ink:'#172F4D',teal:'#007E85',muted:'#57687C',line:'#DCE3EB',light:'#EAF3F5',purple:'#63599A',gold:'#94631B',white:'#FFFFFF',gray:'#97A5B8',pale:'#F0F3F7'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const tableOwners=[],chartOwners=[];

function txt(s,text,x,y,w,h,size=28,color=C.ink,bold=false,align='left'){
  const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  q.text=text;
  q.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};
  return q;
}
function slide(n,subtitle='',dark=false){
  const s=p.slides.add();s.background.fill=dark?C.ink:C.bg;
  const info=data.slides[n-1];
  if(n!==1){txt(s,info.title,64,42,1152,64,44,dark?C.white:C.ink,true);}
  if(subtitle)txt(s,subtitle,64,121,1145,58,25,dark?'#DAE8F2':C.muted);
  txt(s,String(n).padStart(2,'0'),1174,666,42,26,18,dark?'#BDCCDB':C.muted,false,'right');
  s.speakerNotes.textFrame.setText([
    n<=18?`建议讲述 ${info.seconds} 秒`:'备用页，按追问使用',
    `本页核心：${info.takeaway}`,
    '逐字讲稿：',info.script,
    info.transition?`过渡：${info.transition}`:'',
    `讲述提示：${info.cue}`,
    '来源：',...info.sources.map(x=>path.join(root,x)),
    n>=11&&n<=17||n===20||n===21?`数据状态：${data.evidenceStatus}`:''
  ].filter(Boolean).join('\n\n'));
  return s;
}
function foot(s,text,evidence=false){txt(s,text,64,648,1084,47,18,evidence?C.gold:C.muted);}
function takeaway(s,text,y=571){txt(s,text,64,y,1152,56,30,C.teal,true);}
function table(s,n,values,{x=64,y=203,w=1152,h=330,widths=null,fs=26,highlightLast=false}={}){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
  t.styleOptions={headerRow:false,bandedRows:false};
  t.borders.assign({style:'solid',fill:C.line,width:1});
  for(let r=0;r<values.length;r++){
    t.rows[r].height=h/values.length;
    for(let c=0;c<values[0].length;c++){
      const cell=t.getCell(r,c);
      cell.fill=r===0?C.ink:(highlightLast&&r===values.length-1?C.light:C.white);
      cell.text.style={typeface:FONT,fontSize:fs,color:r===0?C.white:C.ink,bold:r===0||highlightLast&&r===values.length-1,alignment:c===0?'left':'center',verticalAlignment:'middle',autoFit:'none',wrap:'square'};
    }
  }
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:15,right:15,top:10,bottom:10},anchor:'center'});
  tableOwners.push(n);return t;
}
const axisStyle={typeface:FONT,fontSize:22,fill:C.muted};
const labelStyle={typeface:FONT,fontSize:24,fill:C.ink,bold:true};
function bar(s,n,categories,values,{x=64,y=198,w=800,h=367,max=40,step=10,fmt='0.00',colors=null,horizontal=false}={}){
  const ch=s.charts.add('bar',{
    position:{left:x,top:y,width:w,height:h},categories,
    series:[{name:'数值',values,fill:C.teal,valuesFormatCode:fmt,...(colors?{points:colors.map((fill,idx)=>({idx,fill,line:{fill,width:0}}))}:{})}],
    hasLegend:false,barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:80},
    xAxis:{textStyle:axisStyle,line:{fill:C.line,width:1},majorGridlines:null,numberFormatCode:horizontal?fmt:undefined,...(horizontal?{min:0,max,majorUnit:step}:{})},
    yAxis:{textStyle:axisStyle,line:{fill:C.line,width:1},majorGridlines:horizontal?null:{fill:C.line,width:1},numberFormatCode:horizontal?undefined:fmt,...(!horizontal?{min:0,max,majorUnit:step}:{})},
    dataLabels:{showValue:true,position:'outEnd',textStyle:labelStyle},
    chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
  });applyPresentationChartFont(ch,{fontFamily:FONT});chartOwners.push(n);return ch;
}

// 01
{
const s=slide(1,'',true);
txt(s,'论文课题答辩',68,65,900,42,26,'#BDCEDD');
txt(s,'PCU-Select',64,218,1150,115,98,C.white,true);
txt(s,'跨 PEFT 配置复用的数据选择',68,361,1140,75,50,'#F4F9FC',true);
txt(s,'Amortizing Target-Specific Data Selection\nAcross PEFT Configurations',70,481,1090,77,27,'#BDCEDD');
txt(s,'方法设计与实验分析',70,640,1040,34,24,'#BDCEDD');
}
// 02
{
const s=slide(2,'训练只使用部分数据，同时要适配多套微调配置');
txt(s,'300,000',65,202,520,110,82,C.ink,true);
txt(s,'候选指令样本',68,316,520,43,28,C.muted);
txt(s,'30,000',706,202,505,110,82,C.teal,true);
txt(s,'每个目标选用 10%',710,316,500,43,28,C.muted);
txt(s,'PEFT 参数高效微调',67,422,1065,43,32,C.ink,true);
txt(s,'只训练少量参数。配置决定更新的位置、方式和容量。',67,483,1140,54,29);
takeaway(s,'研究场景：同一基础模型和候选池，持续增加微调配置');
foot(s,'300K 候选池与 10% 预算来自论文实验设定');
}
// 03
{
const s=slide(3,'比较目标针对性与跨配置复用方式');
table(s,3,[['方法','接收目标配置','每套配置的主要工作'],['RDS+ / 共享 Influence','否','复用任务相关排序'],['每套配置运行 LESS','是','重建梯度缓存，再排序'],['复用一套 LESS 子集','仅源配置','直接沿用所选子集'],['PCU-Select','是','复用特征和评分器']],{h:338,widths:[376,247,529],fs:27,highlightLast:true});
takeaway(s,'复用昂贵计算，保留每个目标自己的训练子集');
foot(s,'RDS+ 与 Influence 指本文实现的基线；LESS 采用逐 PEFT 重算实现');
}
// 04
{
const s=slide(4,'假设：可训练方向变化，会改变样本对目标任务的作用');
table(s,4,[['检验层次','如何比较','需要的对照'],['排序与选集','Spearman 排序相关\nTop-5% 集合重合度','同配置独立重复\n跨配置比较'],['下游表现','A 配置选数，B 配置训练\n与 B 自己选数比较','固定训练预算\n配对训练种子']],{y:223,h:286,widths:[235,510,407],fs:27});
takeaway(s,'只有排序差异与下游差异结合，才能支持配置依赖性');
foot(s,'固定候选池与目标任务，比较配置间的排序和子集差异',true);
}
// 05
{
const s=slide(5,'条件效用是评分器学习的目标');
txt(s,'u(x, p, t)',65,202,560,97,78,C.teal,true);
txt(s,'在配置 p 下训练样本 x，\n对任务 t 能带来多少帮助',677,216,534,102,31,C.ink,true);
table(s,5,[['符号','含义','如何提供'],['x','候选指令与回答','候选数据池'],['p','PEFT 配置','作用位置、算子、容量与配方'],['t','目标任务','32 条训练或开发样本']],{y:356,h:220,widths:[150,371,631],fs:26});
foot(s,'目标是在预算 B 内选择子集。测试集只用于最终评估，不进入训练与选数。');
}
// 06
{
const s=slide(6,'不同参数空间，通过共同的模块输出位置建立联系');
txt(s,'8 层 × 3 类输出 = 24 位点',66,202,1150,69,49,C.teal,true);
table(s,6,[['共同观察位置','主要映射配置','更新特征由编码区分'],['Attention 输出','注意力 LoRA、IA3-attn','加性或乘性'],['MLP 输出','MLP LoRA、IA3-FFN','加性或乘性'],['残差输出','Adapter','瓶颈更新']],{y:301,h:266,widths:[342,485,325],fs:26});
foot(s,'位点提供可复用的近似坐标；算子、容量和训练配方保留进一步的结构区别。');
}
// 07
{
const s=slide(7,'样本、任务和配置共同调节效用预测');
table(s,7,[['输入','信息内容','维度'],['样本 zₓ','语义 768 + 描述 16 + 激活 64','848'],['任务 zₜ','32 条任务样本表示的均值','848'],['配置 zₚ','位点与算子 96 + 容量 16 + 配方 16','128']],{y:208,h:321,widths:[223,734,195],fs:28});
takeaway(s,'同一条样本遇到不同的配置或任务，可以获得不同分数');
foot(s,'位点梯度另存，用于离线监督标签；不拼接到线上任务表示 zₜ。');
}
// 08
{
const s=slide(8,'低保真代理覆盖全池，高保真短更新提供更直接的标签');
table(s,8,[['','梯度代理标签','短更新标签'],['直观含义','样本与任务的更新方向是否一致','少量更新后任务损失是否下降'],['计算方式','目标位点加权的梯度余弦','更新前损失减去更新后损失'],['覆盖与设置','缓存复用，覆盖候选池','10K 标签，2 个起点，1 / 4 步']],{y:220,h:299,widths:[185,464,503],fs:25});
txt(s,'便宜但间接',257,541,435,44,29,C.teal,true,'center');
txt(s,'更直接，但仍是短期近似',733,541,466,44,29,C.purple,true,'center');
foot(s,'短更新使用目标样本与 7 条匹配样本。标签在相同条件桶内做秩归一化。');
}
// 09
{
const s=slide(9,'共享评分器把前期监督转化为可复用的选择能力');
table(s,9,[['离线  一次前期投入','在线  每个配置与任务'],['提取并缓存样本特征','输入配置编码和任务表示'],['构造代理与短更新标签','用同一评分器预测效用与噪声'],['训练条件评分器','保守排序，再按预算选择']],{y:205,h:285,widths:[576,576],fs:29});
txt(s,'q = μ − 0.2σ',66,537,546,76,53,C.teal,true);
txt(s,'预测效用高、噪声小的样本更优先',636,548,564,72,29);
foot(s,'σ 用作排序惩罚，不是置信区间。超出支持范围的目标可能需要额外校准。');
}
// 10
{
const s=slide(10,'在高分样本中保留语义覆盖，控制近似重复');
txt(s,'1',65,211,58,66,49,C.teal,true);txt(s,'按语义分簇',157,212,1000,56,35,C.ink,true);
txt(s,'300K 候选样本，对应 547 个簇',157,278,1010,46,29,C.muted);
txt(s,'2',65,351,58,66,49,C.teal,true);txt(s,'按簇价值和规模分配预算',157,352,1050,56,35,C.ink,true);
txt(s,'高价值簇分得更多，同时考虑簇规模，γ = 0.6',157,418,1030,46,29,C.muted);
txt(s,'3',65,493,58,66,49,C.teal,true);txt(s,'簇内按保守分数选择',157,494,1000,56,35,C.ink,true);
txt(s,'确定性处理取整余量，使总数满足预算',157,559,1030,46,29,C.muted);
foot(s,'簇价值取簇内 Top-10% 样本保守分数的均值；仅正价值参与配额分配。');
}
// 11
{
const s=slide(11,'Llama-2-7B，五套已见配置，候选池 300K，选择预算 10%');
table(s,11,[['任务','能力','指标'],['GSM8K','数学推理','Exact Match'],['HumanEval','代码生成','Pass@1'],['MMLU','知识与理解','Accuracy'],['TyDiQA','多语言问答','F1']],{y:195,h:310,widths:[353,410,389],fs:27});
txt(s,'统一实验协议',66,538,1135,41,29,C.gold,true);
txt(s,'各方法使用相同的目标训练与评测协议。',66,588,1125,41,27,C.gold);
foot(s,'固定选择流程，比较 3 个下游训练种子；相同配置训练 1000 步。');
}
// 12
{
const s=slide(12,'四任务原生百分制指标的等权平均，分数越高越好');
bar(s,12,['Random','RDS+','Influence','复用 LESS','LESS','PCU-Select'],[32.29,34.44,34.68,34.24,35.14,35.18],{x:64,y:202,w:861,h:368,max:40,step:10,colors:[C.gray,C.gray,C.gray,C.gray,C.purple,C.teal]});
txt(s,'+0.04',952,223,263,83,58,C.teal,true);
txt(s,'相对 LESS 的点差',952,309,263,77,27,C.ink);
txt(s,'10 / 20',952,417,263,62,41,C.ink,true);
txt(s,'配置与任务单元胜出',952,480,260,76,25,C.muted);
takeaway(s,'当前均值接近；描述性区间 [−0.26, +0.33] 跨零',581);
foot(s,'实验结果；描述性统计不用于宣称统计显著。',true);
}
// 13
{
const s=slide(13,'PCU-Select 相对逐配置 LESS 的点差，按五套配置平均');
table(s,13,[['任务','PCU-Select','相对 LESS'],['GSM8K','22.20','+0.29'],['HumanEval','17.06','−0.04'],['MMLU','49.14','+0.67'],['TyDiQA','52.30','−0.77']],{y:211,h:309,widths:[414,353,385],fs:31});
takeaway(s,'关注具体任务的收益与退化，不能只看总平均');
foot(s,'实验结果；各行指标不同，点差基于未四舍五入的原始稿件均值。',true);
}
// 14
{
const s=slide(14,'每套配置包含四个任务，纵轴仅计选择成本，单位 GPU 小时');
const ch=s.charts.add('line',{
  position:{left:64,top:214,width:800,height:360},categories:['1','2','3','4','5'],
  series:[
    {name:'每配置 LESS',values:[63.2,126.4,189.6,252.8,316.0],line:{fill:C.purple,width:4},marker:{symbol:'circle',size:8}},
    {name:'PCU-Select',values:[145.51,147.02,148.53,150.04,151.55],line:{fill:C.teal,width:4},marker:{symbol:'circle',size:8}}
  ],hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:axisStyle},lineOptions:{smooth:false},
  xAxis:{title:{text:'四任务配置数量',textStyle:axisStyle},textStyle:axisStyle,line:{fill:C.line,width:1},majorGridlines:null},
  yAxis:{min:0,max:350,majorUnit:100,numberFormatCode:'0',textStyle:axisStyle,majorGridlines:{fill:C.line,width:1},line:{fill:'none',width:0}},
  chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
});applyPresentationChartFont(ch,{fontFamily:FONT});chartOwners.push(14);
txt(s,'约 42 倍',913,214,302,64,49,C.teal,true);
txt(s,'新增配置边际成本比\n63.2 ÷ 1.51',914,288,293,85,26);
txt(s,'约 2.09 倍',913,418,302,64,45,C.ink,true);
txt(s,'五配置选择总成本比\n316.0 ÷ 151.6',914,491,300,85,26);
takeaway(s,'前期投入 144 GPU 小时，约从第 3 套配置起摊回',585);
foot(s,'按论文成本模型推算；42 倍适用于无需校准的新配置，不含下游训练。',true);
}
// 15
{
const s=slide(15,'相对完整方法的质量下降，单位为任务原生指标点');
bar(s,15,['去配置编码','全局 Top-k','仅代理标签','去任务表示','仅家族编码'],[1.37,1.08,0.85,0.82,0.68],{x:65,y:225,w:1149,h:320,max:1.6,step:.4,colors:[C.teal,C.purple,C.gray,C.gray,C.gray]});
takeaway(s,'消融分别检验条件信息、监督信号和覆盖策略');
foot(s,'实验结果；仅 GSM8K、HumanEval 与两套代表配置，不等同于主表评价范围。',true);
}
// 16
{
const s=slide(16,'支持程度相对于评分器训练时的配置注册表定义');
table(s,16,[['层次','结构变化','使用策略'],['L0','接近已见配置','直接评分'],['L1','已见家族内较大变化','用少量兼容标签校准'],['L2','未见方法家族','可校准则校准，否则逐目标选数']],{y:218,h:288,widths:[151,434,567],fs:28});
txt(s,'校准只训练线性残差头，共享评分器保持冻结',66,543,1145,49,30,C.teal,true);
foot(s,'Prefix / P-Tuning 不支持零样本使用，校准依赖外部兼容标签。',true);
}
// 17
{
const s=slide(17,'方法适合固定候选池与模型家族内的重复配置接入');
table(s,17,[['边界','下一步验证'],['短更新近似长期训练贡献','比较不同起点、更新步数与最终下游收益'],['模型、任务与配置覆盖有限','扩展 backbone 与任务，评估缓存更新成本'],['跨家族校准开销','扩展任务与配置覆盖，比较校准成本']],{y:226,h:298,widths:[536,616],fs:27});
takeaway(s,'扩展任务与配置覆盖，继续分析泛化边界');
foot(s,'方法适用范围限定同一候选池和模型家族。',true);
}
// 18
{
const s=slide(18,'',true);
txt(s,'配置条件保留下来，\n选数计算复用起来',65,200,1150,159,63,C.white,true);
txt(s,'条件效用 u(x, p, t) 明确研究对象\n共享位点与结构化编码支持复用\n多配置成本分析检验摊销价值',68,404,1105,155,31,'#DDE8F1');
txt(s,'谢谢各位老师，请批评指正',69,629,1100,42,28,'#DDE8F1');
}
// 19
{
const s=slide(19,'完整细节供问答时展开');
const bytes=await fs.readFile(path.join(root,'paper/Figures/image.png'));
s.images.add({blob:new Uint8Array(bytes),contentType:'image/png',alt:'论文 PCU-Select 完整架构图，含离线编码、监督学习和在线选数',fit:'contain',position:{left:54,top:199,width:1172,height:435}});
foot(s,'来源：论文方法架构原图；输入维度和成本口径以正文讲解为准。');
}
// 20
{
const s=slide(20,'四任务原生指标的等权平均，仅列当前稿件均值');
table(s,20,[['方法','AD-b64','IA3','L-r16\nqkvo','L-r8\nmlp','L-r8\nqv','平均'],['Random','32.79','31.74','33.11','31.97','31.82','32.29'],['RDS+','35.27','33.54','35.00','34.46','33.91','34.44'],['Influence','35.08','33.84','35.24','34.69','34.53','34.68'],['LESS','35.52','34.44','35.80','35.13','34.80','35.14'],['PCU-Select','35.61','34.60','35.93','34.56','35.18','35.18']],{y:210,h:369,widths:[222,155,155,155,155,155,155],fs:24,highlightLast:true});
foot(s,'实验结果。IA3 指 IA3-attnmlp；L 为 LoRA，AD 为 Adapter。',true);
}
// 21
{
const s=slide(21,'所有成本单位为 GPU 小时，属于稿件成本模型');
table(s,21,[['成本项','LESS','PCU-Select'],['共享前期投入','0','144.00'],['已有配置新增一个任务','0.30','0.3775'],['新增一套四任务配置','63.20','1.51'],['每任务 500 标签校准','不适用','额外 1.05']],{y:209,h:318,widths:[622,265,265],fs:27});
txt(s,'C_LESS(P) = 63.2P',66,553,565,57,31,C.purple,true);
txt(s,'C_PCU(P) = 144 + 1.51P',668,553,550,57,31,C.teal,true);
foot(s,'每套配置都做四任务校准时，PCU 斜率变为 5.71，平衡点约 2.50；不含下游训练。',true);
}
// 22
{
const s=slide(22,'先保证数据角色隔离，再解释局部监督的含义');
table(s,22,[['数据 / 信号','用途','关键约束'],['任务样本集','描述任务、构造效用标签','来自训练或开发数据'],['最终测试集','衡量下游表现','不参与评分器训练和选数'],['短更新损失差','近似条件效用','目标样本加 7 条匹配样本'],['桶内秩归一化','比较相对效用','固定配置、任务、起点与步数']],{y:222,h:333,widths:[252,440,460],fs:26});
foot(s,'HumanEval 使用 MBPP / CodeAlpaca 构建任务样本，并进行内容和 AST 签名去重。');
}

await fs.mkdir(path.join(build,'preview'),{recursive:true});
const version=process.env.DEFENSE_VERSION||'v1';
const candidatePath=path.join(build,`candidate-${version}.pptx`);
await (await PresentationFile.exportPptx(p)).save(candidatePath);
console.log(`Exported ${p.slides.items.length} slides: ${candidatePath}`);
await fs.writeFile(path.join(build,'native_manifest.json'),JSON.stringify({tables:[...new Set(tableOwners)],charts:[...new Set(chartOwners)],font:FONT},null,2));
// Preview every authored slide. Final file is rendered again after validation.
for(let i=0;i<p.slides.items.length;i++){
  const s=p.slides.items[i];
  const png=await p.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(build,'preview',`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
  if((i+1)%4===0)console.log(`Preview ${i+1}`);
}
const finalName=version==='v1'?'PCU-Select_论文答辩.pptx':`PCU-Select_论文答辩_${version}.pptx`;
const finalPath=path.join(root,'output/defense',finalName);
const result=await finalizePresentation({
  workspaceDir:root,candidatePath,finalPath,
  pythonExecutable:'/Users/bytedance/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:22,
  requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:[...new Set(chartOwners)],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(build,`validation-${version}.json`)
});
console.log(JSON.stringify(result,null,2));
