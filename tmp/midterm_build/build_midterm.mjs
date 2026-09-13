import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
import {finalizePresentation,applyPresentationChartFont} from '/Users/bytedance/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/bytedance/code/data-selection',build=path.join(root,'tmp/midterm_build');
const skill='/Users/bytedance/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const ref=path.join(build,'reference.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(ref));
const sources=[...p.slides.items], slides=[],tables=[],charts=[];
const F='微软雅黑',C={ink:'#373B48',body:'#52586B',blue:'#778EA8',line:'#C8CACF',light:'#E2E4E9',pale:'#F7F9FC',white:'#FFFFFF',accent:'#4874CB'};
function text(s,t,x,y,w,h,size=27,color=C.body,bold=false,align='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:F,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};return q;
}
function rect(s,x,y,w,h,fill=C.pale,line=C.line,r=true){return s.shapes.add({geometry:r?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1.2}});}
function line(s,x,y,w,h=0){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:h},line:{fill:C.blue,width:2}});}
function arrow(s,x,y,w=40,h=22){return s.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:h},fill:C.blue,line:{fill:'none',width:0}});}
function clone(idx){const s=sources[idx-1].duplicate();s.moveTo(p.slides.items.length-1);slides.push(s);return s;}
function body(title){const s=clone(13);const head=s.shapes.items.find(q=>q.name.startsWith('标题'));for(const q of [...s.shapes.items])if(q!==head)q.delete();for(const q of [...s.images.items])q.delete();head.text=title;head.text.style={typeface:F,fontSize:40,bold:true,color:'#000000'};return s;}
function foot(s,t){text(s,t,75,663,1090,40,18,C.body);}
function lead(s,t){text(s,t,75,156,1130,74,31,C.ink,true);}
function bottom(s,t,y=582){rect(s,75,y,1130,65,'#E9EEF5','#E9EEF5');text(s,t,95,y+15,1090,39,27,C.ink,true,'center');}
function card(s,x,y,w,h,title,bodyText,num=null){rect(s,x,y,w,h,C.pale,C.line);text(s,title,x+25,y+24,w-50,55,31,C.ink,true);line(s,x+25,y+90,w-50);text(s,bodyText,x+25,y+115,w-50,h-132,26);if(num!==null){rect(s,x+25,y-33,58,58,C.blue,C.blue,false);text(s,num,x+25,y-25,58,45,31,C.white,true,'center');}}
function tab(s,n,vals,{x=75,y=224,w=1130,h=323,widths=null,size=25}={}){const t=s.tables.add({rows:vals.length,columns:vals[0].length,left:x,top:y,width:w,height:h,values:vals,...(widths?{columnWidths:widths}:{})});t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({style:'solid',fill:C.line,width:1});for(let r=0;r<vals.length;r++){t.rows[r].height=h/vals.length;for(let c=0;c<vals[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?'#DEE5EF':r%2?'#FFFFFF':'#F6F8FB';cell.text.style={typeface:F,fontSize:size,color:r===0?C.ink:C.body,bold:r===0,alignment:'left',verticalAlignment:'middle',autoFit:'none',wrap:'square'};}}t.cells.block({row:0,column:0,rowCount:vals.length,columnCount:vals[0].length}).assign({margins:{left:16,right:16,top:10,bottom:10},anchor:'center'});tables.push(n);return t;}
function stage(s,x,y,w,title,detail){rect(s,x,y,w,92,C.light,C.line);text(s,title,x+10,y+16,w-20,40,29,C.ink,true,'center');if(detail)text(s,detail,x-12,y+109,w+24,78,24,C.body,false,'center');}
function divider(source,num,title,sub){const s=clone(source);const h=s.shapes.items.find(q=>q.name==='标题');h.text=title;h.text.style={typeface:F,fontSize:64,bold:true,color:'#262626'};if(sub)text(s,sub,590,430,610,58,26,C.body);s.shapes.items.find(q=>q.name==='节编号').text=num;return s;}
// 01 Cover uses the source cover and its master artwork.
{const s=clone(1);s.shapes.items.find(q=>q.name==='标题').text='面向大模型的数据与参数\n联合高效微调算法研究';text(s,'中期答辩',160,215,950,50,32,C.body,false,'center');}
// 02 Agenda preserves source numbered boxes and quote decoration.
{const s=clone(2);const titles=['课题进展概览','工作一：PCU-Select','工作二：实验管理系统','后续计划与总结'];let i=0;for(const q of s.shapes.items)if(q.name==='项标题'){q.text=titles[i++];q.text.style={typeface:F,fontSize:32,color:C.ink,bold:true};}}
divider(3,'01','课题进展','从开题目标到两项具体工作');
// 04
{const s=body('开题目标与中期聚焦');lead(s,'围绕数据选择与参数高效微调，形成两项相互衔接的工作');card(s,75,266,545,259,'工作一：PCU-Select','研究条件数据价值\n复用特征与评分器\n为不同目标生成各自子集','01');card(s,660,266,545,259,'工作二：实验管理系统','组织多任务、多配置实验\n识别可复用计算与产物\n支持批量执行和阶段恢复','02');bottom(s,'算法负责“如何选”，系统负责“如何可靠、重复地开展实验”');foot(s,'原开题中的动态 PEFT 路由保留为后续展望，本阶段聚焦算法与系统两项工作。');}
// 05
{const s=body('当前进展与待完成工作');lead(s,'方法、代码和实验基础已经形成，后续重点是系统化实现');tab(s,5,[['工作层次','当前基础','接下来完成'],['方法与论文','条件评分、配额选数流程\n已形成论文稿件','梳理贡献与边界\n完善论证和论文表述'],['算法实现','特征缓存、PEFT 注册\n训练评测与批量脚本','整理真实实验日志\n记录训练种子与分项计时'],['系统工作','可复用的算法模块\n已明确两项工程机制','依赖缓存、节点去重\n阶段恢复和控制台']],{y:244,h:348,widths:[204,448,478],size:27});foot(s,'状态口径：已有方法与代码基础；实验数据已更新；第二项系统为后续计划。');}
divider(7,'02','工作一','PCU-Select 条件数据选择方法');
// 07
{const s=body('研究问题：数据价值取决于目标条件');text(s,'u(x, p, t)',75,178,530,100,76,C.ink,true);text(s,'样本 x 在配置 p 下训练，\n对目标任务 t 有多大帮助？',655,183,550,104,34,C.ink,true);tab(s,7,[['条件','含义','输入来源'],['x：样本','候选指令与回答','共享候选数据池'],['p：PEFT 配置','更新位置、算子、容量与配方','结构化配置编码'],['t：目标任务','希望提升的下游能力','32 条训练或开发样本']],{y:331,h:259,widths:[230,476,424],size:26});foot(s,'共享评分器，根据目标条件重新评分；每个目标保留自己的训练子集。测试集只用于最终评估。');}
// 08
{const s=body('方法设计：统一表征，保留配置差异');lead(s,'用共享位点建立可比较的坐标，再显式描述样本、配置和任务');card(s,75,288,354,260,'样本表征 zₓ','语义 + 描述统计\n+ 激活特征\n合计 848 维','01');card(s,463,288,354,260,'PEFT 表征 zₚ','位点与算子掩码\n+ 容量 + 训练配方\n合计 128 维','02');card(s,851,288,354,260,'任务表征 zₜ','32 条任务样本\n汇总任务特征\n合计 848 维','03');bottom(s,'8 层 × 3 类模块输出 = 24 个共享位点',584);foot(s,'共同观察位置：Attention 输出、MLP 输出、残差输出；共享位点提供近似坐标。');}
// 09
{const s=body('方法设计：多保真监督与聚类配额');card(s,75,225,545,305,'先学会评分','低成本梯度相似度提供广泛监督\n少量短更新标签校正代理偏差\n输出效用均值 μ 与不确定性 σ');card(s,660,225,545,305,'再构建子集','使用 q = μ − 0.2σ 进行排序\n语义聚类，按簇价值分配预算\n各簇内部选取高分样本');bottom(s,'同时考虑目标效用与覆盖度，避免预算集中在少数语义区域');foot(s,'σ 在这里用于排序惩罚，不是置信区间。各预算独立分配配额，子集不要求互相嵌套。');}
// 10
{const s=body('PCU-Select 完整流程');text(s,'离线准备',75,176,200,45,30,C.ink,true);stage(s,270,170,232,'多保真效用标签','梯度代理 + 短更新');arrow(s,527,204);stage(s,590,170,232,'训练条件评分器','学习 u(x, p, t)');arrow(s,847,204);stage(s,910,170,270,'保存可复用资产','特征、编码、评分器');line(s,75,372,1130);text(s,'在线选数',75,403,200,45,30,C.ink,true);stage(s,270,396,232,'输入目标条件','数据池 + PEFT + 任务');arrow(s,527,430);stage(s,590,396,232,'条件评分与聚类','复用公共特征');arrow(s,847,430);stage(s,910,396,270,'配额选数与微调','预算 B 对应的独立子集');foot(s,'算法层面复用表示与评分器；第二项工作将这些产物的依赖、复用和执行过程系统化。');}
// 11
{const s=body('稿件中的质量与成本分析');text(s,'下游质量：PCU 与 LESS 接近',75,171,550,48,30,C.ink,true);tab(s,11,[['稿件均值','PCU-Select','LESS'],['四任务 × 五配置','35.18','35.14']],{x:75,y:235,w:550,h:160,widths:[218,182,150],size:25});text(s,'差值 +0.04，当前口径应描述为接近\n需要原始日志与独立重复支持结论',75,426,540,105,26);text(s,'选择侧成本：五套配置，四个任务',674,171,531,66,29,C.ink,true);
const ch=s.charts.add('bar',{position:{left:660,top:245,width:540,height:285},categories:['PCU-Select','LESS'],series:[{name:'GPU 小时',values:[151.6,316],fill:C.blue,valuesFormatCode:'0.0',points:[{idx:0,fill:'#778EA8'},{idx:1,fill:'#B7BBC5'}]}],hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:125},xAxis:{textStyle:{typeface:F,fontSize:24,fill:C.body},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min:0,max:400,majorUnit:100,numberFormatCode:'0',textStyle:{typeface:F,fontSize:20,fill:C.body},line:{fill:'none',width:0},majorGridlines:{fill:'#E5E7EB',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:F,fontSize:27,bold:true,fill:C.ink}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});applyPresentationChartFont(ch,{fontFamily:F});charts.push(11);bottom(s,'新加一套四任务配置的边际成本：PCU 1.51，LESS 63.2 GPU 小时');foot(s,'实验结果汇总。成本不含下游训练；PCU 总成本含 144 GPU 小时前期投入。');}
// 12
{const s=body('工作一小结与验证边界');lead(s,'当前已经形成条件选数方案，重点验证“质量保持与多配置摊销”');card(s,75,275,545,276,'方法贡献','条件效用明确数据价值的目标\n共享位点与配置编码支持复用\n多保真监督结合聚类配额');card(s,660,275,545,276,'实验记录与范围','真实训练记录与独立重复\n选择侧、前期投入的分项计时\n未见配置的兼容性与校准范围');bottom(s,'系统建设沿用已验证的骨干和 PEFT 支持范围');foot(s,'首版不将 Prefix、P-Tuning 等结构外推为零样本通用支持；LN-Tuning 校准实验。');}
divider(11,'03','工作二','多配置数据选择与微调实验管理系统');
// 14
{const s=body('系统目标：管理多配置实验的产物与执行');text(s,'基于 PCU-Select 的多配置数据选择与微调实验管理系统',75,155,1130,68,32,C.ink,true);tab(s,14,[['已有基础','第二项工作新增内容'],['样本特征和任务表征缓存','记录内容与版本依赖，识别可复用产物并局部重算'],['条件评分与聚类配额选数','公共评分、聚类独立持久化，为各预算生成独立子集'],['PEFT 注册和训练评测脚本','统一批量执行、阶段恢复、结果隔离与报告导出']],{y:247,h:320,widths:[390,740],size:26});bottom(s,'已有模块提供算法能力，新增机制解决正确复用与可靠执行');foot(s,'代码中的具体需求：预算变化时仍重复评分与聚类；选数结果路径需要纳入预算，避免覆盖。');}
// 15
{const s=body('机制一：分层缓存与增量复用');lead(s,'为每项产物记录依赖，输入变化时只重算受影响节点');stage(s,75,269,235,'样本特征','内容摘要、模型版本');arrow(s,335,303,38);stage(s,395,269,235,'目标评分','特征 + 任务 + PEFT');arrow(s,655,303,38);stage(s,715,269,235,'配额与子集','评分 + 聚类 + 预算');arrow(s,974,303,38);stage(s,1035,269,170,'训练评测','子集 + 训练种子');text(s,'任务表征：任务样本、顺序、预处理',75,471,535,71,27);text(s,'聚类：完整候选池、顺序、聚类种子',671,471,534,71,27);bottom(s,'依赖清单同时记录配置、随机种子与产物校验值');foot(s,'候选顺序纳入依赖；首版固定评分器版本。只有输入兼容、产物完整且校验通过，才允许命中。');}
// 16
{const s=body('机制一：明确复用与重算的边界');tab(s,16,[['输入变化','允许复用','需要重算'],['只改变选数预算','样本特征、任务表示\n目标评分、完整池聚类','配额与对应子集'],['只改变下游训练种子\n选择种子保持不变','全部选择结果','训练与评测'],['追加候选样本\n评分条件保持不变','旧样本的兼容特征\n旧样本逐样本评分','新增样本特征与评分\n完整池聚类、配额与子集'],['内容、版本或顺序变化','经依赖检查仍兼容的节点','受影响节点及其下游']],{y:192,h:420,widths:[348,383,399],size:26});foot(s,'同名样本内容改变不能命中旧结果；训练学习率等若属于 PEFT 条件，也会影响选择依赖。');}
// 17
{const s=body('机制二：公共计算去重与阶段恢复');lead(s,'把实验请求拆成依赖节点，相同输入的公共节点只执行一次');card(s,75,282,354,275,'合并公共计算','特征、任务表示、评分\n聚类、选数、训练、评测\n逐阶段判断输入是否相同','01');card(s,463,282,354,275,'保证产物完整','先写临时文件\n校验后原子提交\n写入完成标记','02');card(s,851,282,354,275,'恢复失败阶段','请求使用独立目录\n复用已完成产物并校验\n仅重跑失败阶段','03');bottom(s,'阶段恢复的目标：减少重复工作，并保持结果隔离');foot(s,'首版恢复到阶段边界；不承诺从训练中断的具体 step 继续执行。');}
// 18
{const s=body('具体示例：24 次训练如何共享前置计算');text(s,'2 种 PEFT × 2 个任务 × 3 个预算 × 2 个训练种子',75,163,1130,55,32,C.ink,true,'center');const xs=[75,365,655,945],counts=['4','1','12','24'],labels=['目标评分','完整池聚类','独立子集','训练实验'];for(let i=0;i<4;i++){rect(s,xs[i],274,260,241,C.pale,C.line);text(s,counts[i],xs[i]+20,300,220,103,82,C.ink,true,'center');text(s,labels[i],xs[i]+20,430,220,44,29,C.body,false,'center');}text(s,'按任务与 PEFT 区分',75,543,260,55,23,C.body,false,'center');text(s,'同一语义候选池',365,543,260,55,23,C.body,false,'center');text(s,'4 个目标 × 3 预算',655,543,260,55,23,C.body,false,'center');text(s,'12 子集 × 2 种子',945,543,260,55,23,C.body,false,'center');foot(s,'固定特征、评分器、聚类配置与选择种子。各目标使用自己的子集；这些执行次数不等于端到端加速倍数。');}
// 19
{const s=body('首版系统范围与操作流程');lead(s,'采用单工作节点，围绕已有算法模块完成可运行的实验闭环');tab(s,19,[['组件','职责'],['简洁控制台','登记数据与任务、提交矩阵、查看复用计划与执行状态'],['任务队列与执行器','按依赖安排阶段、去重公共节点、恢复失败阶段'],['元数据索引与本地产物库','维护依赖关系、版本、校验值、请求目录与结果']],{y:237,h:295,widths:[350,780],size:26});text(s,'登记数据与任务   →   提交实验矩阵   →   查看复用计划',75,564,1130,41,27,C.ink,true,'center');text(s,'执行选数   →   按需训练评测   →   导出报告',75,611,1130,41,27,C.ink,true,'center');foot(s,'继续沿用第一项工作已验证的骨干与 PEFT 范围；首版聚焦本地单工作节点。');}
// 20
{const s=body('定量验证：机制正确，也要测出实际收益');text(s,'对照组：已有脚本流程 / 加入依赖缓存 / 完整系统',75,159,1130,57,30,C.ink,true);tab(s,20,[['验证对象','实验设置','记录指标'],['分层缓存与增量复用','改预算、改训练种子\n追加 1% / 5% / 10% 样本\n同 ID 改内容、改变顺序','重算样本与节点数\n查验开销、选数耗时\n存储增长与结果一致性'],['去重与阶段恢复','冷启动运行 24 组矩阵\n在评分、写子集、训练时中断\n验证独立目录与恢复','节点执行与重复次数\n恢复耗时、额外重算量\n产物完整性与结果隔离']],{y:223,h:314,widths:[286,451,393],size:25});bottom(s,'正确性优先：与完整重算的评分、子集和依赖结果核对');foot(s,'冷启动去重与热缓存增量复用分开测量；分别报告选数时间和端到端时间，不叠加收益比例。');}
divider(18,'04','后续计划','分阶段交付，并行完善论文');
// 22
{const s=body('后续工作安排：建议中期后 8 周');tab(s,22,[['时间','系统工作','可检查的里程碑'],['第 1–2 周','确定范围与依赖清单\n拆分评分、聚类、配额接口','接口与目录规范\n最小选数流程运行'],['第 3–4 周','内容摘要与版本检查\n实现缓存与局部重算','输入变化场景通过\n无错误命中'],['第 5–6 周','节点去重、队列与阶段恢复\n控制台与报告导出','24 组实验矩阵运行\n中断恢复演示'],['第 7–8 周','两机制对照与开销测量\n系统整合、论文修改','可运行系统与验证报告\n论文与演示材料']],{y:175,h:393,widths:[187,471,472],size:25});bottom(s,'并行推进：整理实验记录、统计口径与配置支持范围',607);}
// 23
{const s=body('预期成果与验收标准');lead(s,'交付一套可运行系统，以及两项可定量检验的工程机制');card(s,75,283,354,294,'可运行系统','复现实验矩阵\n查看复用计划与阶段状态\n导出结果与依赖记录','01');card(s,463,283,354,294,'机制验证报告','缓存命中与重算边界正确\n统计重复计算和恢复开销\n报告时间与存储变化','02');card(s,851,283,354,294,'论文与实验材料','原始日志可追溯\n算法实验整理重复与计时\n系统设计和对照结果入文','03');foot(s,'不预设加速倍数；收益由实际运行记录决定，同时报告索引、校验和存储开销。');}
// 24
{const s=body('总结：算法进展与系统计划相互衔接');text(s,'总体目标：提高多配置微调中的数据使用效率与实验执行效率',75,160,1130,80,32,C.ink,true);card(s,75,278,545,280,'工作一：条件数据选择','形成 PCU-Select 方法与代码基础\n共享表征与评分器，保留目标条件\n继续分析实验结果与适用边界');card(s,660,278,545,280,'工作二：实验管理系统','复用第一项工作已有模块\n研究分层缓存、增量复用与阶段恢复\n以系统演示和定量对照完成验证');bottom(s,'下一阶段重点：把方法、实验记录与系统验证一并落到可复现结果');}
// 25 Thanks from original source.
{const s=clone(20);s.shapes.items.find(q=>q.name==='标题').text='感谢各位老师\n请批评指正';}
// 26
{const s=body('备用：缓存键与恢复边界');tab(s,26,[['产物','必须纳入的关键依赖'],['样本特征 / 任务表示','样本内容摘要与顺序、模型与预处理版本、特征配置'],['目标评分','特征校验值、任务表示、PEFT 编码、固定评分器版本'],['聚类 / 选择子集','完整候选池与顺序、聚类配置与种子；另含评分、预算、选择种子'],['训练 / 评测','子集校验值、模型、训练配方与种子、评测配置']],{y:198,h:360,widths:[290,840],size:25});text(s,'完成状态 = 产物存在 + 校验通过 + 完成标记\n请求目录区分任务、配置、预算与训练种子',75,592,1130,70,27,C.ink,true);}
// 27
{const s=body('备用：论文数据核验与支持范围');tab(s,27,[['检查项','实验记录','复现要求'],['主表独立重复','按训练种子汇总实验结果','整理训练日志与独立重复\n按任务与配置汇总'],['动机图与成本','PEFT 配置相关性实验\n成本为稿件口径','关联配置实验与分项计时\n区分前期、选择与训练成本'],['未见结构支持','结构外配置需兼容标签校准\n','明确适用范围与回退条件\n保留独立配置实验记录']],{y:206,h:347,widths:[230,445,455],size:25});bottom(s,'实验结果按质量、成本和配置支持范围分别解释');foot(s,'依据：competition_numbers.py、revise_motivation_figure.py、competition_revision_notes.md。');}
for(const s of sources)s.delete();
const data=JSON.parse(await fs.readFile(path.join(build,'midterm_content.json'),'utf8'));
for(let i=0;i<slides.length;i++){const s=slides[i],d=data.slides[i];s.speakerNotes.textFrame.setText([`第 ${i+1} 页；${d.seconds?`建议 ${d.seconds} 秒`:'备用页'}`,`本页要点：${d.takeaway}`,d.script,d.transition?`转场：${d.transition}`:'',`提示：${d.cue}`,'来源：',...(d.sources||[])].filter(Boolean).join('\n\n'));if(![1,2,3,6,13,21,25].includes(i+1))text(s,String(i+1).padStart(2,'0'),1181,680,30,23,15,C.body,false,'right');}
await fs.mkdir(path.join(build,'preview'),{recursive:true});
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);console.log('Exported '+slides.length+' slides');
for(let i=0;i<slides.length;i++){const b=await p.export({slide:slides[i],format:'png',scale:1});await fs.writeFile(path.join(build,'preview',`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));if((i+1)%4===0)console.log('Rendered '+(i+1));}
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'output/midterm/林肯达_中期答辩.pptx'),pythonExecutable:'/Users/bytedance/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...([...new Set(tables)].flatMap(n=>['--require-native-table-slide',String(n)]))],explicitTotalSlideCount:27,requiredNativeTableOwnerSlides:[...new Set(tables)],requiredNativeChartOwnerSlides:[...new Set(charts)],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:[F,'Arial'],referencePath:ref,referenceSha256:createHash('sha256').update(await fs.readFile(ref)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});console.log(JSON.stringify(result));
