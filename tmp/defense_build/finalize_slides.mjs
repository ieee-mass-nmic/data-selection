import fs from 'node:fs/promises';
import path from 'node:path';
import { finalizePresentation } from '/Users/bytedance/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/bytedance/code/data-selection';
const build=path.join(root,'tmp/defense_build');
const skill='/Users/bytedance/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const m=JSON.parse(await fs.readFile(path.join(build,'native_manifest.json'),'utf8'));
console.log(await finalizePresentation({
  workspaceDir:root,candidatePath:path.join(build,'candidate-v1.pptx'),finalPath:path.join(root,'output/defense/PCU-Select_论文答辩.pptx'),
  pythonExecutable:'/Users/bytedance/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...m.tables.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:22,requiredNativeTableOwnerSlides:m.tables,requiredNativeChartOwnerSlides:m.charts,
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[m.font]},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'validation-v1.json')
}));
