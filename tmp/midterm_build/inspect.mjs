import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('tmp/midterm_build/reference.pptx'));
console.log(p.help('*',{search:'slide.delete|slides.delete|shape.delete|image.delete|slide.duplicate|slide.moveTo',include:['index','examples'],maxChars:6500}).ndjson);
console.log((await p.inspect({kind:'layout',maxChars:5000})).ndjson);
for(const i of [0,1,2,5,9,12,18,19]) {const s=p.slides.items[i]; console.log(JSON.stringify({slide:i+1,id:s.id,layoutId:s.layoutId,shapes:s.shapes.items.map(q=>({id:q.id,name:q.name,text:q.text.toString(),position:q.position})),images:s.images.items.map(q=>({id:q.id,alt:q.alt,frame:q.frame}))}));}
