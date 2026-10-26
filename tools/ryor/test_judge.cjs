const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const site='/Volumes/Tashon/opencv/showcase/site-next',ts=require(site+'/node_modules/typescript');
function load(name,req,fetcher){const m={exports:{}};vm.runInNewContext(ts.transpileModule(fs.readFileSync(`${site}/src/rescue/${name}.ts`,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,{module:m,exports:m.exports,require:req,fetch:fetcher,Map,Set,AbortSignal,performance,setTimeout,clearTimeout});return m.exports;}
const data=load('data',()=>{},()=>{}),live=load('live',n=>n==='./data'?data:{API:'https://test.invalid'});
const house={rooms_poly:[{id:'a',poly:[[0,0],[2,0],[2,2],[0,2]]}],people:5};
const frame=t=>({t,robots:[{id:'robot_A',x:1,y:1,yaw:0,out:t===180}],fire:[],found:[],events:[],seen:{robot_A:[]}});
const chunks=Array.from({length:19},(_,i)=>({chunk:i,frames:Array.from({length:i===18?1:20},(_,j)=>frame(i*10+j*.5))}));
const status={setup:'test__r1__easy',state:'done',chunks:19,validated:true,score:{robots_out:1,false_confirmations:0}};
const stream=live.streamBuilder(house,1,status.setup);
const first=stream.append(chunks.slice(0,6),status);
assert.equal(first.validated,false,'A completed server mission still arrives in pages; keep reading after page 1');
assert.equal(first.frames.length,120);
const second=stream.append(chunks.slice(6,12),status);assert.equal(second.validated,false);
const last=stream.append(chunks.slice(12),status);assert.equal(last.frames.length,361);assert.equal(last.validated,true);
assert.throws(()=>live.streamBuilder(house,1,status.setup).append([chunks[1]],status),/missing chunk/);
assert.throws(()=>live.streamBuilder(house,1,status.setup).append(chunks,{...status,setup:'different'}),/different setup/);
assert.throws(()=>live.streamBuilder(house,1,status.setup).append(chunks,{...status,score:{...status.score,false_confirmations:1}}),/failed its checks/);
(async()=>{let calls=0;const bad=load('data',()=>{},async()=>{calls++;return{ok:true,json:async()=>({validated:true,difficulty:'easy',setup:'old',frames:chunks.flatMap(c=>c.frames)})}});await assert.rejects(()=>bad.loadRecording('old'),/checked replay/);await assert.rejects(()=>bad.loadRecording('old'),/checked replay/);assert.equal(calls,2,'Failed fallback promises must not stay cached');console.log('Paging, order, wrong setup, false confirmations and old fallback rejection passed');})().catch(e=>{console.error(e);process.exit(1)});
