import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
const source=fs.readFileSync('version/v8/tools/ir_visualizer.html','utf8');
const start=source.indexOf('function componentText(');
const end=source.indexOf('function renderMultimodal()',start);
const context={escapeHtml:value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;')};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const payload={schema:'cke.visualizer.components.v1',issues:[],components:[{id:'encoder',artifacts:[
    {path:'encoder/call.json',schema:'lowered-ir-v3',phase:'prefill',model:'new_family',providers:['real_kernel'],buffer_bindings:['state'],reported_status:'PASS',sha256:'123'}]}],
    catalog:[{name:'qwen_tts',family:'bounded_graph',path:'talker.json',contract_sections:['runtime_invariants'],runtime_invariants:{complete_synthesis:false}}]};
test('renders artifacts, provenance and declared-not-certified boundaries',()=>{
    const html=context.buildComponentInventoryHTML(payload);
    assert.match(html,/real_kernel/);assert.match(html,/Reported PASS — not independently verified/);
    assert.match(html,/complete_synthesis/);assert.match(html,/false/);
    assert.match(html,/not execution evidence/);assert.match(html,/not verified ownership/);
    assert.match(context.componentSummary(payload),/Artifact components: 1/);
});
test('missing reports and unknown schemas do not invent a pipeline',()=>{
    assert.match(context.buildComponentInventoryHTML(null),/No component inventory/);
    assert.match(context.buildComponentInventoryHTML({schema:'unknown'}),/No component inventory/);
    assert.match(context.buildComponentInventoryHTML({...payload,components:[]}),/No matching component artifacts/);
});
test('escaping and filtering apply to artifact and declaration text',()=>{
    const hostile=structuredClone(payload);
    hostile.components[0].id='<img src=x onerror=alert(1)>';
    const html=context.buildComponentInventoryHTML(hostile);
    assert.ok(!html.includes('<img'));assert.match(html,/&lt;img/);
    assert.ok(!context.buildComponentInventoryHTML(payload,'unmatched').includes('real_kernel'));
    assert.match(context.buildComponentInventoryHTML({...payload,issues:['missing call.json']}),/Inventory gaps/);
});

test('operation dependencies preserve unresolved and truncated evidence',()=>{
    const html=context.buildComponentOperationHTML({total_operations:4,truncated:true,nodes:[
        {id:'1',operation:'recurrent_update',provider:'<unsafe>',layer:0,outputs:['state'],inputs:[
            {input:'history',from_op:'0',from_output:'state',status:'unresolved_or_outside_capture'}]}]});
    assert.match(html,/1 of 4/);assert.match(html,/Capture truncated/);
    assert.match(html,/unresolved_or_outside_capture/);assert.match(html,/&lt;unsafe&gt;/);
    assert.ok(!html.includes('<unsafe>'));
    assert.match(context.buildComponentOperationHTML(null),/No captured operations/);
});
