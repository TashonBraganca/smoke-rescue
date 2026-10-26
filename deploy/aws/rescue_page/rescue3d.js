/* Uses the existing house viewer's three.js r128, orbit camera, merged walls and box furniture.
   Coordinates are always the brain's gate frame. Geometry stays viewer/world-side. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const clone = value => JSON.parse(JSON.stringify(value));
  const ROBOT = {robot_A:0x40bfff,robot_B:0xff65bb,robot_C:0xb6ce68}; // Match the plan replay's robot colours.
  const FLOOR = {bedroom:0x7c8274,bathroom:0x789599,kitchen:0x9b886b,living:0xa28d79,corridor:0x747a80,laundry:0x827f98,study:0x9b876f}; // Room kinds follow the existing viewer's material colours.
  let editor = null, player = null, draft = null, house = null, lastValid = null, history = [], revision = 0, selected = null, pending = false;
  let moveMode = true, drag = null, lastFrame = null, textures = {}, replayHouse = null, replayMission = null;
  const message = (text, bad) => { $('layout-message').textContent = text; $('layout-message').className = bad ? 'bad' : ''; };

  function dispose(group) {
    group.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) { const materials = Array.isArray(o.material) ? o.material : [o.material]; materials.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); }); } });
    group.clear();
  }
  function box(parent, x, y, z, w, h, d, colour, opacity) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshLambertMaterial({color:colour,transparent:opacity!==undefined,opacity:opacity===undefined?1:opacity}));
    mesh.position.set(x,y,z);parent.add(mesh);return mesh;
  }
  function label(parent, text, x, y, z, width) {
    const cv=document.createElement('canvas');cv.width=512;cv.height=96;
    const ctx=cv.getContext('2d');ctx.fillStyle='rgba(17,22,25,.88)';ctx.fillRect(0,0,512,96);ctx.fillStyle='#f2f0e9';ctx.font='32px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,256,48,490);
    const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),depthTest:false}));s.position.set(x,y,z);s.scale.set(width,width*96/512,1);parent.add(s);return s;
  }
  class View {
    constructor(host) {
      this.host=host;this.renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
      this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.appendChild(this.renderer.domElement);
      this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0x151c20);
      this.camera=new THREE.PerspectiveCamera(48,1,.05,300); // Same perspective/orbit construction as the existing viewer.
      this.orbit=new THREE.OrbitControls(this.camera,this.renderer.domElement);this.orbit.enableDamping=true;this.orbit.maxPolarAngle=Math.PI*.48;
      this.scene.add(new THREE.HemisphereLight(0xf9f4e9,0x545c55,.85));
      const sun=new THREE.DirectionalLight(0xffefda,.8);sun.position.set(-10,25,-8);this.scene.add(sun);
      this.base=new THREE.Group();this.dynamic=new THREE.Group();this.search=new THREE.Group();this.scene.add(this.base,this.search,this.dynamic);
      this.pick=[];this.roomMeshes={};this.active=true;
      new ResizeObserver(()=>this.resize()).observe(host);
      const animate=()=>{requestAnimationFrame(animate);if(!this.host.offsetWidth||!this.host.offsetHeight||!this.active)return;this.orbit.update();this.renderer.render(this.scene,this.camera);};animate();
    }
    resize(){const w=this.host.clientWidth,h=this.host.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
    build(h, editing, reset) {
      dispose(this.base);dispose(this.dynamic);dispose(this.search);this.pick=[];this.roomMeshes={};
      const rooms=h.placement_geometry.rooms;
      rooms.forEach(r=>{const [x0,y0,x1,y1]=r.rect;const mesh=box(this.base,(x0+x1)/2,-.06,-(y0+y1)/2,x1-x0,.12,y1-y0,FLOOR[r.kind]||0x847c70);mesh.userData.room=r.id;this.pick.push(mesh);this.roomMeshes[r.id]=mesh;
        if(editing)label(this.base,r.name,(x0+x1)/2,.65,-(y0+y1)/2,Math.min(3.4,x1-x0-.15));});
      h.walls.forEach(w=>{const [a,b,c,d]=w;const length=Math.hypot(c-a,d-b);const wall=box(this.base,(a+c)/2,editing?.2:.48,-(b+d)/2,length,editing?.4:.96,.10,0xd2c9b8);wall.rotation.y=Math.atan2(d-b,c-a);}); // Cut walls retain sight of the moving robots.
      h.furniture.forEach(poly=>{const xs=poly.map(p=>p[0]),ys=poly.map(p=>p[1]);box(this.base,(Math.min(...xs)+Math.max(...xs))/2,.30,-(Math.min(...ys)+Math.max(...ys))/2,Math.max(...xs)-Math.min(...xs),.60,Math.max(...ys)-Math.min(...ys),0x454e54);}); // Furniture boxes show exact footprints; their 0.6 m illustration height is not sensor geometry.
      const door=h.door;if(door){box(this.base,door[0],.03,-door[1],1.25,.06,.16,0xffb866);label(this.base,'Fixed front door',door[0],.65,-door[1]+.6,3.0);}
      (h.pads||[]).forEach((p,i)=>box(this.base,p[0],-.02,-p[1],.55,.05,.55,ROBOT['robot_'+'ABC'[i]],.65));
      if(reset){const b=h.bounds,cx=(b[0]+b[2])/2,cy=(b[1]+b[3])/2,span=Math.max(b[2]-b[0],b[3]-b[1]);this.orbit.target.set(cx,0,-cy);this.camera.position.set(cx+(editing?0:span*.65),span*(editing?1.55:1.05),-cy+span*(editing?.12:.85));this.orbit.update();}
      this.resize();
    }
    tint(id,bad){Object.values(this.roomMeshes).forEach(m=>{m.material.emissive.setHex(m.userData.room===id?(bad?0x8b1f16:0x253644):0x000000);});}
    point(event){const rect=this.renderer.domElement.getBoundingClientRect(),p=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);const ray=new THREE.Raycaster();ray.setFromCamera(p,this.camera);const out=new THREE.Vector3();return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),out)?[out.x,-out.z]:null;}
    roomAt(event){const rect=this.renderer.domElement.getBoundingClientRect();const ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),this.camera);const hit=ray.intersectObjects(this.pick);return hit.length?hit[0].object.userData.room:null;}
  }
  function ensureEditor(){if(editor)return true;try{editor=new View($('layout-3d'));wireDrag();return true;}catch(e){message('3D could not start on this browser. '+e.message,true);return false;}}
  function ensurePlayer(){if(player)return true;try{player=new View($('replay-3d'));return true;}catch(e){$('three-caption').textContent='3D is unavailable on this browser. Use Plan to watch the full replay.';return false;}}
  function draftHouse(){const h=clone(house),old=new Map(house.placement_geometry.rooms.map(r=>[r.id,r]));h.placement_geometry.rooms.forEach(r=>{const next=draft.find(v=>v.id===r.id);const dx=next.rect[0]-r.rect[0],dy=next.rect[1]-r.rect[1];r.rect=next.rect;h.furniture.forEach((poly,i)=>{if(house.placement_geometry.furniture[i].room===r.id)poly.forEach(p=>{p[0]+=dx;p[1]+=dy;});});});return h;}
  function fields(){const r=draft.find(r=>r.id===selected);if(!r)return;$('layout-room').value=selected;$('room-x').value=r.rect[0];$('room-y').value=r.rect[1];editor.tint(selected,false);const source=house.placement_geometry.rooms.find(v=>v.id===selected);$('room-dimensions').textContent=source.name+' keeps its generator size, '+(source.rect[2]-source.rect[0]).toFixed(2)+' m by '+(source.rect[3]-source.rect[1]).toFixed(2)+' m. All doors must be at least 0.9 m wide. The server checks 0.9 m paths, 0.6 m person-to-wall clearance, 0.9 m furniture clearance and 1.2 m doorway clearance.';}
  function fillRooms(){for(const id of ['layout-room','swap-room']){$(id).textContent='';house.placement_geometry.rooms.forEach(r=>{const o=document.createElement('option');o.value=r.id;o.textContent=r.name;$(id).appendChild(o);});}selected=selected&&draft.some(r=>r.id===selected)?selected:draft[0].id;fields();}
  function preview(bad){editor.build(draftHouse(),true,false);editor.tint(selected,bad);}
  function simpleError(){const original=house.placement_geometry.rooms;for(const r of draft){const b=r.rect;const width=b[2]-b[0],depth=b[3]-b[1];let covered=0;for(const o of original){const a=o.rect;covered+=Math.max(0,Math.min(b[2],a[2])-Math.max(b[0],a[0]))*Math.max(0,Math.min(b[3],a[3])-Math.max(b[1],a[1]));}if(Math.abs(covered-width*depth)>.0001)return 'This room leaves the fixed house outline.';for(const t of draft){if(t.id!==r.id&&Math.min(b[2],t.rect[2])-Math.max(b[0],t.rect[0])>.0001&&Math.min(b[3],t.rect[3])-Math.max(b[1],t.rect[1])>.0001)return 'Rooms overlap and leave a gap. Drop on a matching block to swap, or undo.';}}return null;}
  async function validate(){const rev=++revision;pending=true;$('prepare-layout').disabled=true;window.RescueLayout.invalid();const local=simpleError();preview(!!local);message(local||'Checking every doorway, furniture footprint and clear path.',!!local);if(local){pending=false;return false;}
    try{const result=await window.RescueLayout.validate(draft);if(rev!==revision)return false;house=result.house;draft=clone(result.layout);lastValid=clone(result);editor.build(house,true,false);editor.tint(selected,false);window.RescueLayout.accept(result);$('prepare-layout').disabled=false;fields();message('Layout accepted. '+house.placement_geometry.doors.length+' clear doorways connect every room. '+result.placement.people.length+' people and a fire origin fit on checked clear floor.');return true;}
    catch(error){if(rev!==revision)return false;preview(true);message(error.message,true);return false;}finally{if(rev===revision)pending=false;}}
  function move(dx,dy,allowSwap){const room=draft.find(r=>r.id===selected),before=room.rect.slice();history.push(clone(draft));room.rect=[before[0]+dx,before[1]+dy,before[2]+dx,before[3]+dy];if(allowSwap){const cx=(room.rect[0]+room.rect[2])/2,cy=(room.rect[1]+room.rect[3])/2;const other=draft.find(r=>r.id!==selected&&cx>r.rect[0]&&cx<r.rect[2]&&cy>r.rect[1]&&cy<r.rect[3]);if(other&&Math.abs(other.rect[2]-other.rect[0]-before[2]+before[0])<.00001&&Math.abs(other.rect[3]-other.rect[1]-before[3]+before[1])<.00001){room.rect=other.rect.slice();other.rect=before;}}fields();return validate();}
  function wireDrag(){const canvas=editor.renderer.domElement;
    canvas.addEventListener('pointerdown',e=>{if(!moveMode||pending)return;const id=editor.roomAt(e),at=editor.point(e);if(!id||!at)return;selected=id;fields();drag={start:at,rooms:clone(draft),rect:draft.find(r=>r.id===id).rect.slice()};editor.orbit.enabled=false;canvas.setPointerCapture(e.pointerId);e.preventDefault();},true);
    canvas.addEventListener('pointermove',e=>{if(!drag)return;const p=editor.point(e);if(!p)return;const dx=Math.round((p[0]-drag.start[0])*2)/2,dy=Math.round((p[1]-drag.start[1])*2)/2;draft=clone(drag.rooms);draft.find(r=>r.id===selected).rect=drag.rect.map((v,i)=>v+(i%2?dy:dx));const bad=simpleError();preview(!!bad);$('room-x').value=draft.find(r=>r.id===selected).rect[0];$('room-y').value=draft.find(r=>r.id===selected).rect[1];message(bad||'Release to check doors and paths.',!!bad);$('prepare-layout').disabled=true;window.RescueLayout.invalid();});
    canvas.addEventListener('pointerup',e=>{if(!drag)return;const end=draft.find(r=>r.id===selected).rect,dx=end[0]-drag.rect[0],dy=end[1]-drag.rect[1];draft=drag.rooms;drag=null;editor.orbit.enabled=!moveMode;canvas.releasePointerCapture(e.pointerId);move(dx,dy,true);});
    canvas.addEventListener('pointercancel',()=>{if(!drag)return;draft=drag.rooms;drag=null;editor.orbit.enabled=!moveMode;validate();});
  }
  $('edit-layout').addEventListener('click',()=>{const selection=window.RescueLayout.selection();if(!selection.house)return;$('layout-editor').showModal();if(!ensureEditor())return;house=clone(selection.house);draft=clone(selection.layout||house.placement_geometry.rooms.map(r=>({id:r.id,rect:r.rect})));history=[];lastValid=null;editor.orbit.enabled=!moveMode;editor.build(house,true,true);fillRooms();validate();});
  $('layout-room').addEventListener('change',()=>{selected=$('layout-room').value;fields();});
  $('move-room').addEventListener('click',()=>{if(pending)return;const r=draft.find(r=>r.id===selected),x=+$('room-x').value,y=+$('room-y').value;move(x-r.rect[0],y-r.rect[1],false);});
  $('swap-rooms').addEventListener('click',()=>{if(pending)return;const a=draft.find(r=>r.id===selected),b=draft.find(r=>r.id===$('swap-room').value);if(a.id===b.id)return message('Choose a different room to swap.',true);history.push(clone(draft));[a.rect,b.rect]=[b.rect,a.rect];validate();});
  $('undo-layout').addEventListener('click',()=>{if(pending||!history.length)return;draft=history.pop();validate();});
  $('reset-layout').addEventListener('click',()=>{revision++;window.RescueLayout.reset();const selection=window.RescueLayout.selection();house=clone(selection.house);draft=house.placement_geometry.rooms.map(r=>({id:r.id,rect:r.rect}));history=[];editor.build(house,true,true);fillRooms();validate();});
  $('move-mode').addEventListener('click',()=>{moveMode=true;editor.orbit.enabled=false;$('move-mode').setAttribute('aria-pressed','true');$('orbit-mode').setAttribute('aria-pressed','false');});
  $('orbit-mode').addEventListener('click',()=>{moveMode=false;editor.orbit.enabled=true;$('move-mode').setAttribute('aria-pressed','false');$('orbit-mode').setAttribute('aria-pressed','true');});
  $('close-layout').addEventListener('click',()=>{$('layout-editor').close();});
  $('prepare-layout').addEventListener('click',()=>{if(pending||$('prepare-layout').disabled)return;$('layout-editor').close();window.RescueLayout.prepare();});
  function toggle(on){window.Rescue3D.replayEnabled=on;$('replay-3d').hidden=!on;$('replay').hidden=on;$('three-caption').hidden=!on;$('terra-caption').hidden=!on;$('view-plan').setAttribute('aria-pressed',String(!on));$('view-3d').setAttribute('aria-pressed',String(on));if(on){ensurePlayer();player&&player.resize();}}
  $('view-plan').addEventListener('click',()=>toggle(false));$('view-3d').addEventListener('click',()=>toggle(true));
  window.Rescue3D={replayEnabled:false,replay(h,f,m){if(!ensurePlayer())return;if(h!==replayHouse||m!==replayMission){player.build(h,false,true);replayHouse=h;replayMission=m;lastFrame=null;textures={};Object.entries(m.layers||{}).forEach(([id,cv])=>{const texture=new THREE.CanvasTexture(cv);texture.magFilter=THREE.NearestFilter;const mesh=new THREE.Mesh(new THREE.PlaneGeometry(m.gw*.1,m.gh*.1),new THREE.MeshBasicMaterial({map:texture,transparent:true,opacity:.5,depthWrite:false,side:THREE.DoubleSide}));mesh.rotation.x=-Math.PI/2;mesh.position.set(-25+(m.gx0+m.gw/2)*.1,.012,-(-25+(m.gy0+m.gh/2)*.1));player.search.add(mesh);textures[id]=texture;});}
    if(!f){player.resize();return;}if(lastFrame===f&&player.dynamic.children.length)return;lastFrame=f;dispose(player.dynamic);Object.values(textures).forEach(t=>t.needsUpdate=true);
    f.robots.forEach(r=>{const robot=new THREE.Group();robot.position.set(r.x,.14,-r.y);robot.rotation.y=(r.yaw||0)*Math.PI/180;const col=ROBOT[r.id]||0xffffff;const mesh=new THREE.Mesh(new THREE.CylinderGeometry(.24,.27,.22,20),new THREE.MeshLambertMaterial({color:col}));robot.add(mesh);box(robot,0,.17,0,.26,.10,.28,0x192327);for(const x of [-.25,.25])box(robot,x,-.02,0,.10,.18,.30,0x1d2125);box(robot,0,.09,-.24,.10,.08,.12,0xf2eee2);player.dynamic.add(robot);label(player.dynamic,r.id.replace('robot_',''),r.x,.8,-r.y,.65);});
    (m.found||[]).forEach(p=>{const mesh=new THREE.Mesh(new THREE.SphereGeometry(.18,12,8),new THREE.MeshLambertMaterial({color:0xf3f0dc}));mesh.position.set(p.x,.3,-p.y);player.dynamic.add(mesh);const ring=new THREE.Mesh(new THREE.TorusGeometry(.28,.045,6,20),new THREE.MeshBasicMaterial({color:ROBOT[p.by]||0xffffff}));ring.rotation.x=Math.PI/2;ring.position.set(p.x,.08,-p.y);player.dynamic.add(ring);});
    let fire=null;for(let i=m.lastIdx;i>=0&&!fire;i--)if(m.frames[i]&&Array.isArray(m.frames[i].fire))fire=m.frames[i].fire;(fire||[]).forEach(p=>{const flame=new THREE.Mesh(new THREE.ConeGeometry(Math.max(.10,p[2]),.7,10),new THREE.MeshBasicMaterial({color:0xff783a,transparent:true,opacity:.78}));flame.position.set(p[0],.35,-p[1]);player.dynamic.add(flame);});
    const event=(m.events||[]).filter(e=>e.t<=m.t&&(e.kind==='agent_decision'||e.kind==='agent_unavailable')).slice(-1)[0];$('terra-caption').textContent=event?event.text:'The brain keeps moving while Terra reads the robots\' reports.';
    Object.keys(ROBOT).forEach(id=>{const points=[];for(let i=0;i<=m.lastIdx;i++){const frame=m.frames[i];if(!frame)continue;const r=frame.robots.find(r=>r.id===id);if(r)points.push(new THREE.Vector3(r.x,.035,-r.y));}if(points.length>1)player.dynamic.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:ROBOT[id],transparent:true,opacity:.7})));});player.resize();
  }};
})();
