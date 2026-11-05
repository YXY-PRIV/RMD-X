"""K data actor. Fresh RGB material, fixed proposal geometry, training-only GT.

No old experiment writer is imported. No test annotation is opened.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import collections, concurrent.futures, csv, datetime, hashlib, json, math, multiprocessing, pathlib, resource, subprocess, sys, time
sys.dont_write_bytecode=True
import numpy as np
import PIL
from PIL import Image,features as PILfeatures
ROOT=pathlib.Path('/mnt/data/yxy/program/RAXO')
CODE=ROOT/'experiments/night_exploration_20261101/batch_k_multisource_quality'
NIGHT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101')
OUT=NIGHT/'batch_k_multisource_quality';CROOT=NIGHT/'batch_c_shared_union'
HROOT=NIGHT/'batch_h_multisource_material';JROOT=NIGHT/'batch_j_fresh_material_quality'
P16=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/P16_P25_revision_20261024/validation')
AUDIT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/batch01_20261023/data_audit')
sys.path.insert(0,str(ROOT))
from rmdx import material,preargmax_fusion as producer
CASES=['CLC_tune','DET','CLCXray_test','PIXray','PIDray','DvXray','Hi']
FEATURES=['confidence_percentile','clip_log_aspect_over4','width_over_image_width','height_over_image_height','centerx_over_width','centery_over_height','max_other_proposal_IoU','sqrt_relative_box_area','fresh_B_class','descending_B_rank_over_C_minus1','B_top_minus_second_over2','B_class_minus_top_over2','has_actual_material_support']
DS=dict(CLC_tune='CLCXray',DET='DET-COMPASS',CLCXray_test='CLCXray',PIXray='PIXray',PIDray='pidray',DvXray='DvXray',Hi='HiXray')
_CONFIG={}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def ref(p):
    p=pathlib.Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):
    p=pathlib.Path(r['path']);assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'],str(p);return p
def save(p,d):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)
def savez(p,**d):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    with p.with_name(p.name+'.tmp').open('wb')as f:np.savez_compressed(f,**d)
    p.with_name(p.name+'.tmp').replace(p)
def savearray(p,a):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    with p.open('xb')as f:np.save(f,a,allow_pickle=False)
def loadz(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def available():
    with open('/proc/meminfo')as f:return next(int(s.split()[1])*1024 for s in f if s.startswith('MemAvailable:'))
def protocol():
    p=CODE/'config/protocol.json';P=read(p);assert p.read_bytes()==(OUT/'protocol/protocol.json').read_bytes()
    assert P['status']=='FROZEN_BEFORE_ANY_K_SCIENTIFIC_RUN'
    return P
def verify_actor():
    f=read(OUT/'automation/data_code_freeze.json')
    for r in f['sources']:verify(r)
    assert f['protocol']==ref(CODE/'config/protocol.json')
    return f
def rgbhash(rgb):return hashlib.sha256(f'RGB:{rgb.shape[1]}x{rgb.shape[0]}:'.encode()+rgb.tobytes()).hexdigest()

def source_image_inventory():
    result={}
    for fn in ['image_fingerprints.csv','extended_image_fingerprints.csv']:
        with(AUDIT/fn).open(encoding='utf-8-sig')as f:
            for r in csv.DictReader(f):
                if r['role']!='evaluation':continue
                ds=r.get('dataset','DET-COMPASS');key=(ds,r['file_name'])
                if not r.get('file_sha256'):continue
                path=pathlib.Path(r['path'])
                if not path.exists()and str(path).startswith('/data/'):path=pathlib.Path('/mnt'+str(path))
                value=dict(path=str(path),file_name=r['file_name'],file_sha256=r['file_sha256'],decoded_rgb_sha256=r['decoded_rgb_sha256'],width=int(r['width']),height=int(r['height']))
                assert key not in result;result[key]=value
    with(P16/'data/roles.csv').open()as f:
        for r in csv.DictReader(f):
            with Image.open(r['path'])as im:width,height=im.size
            result[('CLC_'+r['role'],r['file_name'])]=dict(path=r['path'],file_name=r['file_name'],file_sha256=r['file_sha256'],decoded_rgb_sha256=r['decoded_rgb_sha256'],width=width,height=height,original_image_id=int(r['original_image_id']))
    return result

def bank_info(assets):
    for k in ['material_bank','category_meta']:verify(assets[k])
    categories=read(assets['category_meta']['path']);bank=read(assets['material_bank']['path'])
    byname={str(v['category_name']):v for v in bank['prototypes'].values()}
    counts=[]
    for cat in categories:
        v=byname.get(cat['category_name'])
        if v is None:counts.append(0);continue
        assert 'sample_count'in v,'Actual support count missing: '+cat['category_name']
        n=v['sample_count'];assert isinstance(n,(int,float))and n>=0 and int(n)==n
        counts.append(int(n))
    return dict(category_meta=assets['category_meta'],material_bank=assets['material_bank'],class_order=categories,actual_material_support_counts=counts)

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (CODE/'config/protocol.json').exists()
    inputs={}
    for case in CASES:
        fmref=ref(CROOT/'features'/case/'feature_manifest.json');fm=read(fmref['path']);assets=fm['identity']['assets']
        b=bank_info(assets)
        inputs[case]=dict(feature_manifest=fmref,source_index=ref(CROOT/'features'/case/'source_index.json'),
            source_trace=fm['outputs']['source_trace.npz'],candidate_valid=fm['outputs']['candidate_valid.npy'],candidate_category_id=fm['outputs']['candidate_category_id.npy'],
            candidate_row_key_sha256=fm['candidate_row_key_sha256'],row_key_sha256=fm['row_key_sha256'],shape=fm['shape'][:2],
            proposal_meta=assets['proposal_meta'],**b,
            source_native_F=fm['source_native_records']['F'],source_native_V=fm['source_native_records']['V'],
            base_quality=ref(NIGHT/'batch_e_affine_robustness'/'ensembles'/case/'visual_clean'/'all3/quality_scores.npz'),
            base_ensemble=ref(NIGHT/'batch_e_affine_robustness'/'ensembles'/case/'visual_clean'/'all3/ensemble.json'))
    cm=read('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_reliability_pilot_20261031/endpoint_assets/CLC/cal_fit_manifest.json')
    calfit=dict(asset_manifest=ref('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_reliability_pilot_20261031/endpoint_assets/CLC/cal_fit_manifest.json'),
        proposal_meta=cm['assets']['proposal_meta'],**bank_info(cm['assets']),
        allowed_GT=ref(P16/'data/cal_fit_coco.json'),image_split=ref('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_reliability_pilot_20261031/training/cal_fit_image_split.json'),
        C_teacher=ref(CROOT/'training/union_train_val.npz'))
    source_plan=ref(HROOT/'verification/source_candidate_plan.csv');snapshot=ref(HROOT/'verification/source_exclusion_snapshot.json')
    gd=read(P16/'protocols/upstream_protocol.json');modelinputs=read(P16/'protocols/model_source_inputs.json')
    gdsource=pathlib.Path('/mnt/data/yxy/external/GroundingDINO_clean')
    selected_gd_sources=sorted((gdsource/'groundingdino').rglob('*.py'))
    cfg=pathlib.Path(gd['detector']['config']);weight=pathlib.Path(gd['detector']['checkpoint'])
    BERT=read('/mnt/data/yxy-codex/home/outputs/RAXO/P16_P25_revision_20261024/models/bert_download_receipt.json')
    model_refs=[ref(cfg),ref(weight)]+[ref(p)for p in selected_gd_sources]+[ref(r['path'])for r in BERT['files']]
    with pathlib.Path(source_plan['path']).open()as f:planned=list(csv.DictReader(f))
    hi_anns=[ref(r['annotation_source'])for r in planned if r['dataset']=='HiXray']
    source_inputs=dict(CLC=calfit,PID=dict(**bank_info(read(inputs['PIDray']['feature_manifest']['path'])['identity']['assets']),allowed_GT=ref('/mnt/data/raxo/dataset_specific_audit/raw_unpacked/pidray/pidray/annotations/xray_train.json')),
        Hi=dict(**bank_info(read(inputs['Hi']['feature_manifest']['path'])['identity']['assets']),allowed_mapping=ref(ROOT/'outputs_rmdx/dataset_specific_material/hixray_mapping/hixray_user_confirmed_category_mapping.json'),actual_selected_train_annotations=hi_anns))
    names=FEATURES
    P=dict(study_id='K_multisource_fresh_material_quality',status='FROZEN_BEFORE_ANY_K_SCIENTIFIC_RUN',created_utc=now(),
       question='Matched extra officialtrain supervision: geometry-only vs geometry+freshRGB material quality, fixed existing nativeV/F proposals and frozen Evisual_clean base.',
       sources=dict(C_protocol=ref(ROOT/'experiments/night_exploration_20261101/batch_c_shared_union/config/protocol.json'),E_protocol=ref(ROOT/'experiments/night_exploration_20261101/batch_e_affine_robustness/config/protocol.json'),
          source_plan=source_plan,source_eligibility=snapshot,CLC_calfit=calfit,training_sources=source_inputs,
          image_catalogues={n:ref(AUDIT/n)for n in ['image_fingerprints.csv','extended_image_fingerprints.csv','rebuilt_bank_image_fingerprints.csv']},CLC_roles=ref(P16/'data/roles.csv'),
          J0_success=ref(JROOT/'completion_attempt02.json'),J0_prior_failure=ref(JROOT/'probe/failure.json'),H0_mismatch=ref(HROOT/'gate/attempt02/completion.json'),
          production_helpers={n:ref(ROOT/n)for n in ['rmdx/material.py','rmdx/preargmax_fusion.py']},GDINO_successful_prior=ref(P16/'protocols/upstream_protocol.json'),GDINO_source_refs=model_refs),
       cases=inputs,inference_cases=CASES,test_cases=CASES[1:],
       feature_schema=dict(names=names,dimensions=13,dtype='float32',normalization='Natural stated units only; no train/test fitted means or variances.',
          geometry_universe='Every original cached proposal for that source image before native V/F/UNC selection, then gather frozen C union. For new PID/Hi all retained class-agnostic detector proposals; never GT deletes.',
          confidence_percentile='Descending raw score, ties original global cache row/query stable;1-rank/(n-1), n1=1.',
          other_proposal_IoU='Float64 raw xywh pair IoU, self excluded; n1=0.',
          B='CurrentPIL RGB uint8, actual frozen bbox_only exact22; actual class-order lookup equal-category F64zstats/scalarcos thenF32, wholly fresh query features for train/CT/test.',
          material_rank='Descending freshB, stable class_index ties, rank/(C-1), C1=0.',
          missing='Actual sample_count0 or absent bank category: positions8..12 allzero. Generic other/global-prior is not positive support.',
          query_valid='First hardverify sourcefilename/path/fileSHA/header dimensions/originalbbox. If genuine legacybbox crop empty, retain row/geometry, query_validFalse andallM8..12zero; never disguise mappingfailure asmissing. Raw22 invalid isNaN plusmask, not a fabricated descriptor.',
          active_coordinates=dict(geometry=list(range(8)),material=list(range(13))),
          full='13 dimensions; has_actual_support can be learned unless unique-train constantguard.',geometry='8..12 zero.',padding='allzero invalid slot; same-class V/F branches share exactsame candidate slot/q.'),
       detector=dict(source_path=str(gdsource),config=str(cfg),checkpoint=str(weight),model_source_refs=model_refs,
          input='Sanitized image filename/path/fileSHA+header size only; no GTbbox in detector process.',
          captions=dict(PID='baton . pliers . hammer . powerbank . scissors . wrench . gun . bullet . sprayer . handcuffs . knife . lighter .',Hi='mobile phone . laptop . portable charger . tablet . cosmetic . water . nonmetallic lighter .'),
          prompt_semantics='One all-target-name caption per source, class-agnostic raw max token sigmoid, never phrase-to-category assignment or exact-phrase discard.',
          box_threshold=.10,text_threshold=0.,transform=gd['detector']['transform'],NMS=gd['detector']['NMS'],bbox=gd['detector']['bbox'],model_parameters='F32 eval/noAMP/noTF32; exact P16successful config/checkpoint/load and native transform.'),
       teacher=dict(candidate_population='Each actual proposal times every target class; no GTboxes as proposals.',
          target='Float64 maximum sameclass IoU; IoU>=.5 retained else0, finallyFloat32. CID0 valid. Ignore/crowd records explicitly counted; unexpected nonzero is failstop.',
          allowed_GT_only='Existing CLC calfit191 and selected official PID/Hi train annotations; never CT/six-test annotations.',
          C_teacher_gate='All valid2013train+452val shared tuples must exactmatch old C y uint32 by globalcacheindex+category_id.',
          source_split=dict(CLC=[153,38],PID=[307,77],Hi=[307,77]),empty_proposals='Reserved images retained in audit; effective images reported separately, no fake row.'),
       training=dict(variants=['geometry','material'],seeds=[17,29,43],architecture=[13,32,16,1],activation='ReLU',optimizer='AdamW',learning_rate=.001,weight_decay=1e-4,batch_size=1024,max_epochs=100,patience=15,min_delta=1e-6,
          source_names=['CLC','PID','Hi'],source_input=dict(path=str(OUT/'training/multisource_train_val.npz'),actual_ref_from=str(OUT/'training/data_receipt.json')),base_draws_per_epoch=29370,draws_per_source=9790,
          draws='Torch CPUgenerator(seed); source tuple weights1/(image_proposal_count*C), multinomialreplacement9790each source; concatenate then sharedgenerator randperm29370. Same seed variants replay same draws.',
          paired_views='Two identical clean views perbasebatch; mean theirBCE, oneupdate.29stepsperepoch, noaffine/dropout augmentation.',
          positive_weight='Per tuple C-1, only y*log(sigmoid) term; same extra labels/weights anddraws bothvariants.',
          early_stop='Val naturalweightedBCE: meanpersource of meanperimage of meanpertuple. Same weights C-1. Report same-balanced unweightedBCE/MSE separately.',
          constant_guard='Per variant onlyunique train transformed13 features; exactconstantcoordinates zero in all splits/inference. No empirical scaling or test fitting.',
          checkpoint='Bestval epoch; seed init deterministic; actualGPU2CUDA only; noV/backbone optimization.'),
       inference=dict(cases=CASES,variants=['geometry','material'],actual_GPU_runs=42,mean_outputs=14,ensemble='3seed float64mean thenfloat32, pad0, same-class shared slot q.'),
       mixing=dict(base='Frozen Evisual_clean3seed q',lambdas=[0.,.1,.25,.5],lambda0='Exact original base q bytes copied/reference, not roundtrip.',lambda_positive='Float64 clip(q,1e-6,1) then qbase**(1-lambda)*qhead**lambda, finallyFloat32.'),
       selection=dict(dataset='CLC_tune',criterion='Global lambda maximizing mean routerAP ofgeometry/material variants; ties smallerlambda; freeze before six-testAP.',new_test_parameter_selection=False),
       routing=dict(tau=0.,routes=['V','F','router','matched_priority'],same_as_C=True),scoring=dict(alpha=.5,gaussian_sigma=1.),
       final_routes=['V','F','router','matched_priority'],source_logical_AP=8,final_logical_AP=48,primary_variant='material',primary_route='router',
       resources=dict(GPU_index=2,GPU_UUID='GPU-cfd4e982-ea9c-6090-3f22-88ea28c7360d',GPU_single_worker=True,GPU3_do_not_touch=True,CPU_workers=8,threads_per_worker=1,min_MemAvailable_GiB=64,
          native_global_slots=4,native_lock_root=str(NIGHT/'resources/native_slots'),native_threads=8),
       versions=dict(python=sys.version,PIL=PIL.__version__,numpy=np.__version__,JPEG=PILfeatures.version_codec('jpg')),
       publication_edits=False,limits=['Additional PID/Hi officialtrain labels change supervision budget; full vs geometry matched. Not whollytrainingfree/sourceonlytransfer toalltargetdomains.',
          'NativeV/F membership/classes still inherit old visual/cache inputs, whose unknown encoders/physicalsource identities are not authenticated by this experiment.',
          'H0/J0 oldB differences remain preserved. All new aux material train/CT/test from same frozen currentproducer; no oldB used as headinput.',
          'File/RGB/name exclusions do not certify physicalspecimen/acquisitionsession isolation; five legacy source origins remain missing.',
          'Evisual_clean base selected as followup to prior Enegative results after known six-test exposure. Addedsame-domain officialtrainlabels, not all-domainzero-shot.',
          'No manuscript/figures/old numeric root modifications. Exploratory after prior six-test observations, not independent preregistration.'])
    save(CODE/'config/protocol.json',P);save(OUT/'protocol/protocol.json',P)
    paths=[CODE/'src/data_pipeline.py',CODE/'scripts/prepare_data.py',CODE/'scripts/detect_sources.py',CODE/'scripts/run_data.py']
    for p in paths:assert p.exists(),str(p)
    save(OUT/'automation/data_code_freeze.json',dict(protocol=ref(CODE/'config/protocol.json'),sources=[ref(p)for p in paths],scope='Dataactor only; root endpoint/modelcode have separate explicitlists, never recursive freeze.'))
    print('K_PROTOCOL_FROZEN',json.dumps(ref(CODE/'config/protocol.json')),flush=True)

def eligibility():
    P=protocol();verify_actor();planpath=verify(P['sources']['source_plan'])
    with planpath.open()as f:plan=list(csv.DictReader(f))
    images=[];seenfile=set();seenrgb=set();counts=collections.Counter()
    for index,r in enumerate(plan):
        p=pathlib.Path(r['image_path']);assert sha(p)==r['file_sha256']and p.stat().st_size==int(r['file_bytes'])
        with Image.open(p)as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
        pix=rgbhash(rgb);assert pix==r['decoded_rgb_sha256'];assert pix not in seenrgb and r['file_sha256']not in seenfile
        seenfile.add(r['file_sha256']);seenrgb.add(pix);source='PID'if r['dataset']=='pidray'else'Hi';role='train'if r['proposed_role']=='train_source'else'val'
        assert r['proposed_role']in ['train_source','heldout_source']
        row=dict(source=source,role=role,file_name=p.name,path=str(p),file_sha256=r['file_sha256'],decoded_rgb_sha256=pix,width=rgb.shape[1],height=rgb.shape[0],source_annotation=r['annotation_source'],image_key=source+'|'+p.name+'|'+r['file_sha256'])
        images.append(row);counts[(source,role)]+=1
    assert counts==collections.Counter({('PID','train'):307,('PID','val'):77,('Hi','train'):307,('Hi','val'):77})
    images=sorted(images,key=lambda r:(r['source'],r['file_name']))
    for index,r in enumerate(images):r['sanitized_image_id']=index
    save(OUT/'data/new_source_images.json',dict(status='READY_SANITIZED_NO_GT_BOXES',images=images,source_plan=P['sources']['source_plan'],protocol=ref(CODE/'config/protocol.json')))
    save(OUT/'verification/image_eligibility.json',dict(status='PASS768_ACTUAL_FILE_AND_RGB_SHA_CROSSSOURCE_UNIQUE',images=768,counts={str(k):v for k,v in counts.items()},old_exclusion_snapshot=P['sources']['source_eligibility'],scope='Exact existing exclusion evidence inherited withsameSHA; allchosen768 actualfile/RGB refreshed andcrosssourceunique. No unknownphysicalidentity claim.',protocol=ref(CODE/'config/protocol.json')))

def rows_iter(path):
    path=pathlib.Path(path)
    if path.suffix=='.json':yield from enumerate(read(path))
    else:
        with path.open()as f:
            for i,line in enumerate(f):yield i,json.loads(line)

def max_iou(a,b):
    a=np.asarray(a,np.float64).reshape(-1,4);b=np.asarray(b,np.float64).reshape(-1,4)
    if not len(b):return np.zeros((len(a),),np.float64)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:]);wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1];union=a[:,2,None]*a[:,3,None]+b[None,:,2]*b[None,:,3]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0).max(axis=1)
def geometry(rows,W,H):
    b=np.asarray([r['bbox']for r in rows],np.float64);score=np.asarray([r['detector_score']for r in rows],np.float64);n=len(rows);X=np.zeros((n,8),np.float64)
    rank=np.empty(n,np.int64);rank[np.lexsort((np.asarray([r['cache_index']for r in rows]),-score))]=np.arange(n)
    X[:,0]=1-rank/max(n-1,1)if n>1 else 1
    X[:,1]=np.clip(np.log(np.maximum(b[:,2],1e-12)/np.maximum(b[:,3],1e-12)),-4,4)/4
    X[:,2]=np.clip(b[:,2]/W,0,1);X[:,3]=np.clip(b[:,3]/H,0,1);X[:,4]=np.clip((b[:,0]+b[:,2]/2)/W,0,1);X[:,5]=np.clip((b[:,1]+b[:,3]/2)/H,0,1)
    if n>1:
        lo=np.maximum(b[:,None,:2],b[None,:,:2]);hi=np.minimum(b[:,None,:2]+b[:,None,2:],b[None,:,:2]+b[None,:,2:]);wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1];union=b[:,2,None]*b[:,3,None]+b[None,:,2]*b[None,:,3]-inter;ious=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);np.fill_diagonal(ious,0);X[:,6]=ious.max(axis=1)
    X[:,7]=np.sqrt(np.clip(b[:,2]*b[:,3]/(W*H),0,1));return X.astype(np.float32)

def configure(name,c):
    names=[r['category_name']for r in c['class_order']];lk=producer.build_category_material_matrix(names,names,read(c['material_bank']['path']));_CONFIG[name]=dict(c=c,lookup=lk)
def material_features(rgb,rows,lk):
    valid=[]
    for r in rows:
        x0,y0,x1,y1=material.xywh_to_int_bounds(r['bbox'],rgb.shape[1],rgb.shape[0]);valid.append(rgb[y0:y1,x0:x1].size>0)
    valid=np.asarray(valid,bool);f=np.full((len(rows),22),np.nan,np.float64);B=np.zeros((len(rows),len(lk['prototype_categories'])),np.float32)
    for i in np.flatnonzero(valid):
        raw=material.extract_colour_feature_from_array(rgb,rows[i]['bbox']);f[i]=raw
        z=material.zscore_feature(raw,lk['stats']);B[i]=np.asarray([0. if lk['category_features'].get(n)is None or n=='negative'else material.cosine_similarity(z,material.zscore_feature(lk['category_features'][n],lk['stats']))for n in lk['prototype_categories']],np.float32)
    return f,B,valid
def append_material(G,B,c,query_valid):
    n,C=B.shape;X=np.zeros((n,C,13),np.float32);X[:,:,:8]=G[:,None,:]
    rank=np.empty((n,C),np.int64);order=np.argsort(-B,axis=1,kind='stable');np.put_along_axis(rank,order,np.broadcast_to(np.arange(C),(n,C)),axis=1)
    top=B.max(axis=1).astype(np.float64);second=np.partition(B,-2,axis=1)[:,-2].astype(np.float64)if C>1 else top
    X[:,:,8]=B;X[:,:,9]=rank/max(C-1,1);X[:,:,10]=((top-second)/2)[:,None];X[:,:,11]=(B.astype(np.float64)-top[:,None])/2;X[:,:,12]=(np.asarray(c['actual_material_support_counts'])>0)[None,:]
    X[:,np.asarray(c['actual_material_support_counts'])==0,8:]=0;X[~query_valid,:,8:]=0
    assert np.isfinite(X).all();return X

def process_image(task):
    name,im,allrows,selected,training=task;assert available()>64*1024**3
    t=time.perf_counter();cfg=_CONFIG[name];c=cfg['c'];assert sha(im['path'])==im['file_sha256'],im['path']
    with Image.open(im['path'])as pic:rgb=np.asarray(pic.convert('RGB'),np.uint8)
    assert rgb.shape==(im['height'],im['width'],3)and rgbhash(rgb)==im['decoded_rgb_sha256'],im['path']
    for r in allrows:
        b=np.asarray(r['bbox'],np.float64);assert np.isfinite(b).all()and b[2]>0 and b[3]>0
        if 'width'in r:assert r['width']==im['width']and r['height']==im['height'],im['path']
    rows=[allrows[i]for i in selected];G=geometry(allrows,im['width'],im['height'])[selected];f,B,query_valid=material_features(rgb,rows,cfg['lookup']);X=append_material(G,B,c,query_valid)
    y=None
    if training:
        y=np.zeros(B.shape,np.float32)
        for j,cat in enumerate(c['class_order']):
            boxes=[a['bbox']for a in im['GT']if a['category_id']==cat['category_id']];v=max_iou([r['bbox']for r in rows],boxes);y[:,j]=np.where(v>=.5,v,0).astype(np.float32)
    return dict(name=name,image_key=im['image_key'],file_name=im['file_name'],indices=[r['cache_index']for r in rows],X=X,B=B,exact22=f,query_valid=query_valid,y=y,elapsed=time.perf_counter()-t,RGB_sha256=rgbhash(rgb),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

def parallel(tasks):
    assert available()>64*1024**3
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('fork'))as pool:
        for r in pool.map(process_image,tasks,chunksize=1):yield r

def load_training_images(P):
    inventory=source_image_inventory();new=read(OUT/'data/new_source_images.json')['images'];by_source={s:[]for s in ['CLC','PID','Hi']};sourcecfg=P['sources']['training_sources']
    clc=read(verify(sourcecfg['CLC']['allowed_GT']));split=read(verify(sourcecfg['CLC']['image_split']));trainids=set(split['train_original_image_ids']);valids=set(split['val_original_image_ids']);annmap=collections.defaultdict(list)
    for a in clc['annotations']:
        assert not a.get('iscrowd',0)and not a.get('ignore',0);annmap[a['image_id']].append(dict(category_id=int(a['category_id']),bbox=a['bbox']))
    for im in clc['images']:
        original=int(im['original_image_id']);v=dict(inventory[('CLC_cal_fit',im['file_name'])]);assert original==v['original_image_id']
        v.update(source='CLC',role='train'if original in trainids else'val',image_key='CLC|'+v['file_name']+'|'+v['file_sha256'],GT=annmap[im['id']]);assert original in trainids|valids;by_source['CLC'].append(v)
    pid=read(verify(sourcecfg['PID']['allowed_GT']));pidnames={i['id']:i['file_name']for i in pid['images']};pidheaders={i['file_name']:i for i in pid['images']};pid_selected={im['file_name']for im in new if im['source']=='PID'};pann=collections.defaultdict(list);targetPID={r['category_id']for r in sourcecfg['PID']['class_order']};extra=collections.Counter()
    for a in pid['annotations']:
        if pidnames[a['image_id']]not in pid_selected:continue
        if a['category_id']not in targetPID:extra[str(a['category_id'])]+=1;continue
        assert not a.get('iscrowd',0)and not a.get('ignore',0);pann[pidnames[a['image_id']]].append(dict(category_id=int(a['category_id']),bbox=a['bbox']))
    mapping=read(verify(sourcecfg['Hi']['allowed_mapping']))['mapping'];Hi_targets={r['category_id']for r in sourcecfg['Hi']['class_order']}
    for item in sourcecfg['Hi']['actual_selected_train_annotations']:verify(item)
    for row in new:
        im=dict(row)
        if row['source']=='PID':
            header=pidheaders[row['file_name']];assert header['width']==im['width']and header['height']==im['height'];im['GT']=pann[row['file_name']]
        else:
            parsed=[]
            for line in pathlib.Path(row['source_annotation']).read_text().splitlines():
                parts=line.split();assert len(parts)==6 and parts[0]==row['file_name'],line
                name=parts[1]
                if name not in mapping:extra['Hi:'+name]+=1;continue
                cid=int(mapping[name]['fulltest_category_id'])
                if cid not in Hi_targets:extra['Hi:'+name]+=1;continue
                x0,y0,x1,y1=map(float,parts[2:]);assert x1>x0 and y1>y0;parsed.append(dict(category_id=cid,bbox=[x0,y0,x1-x0,y1-y0]))
            im['GT']=parsed
        by_source[row['source']].append(im)
    save(OUT/'verification/allowed_source_GT_load.json',dict(status='PASS_ALLOWED_TRAINING_GT_ONLY',extra_GT_categories=dict(extra),crowd_ignore_records=0,test_GT_opened=False,source_inputs={s:sourcecfg[s]for s in ['CLC','PID','Hi']}))
    return by_source

def training():
    P=protocol();verify_actor();by_source=load_training_images(P);sourcecfg=P['sources']['training_sources'];alloutputs={};summaries=[];meta_identity=[];teacher_index={};math_gates=[]
    for sid,name in enumerate(['CLC','PID','Hi']):
        c=sourcecfg[name];configure(name,c);images=sorted(by_source[name],key=lambda r:r['file_name']);C=len(c['class_order']);cid=[r['category_id']for r in c['class_order']]
        if name=='CLC':
            wanted={im['file_name']for im in images};groups=collections.defaultdict(list)
            for idx,row in rows_iter(verify(c['proposal_meta'])):
                if row['file_name']in wanted:
                    assert row.get('cache_index',idx)==idx;row=dict(row,cache_index=idx);groups[row['file_name']].append(row)
        else:
            result=read(OUT/'data/detector_proposals.json');groups=collections.defaultdict(list)
            for row in result['rows']:
                if row['source']==name:groups[row['file_name']].append(row)
        tasks=[];image_meta={};offset=0
        for im in images:
            im['training_image_index']=len(meta_identity);meta_identity.append(dict(source=name,role=im['role'],file_name=im['file_name'],image_key=im['image_key'],index=im['training_image_index']))
            rows=groups[im['file_name']];image_meta[im['image_key']]=im
            if rows:tasks.append((name,im,rows,list(range(len(rows))),True))
        probe=tasks[0];probe_im=probe[1];probe_rows=probe[2][:4 if name=='CLC'else 2]
        with Image.open(probe_im['path'])as pic:probe_rgb=np.asarray(pic.convert('RGB'),np.uint8)
        probe_f,probe_B,probe_valid=material_features(probe_rgb,probe_rows,_CONFIG[name]['lookup']);assert probe_valid.all()
        reference_f=np.asarray(producer._proposal_features(probe_rows,pathlib.Path(probe_im['path']).parent),np.float64)
        reference_B=producer.compute_material_similarity_for_proposals(probe_rows,pathlib.Path(probe_im['path']).parent,_CONFIG[name]['lookup'],device='cpu').numpy()
        assert np.array_equal(probe_f.view(np.uint64),reference_f.view(np.uint64))and np.array_equal(probe_B.view(np.uint32),reference_B.view(np.uint32))
        math_gates.append(dict(source=name,rows=len(probe_rows),exact22_bitdifferences=0,B_bitdifferences=0,file_sha256=probe_im['file_sha256'],global_cache_indices=[r['cache_index']for r in probe_rows]))
        data={role:collections.defaultdict(list)for role in ['train','val']};positives=collections.Counter();prop_counts=collections.Counter();effective=collections.Counter();invalid=collections.Counter()
        for r in parallel(tasks):
            im=image_meta[r['image_key']];role=im['role'];n=len(r['indices']);X=r['X'].reshape(-1,13);Y=r['y'].reshape(-1);dest=data[role]
            for k,v in dict(X=X,y=Y,source_index=np.full(n*C,sid,np.int64),image_index=np.full(n*C,im['training_image_index'],np.int64),class_count=np.full(n*C,C,np.int64),category_id=np.tile(np.asarray(cid,np.int64),n),global_cache_index=np.repeat(np.asarray(r['indices'],np.int64),C),query_valid=np.repeat(r['query_valid'],C)).items():dest[k].append(v)
            prop_counts[role]+=n;effective[role]+=1;positives[role]+=int(np.count_nonzero(Y>0));invalid[role]+=int(np.count_nonzero(~r['query_valid']))
            if name=='CLC':
                for i,g in enumerate(r['indices']):
                    for j,cat in enumerate(cid):teacher_index[(g,cat)]=r['y'][i,j]
        for role,d in data.items():
            for k,v in d.items():alloutputs.setdefault(k+'_'+role,[]).append(np.concatenate(v))
        summary=dict(source=name,class_ids=cid,target_class_count=C,reserved_images={r:sum(im['role']==r for im in images)for r in ['train','val']},effective_images=dict(effective),proposal_counts=dict(prop_counts),allclass_tuple_counts={r:prop_counts[r]*C for r in ['train','val']},positive_tuple_counts=dict(positives),query_invalid_empty_ROIs=dict(invalid));summaries.append(summary)
        print(now(),'K_TRAIN_SOURCE_READY',json.dumps(summary),flush=True)
    old=loadz(verify(sourcecfg['CLC']['C_teacher']));checks={}
    for role,want in [('train',2013),('val',452)]:
        found=0
        for i,g in enumerate(old['global_cache_index_'+role]):
            for j,valid in enumerate(old['candidate_valid_'+role][i]):
                if not valid:continue
                key=(int(g),int(old['candidate_category_id_'+role][i,j]));actual=np.float32(teacher_index[key]);expected=np.float32(old['y_'+role][i,j]);assert actual.view(np.uint32)==expected.view(np.uint32),(key,float(actual),float(expected));found+=1
        assert found==want;checks[role]=dict(shared_valid_tuples=found,teacher_uint32_differences=0)
    assert sum(r['rows']for r in math_gates)==8;save(OUT/'verification/fresh_material_current_helper_math.json',dict(status='PASS_CURRENT_GROUPED_VS_ORIGINAL8',gates=math_gates,scope='Current declared producer equivalence only; not legacyB identity.'))
    output={k:np.concatenate(v)for k,v in alloutputs.items()};meta=dict(protocol_sha256=sha(CODE/'config/protocol.json'),feature_names=FEATURES,source_names=['CLC','PID','Hi'],source_summaries=summaries,image_identities=meta_identity,teacher_C_exact=checks)
    output['metadata_json']=np.asarray(json.dumps(meta));output['feature_names']=np.asarray(FEATURES)
    dest=OUT/'training/multisource_train_val.npz';savez(dest,**output);save(OUT/'training/data_receipt.json',dict(status='COMPLETE',protocol=ref(CODE/'config/protocol.json'),training_npz=ref(dest),source_summaries=summaries,C_teacher_exact_gate=checks,feature_names=FEATURES,sources=sourcecfg))
    save(OUT/'training/status.json',dict(status='COMPLETE',training_npz=ref(dest),data_receipt=ref(OUT/'training/data_receipt.json'),protocol=ref(CODE/'config/protocol.json')))

def inference_case(case):
    P=protocol();verify_actor();c=P['cases'][case];fm=read(verify(c['feature_manifest']));t=loadz(verify(c['source_trace']));valid=np.load(verify(c['candidate_valid']),allow_pickle=False);cats=np.load(verify(c['candidate_category_id']),allow_pickle=False);configure(case,c)
    inventory=source_image_inventory();ds='CLC_tune'if case=='CLC_tune'else DS[case];groups=collections.defaultdict(list);needed=set(map(int,t['global_cache_index']));seen=0
    for idx,row in rows_iter(verify(c['proposal_meta'])):
        row=dict(row,cache_index=idx);groups[row['file_name']].append(row)
    trace_lookup={int(g):i for i,g in enumerate(t['global_cache_index'])};tasks=[];image_refs=[];C=len(c['class_order']);cid_to_j={r['category_id']:j for j,r in enumerate(c['class_order'])}
    for fname,rows in groups.items():
        sel=[i for i,r in enumerate(rows)if r['cache_index']in needed]
        if not sel:continue
        im=dict(inventory[(ds,fname)]);im['image_key']=case+'|'+fname+'|'+im['file_sha256']
        for i in sel:
            r=rows[i];j=trace_lookup[r['cache_index']];assert int(r['image_id'])==int(t['original_image_id'][j])and np.array_equal(np.asarray(r['bbox'],np.float64),t['bbox'][j]);seen+=1
        tasks.append((case,im,rows,sel,False));image_refs.append(im)
    N=len(t['global_cache_index']);assert seen==N
    X=np.zeros((N,2,13),np.float32);B=np.zeros((N,C),np.float32);F=np.full((N,22),np.nan,np.float64);QV=np.zeros(N,bool);assigned=np.zeros(N,bool);receipts=[]
    for r in parallel(tasks):
        target=np.asarray([trace_lookup[g]for g in r['indices']],np.int64);assert not assigned[target].any();assigned[target]=True;B[target]=r['B'];F[target]=r['exact22'];QV[target]=r['query_valid']
        for slot in range(2):
            good=valid[target,slot];rr=target[good];jj=np.asarray([cid_to_j[int(cid)]for cid in cats[rr,slot]],np.int64);X[rr,slot]=r['X'][np.flatnonzero(good),jj]
        receipts.append({k:v for k,v in r.items()if k not in ['X','B','exact22','y','query_valid']})
    assert assigned.all()and np.all(X[~valid]==0)and np.isfinite(X).all()
    folder=OUT/'features'/case;folder.mkdir(parents=True,exist_ok=False)
    for n,a in [('candidate_features.npy',X),('fresh_material_B.npy',B),('fresh_exact22.npy',F),('query_valid.npy',QV)]:savearray(folder/n,a)
    save(folder/'source_image_receipts.json',dict(images=image_refs,processing=receipts,source_proposal_meta=c['proposal_meta'],scope='Everyoriginalproposalforgeometry, onlynativeunionRGB22 material extracted, noGT.'))
    outputs={p.name:ref(p)for p in folder.iterdir()};outputs['candidate_valid.npy']=c['candidate_valid'];outputs['candidate_category_id.npy']=c['candidate_category_id']
    manifest=dict(status='COMPLETE_FRESH_K_AUXILIARY_FEATURES',case=case,protocol=ref(CODE/'config/protocol.json'),protocol_sha256=sha(CODE/'config/protocol.json'),source_C_feature_manifest=c['feature_manifest'],source_C_rowkey=c['candidate_row_key_sha256'],candidate_row_key_sha256=c['candidate_row_key_sha256'],shape=list(X.shape),dtype='float32',feature_names=FEATURES,outputs=outputs,actual_fresh_producer=P['sources']['production_helpers'],actual_bank=c['material_bank'],class_order=c['class_order'],actual_material_support_counts=c['actual_material_support_counts'],base_visual_clean_ensemble=c['base_ensemble'],base_quality=c['base_quality'],query_invalid_empty_ROIs=int(np.count_nonzero(~QV)),GT_annotations_opened=False,old_B_used_for_auxiliary_input=False)
    save(folder/'feature_manifest.json',manifest);save(folder/'status.json',dict(status='COMPLETE',feature_manifest=ref(folder/'feature_manifest.json'),protocol=ref(CODE/'config/protocol.json')))
    print(now(),'K_CASE_FRESH_READY',case,N,flush=True)
