"""MDGOD实验入口；在下方修改路径、参数及并行数。"""
from pathlib import Path
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(_name,'1')
import argparse
from datetime import datetime
import hashlib
import itertools
import json
import pickle
import shutil
import sys
import time
import uuid
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits
from OD.MDGOD import PreparedMDGOD,warmup


# ==================== 可修改接口：通常只改这里 ====================
BASE_DIR=Path(__file__).resolve().parent
ALGORITHM='MDGOD'
DATA_DIR=BASE_DIR/'Datasets'  # 输入文件夹；自定义示例：Path(r'D:\data\mydata')
RESULT_DIR=BASE_DIR/'Result'  # 输出根目录；自定义示例：Path(r'D:\results')，实际存入其MDGOD子文件夹
DATASETS=None  # None：跑输入文件夹全部MAT；只跑指定数据集示例：['wine', 'hepatitis_2_9_variant1']
WORKERS=2  # 并行进程数；1为串行
THREADS=1  # 每个进程的数值计算线程数
SEEDS=list(range(10))  # 随机种子：0～9，共10次
SEARCH_MODE='sequential'  # sequential：先固定α=1选σ，再固定σ选α；grid：遍历所有组合
SIGMAS=[i/20 for i in range(21)]  # σ搜索范围：0～1，步长0.05
ALPHAS=[i/20 for i in range(21)]  # α搜索范围：0～1，步长0.05
LAYERS=5  # 尺度数
NOISE_STRENGTH=.05  # 内部噪声强度η：0.05即5%
INLINE_MB=16  # 得分总量超过此MiB阈值时改存外部NPY；否则存PKL的OF字段
EXTERNAL_SCORES=False  # True：无论大小都使用外部NPY，路径记录在PKL中
BLOCK_ROWS=256  # 保留兼容接口；当前近似粒球不使用此值
STATE_DIR=BASE_DIR/'.mdgod_shared_runs'  # 断点目录，也可改为Path(r'D:\temp\mdgod')
# 双击/直接运行使用以上设置；命令行同名选项优先。MAT变量trandata最后一列须为0/1标签。
# 每组(σ,α)仅汇报一次重复均值；每个数据集结束后额外汇报分别选优的AUC、AP及其参数。PKL保留全部原始结果。
# ==================== 以下为实验实现 ====================
SCHEMA_VERSION='MDGOD_results_v1'

PREPROCESSING = 'drop raw columns whose every value belongs to {0,1}; then per-column MinMaxScaler[0,1]; no clipping after internal noise'

def preprocess(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or len(raw) < 2 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError('MAT需包含有限二维trandata，最后一列为0/1标签')
    y = raw[:, -1].copy()
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError('标签须同时包含0与1，1为异常')
    features = raw[:, :-1]
    remove = np.all((features == 0) | (features == 1), axis=0)
    binary_fallback = bool(np.all(remove))
    if np.all(remove):
        remove = np.zeros(features.shape[1], dtype=bool)
    X = MinMaxScaler().fit_transform(features[:, ~remove])
    metadata = dict(Preprocessing=PREPROCESSING, RawFeatures=features.shape[1],
                    UsedFeatures=X.shape[1], RemovedRawBinaryColumns=np.flatnonzero(remove).tolist(),
                    KeptRawColumns=np.flatnonzero(~remove).tolist(), ColumnIndices='zero-based',
                    LabelPolicy='last column; 0 normal / 1 anomaly; original row order',
                    PreprocessingSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if binary_fallback:
        metadata['Preprocessing'] = 'all raw features retained because every feature is binary; per-column MinMaxScaler[0,1]'
        metadata['AllBinaryFallbackApplied'] = True
    return X, y, metadata

def load_dataset(path):
    raw = loadmat(path).get('trandata')
    if raw is None:
        raise ValueError('MAT缺少trandata变量')
    return preprocess(raw)

def select_files(data_dir, datasets=None, all_datasets=False):
    directory = Path(data_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if isinstance(datasets, str):
        datasets = [datasets]

    files = ([directory / (name if name.lower().endswith('.mat') else name+'.mat') for name in datasets]
             if datasets else list(directory.glob('*.mat')))
    missing = [str(p) for p in files if not p.is_file()]
    if not files or missing:
        raise FileNotFoundError('没有数据集或文件缺失：'+str(missing))
    return sorted(set(files), key=lambda p: (p.stat().st_size, p.name))

def normalized_config(config):


    if not isinstance(config,dict): return {}
    value=json.loads(json.dumps(config))
    for key in ['RunnerSHA256','EntrySHA256','MainSHA256']:
        value.pop(key,None)
    for key in ['Input','FeatureSelection']:
        if isinstance(value.get(key),dict): value[key].pop('PreprocessingSHA256',None)
    return value


def same_config(previous,current):
    return normalized_config(previous)==normalized_config(current)


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def atomic_npz(path,**arrays):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temporary,path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path):
    features,labels,_=load_dataset(path)
    return features,labels


def signature(path,args):
    import scipy,sklearn,numba
    input_metadata=load_dataset(path)[2]
    return dict(Algorithm=ALGORITHM,SchemaVersion=SCHEMA_VERSION,Engine='MDGOD_approximate_v1',Approximate=True,
        BallGeneration='approximate_centroid_anchor',Reuse='per seed preparation and per sigma spectral matrix',
        BallGeneratorSHA256=digest(BASE_DIR/'OD/GB.py'),EntrySHA256=digest(Path(__file__)),Source=str(path.resolve()),
        SourceSHA256=digest(path),AlgorithmSHA256=digest(BASE_DIR/'OD/MDGOD.py'),
        MainSHA256=digest(Path(__file__)),Preprocessing=input_metadata['Preprocessing'],
        FeatureSelection=input_metadata,
        Search=args.search,Sigmas=args.sigmas,Alphas=args.alphas,Seeds=args.seeds,
        Layers=args.layers,NoiseStrength=args.eta,BlockRows=args.block_rows,
        TimingMode='reconstructed_independent',ThreadLimit=args.threads,
        Workers=getattr(args,'workers',1),ProcessStartMethod='spawn',
        Spectrum='deterministic LAPACK eigh/evr; lowest min(n-1,50) values for n>10',
        Score='weighted mean of layer-wise lower/upper ratios, as in legacy code',
        Versions=dict(Python=sys.version.split()[0],NumPy=np.__version__,SciPy=scipy.__version__,Sklearn=sklearn.__version__,Numba=numba.__version__))


def computation_signature(config):


    ignored={'EntrySHA256','MainSHA256','Workers','ProcessStartMethod'}
    return normalized_config({k:v for k,v in config.items() if k not in ignored})


def compatible_config(previous,current):
    return computation_signature(previous)==computation_signature(current)


def checkpoint_folder(path,config):
    matching=[]
    for candidate in STATE_DIR.glob(path.stem+'_*/config.json'):
        try:
            old=json.loads(candidate.read_text(encoding='utf-8'))
            if compatible_config(old,config):
                matching.append((len(list(candidate.parent.glob('param_*_seed_*.npz'))),candidate.parent))
        except (OSError,ValueError,TypeError):pass
    if matching:
        return sorted(matching,key=lambda entry:(-entry[0],str(entry[1])))[0][1]
    key=hashlib.sha256(json.dumps(computation_signature(config),sort_keys=True).encode()).hexdigest()[:20]
    return STATE_DIR/(path.stem+'_'+key)


def read_checkpoint(checkpoint,pair,seed):
    with np.load(checkpoint,allow_pickle=False) as saved:
        if tuple(saved['parameter'])!=tuple(pair) or int(saved['seed'])!=seed:
            raise ValueError('断点参数或种子不匹配：'+str(checkpoint))
        a=float(saved['auc']);p=float(saved['ap']);elapsed=float(saved['seconds']);cost=saved['components'].tolist()
        if (not np.isfinite([a,p,elapsed,*cost]).all() or not -1e-12<=a<=1+1e-12 or not -1e-12<=p<=1+1e-12
                or elapsed<=0 or not np.isclose(sum(cost),elapsed,rtol=0,atol=1e-8)):
            raise ValueError('断点指标或时间无效：'+str(checkpoint))
        workers=int(saved['worker_limit']) if 'worker_limit' in saved else 1
        mode=str(saved['execution_mode']) if 'execution_mode' in saved else 'legacy_serial'
    return a,p,elapsed,cost,workers,mode


def seed_stage(X,y,pairs,seed,args,folder,offset):


    folder=Path(folder)
    records=[]
    needed=[i for i,pair in enumerate(pairs) if not (folder/f'param_{offset+i:04d}_seed_{seed:03d}.npz').exists()]
    if not needed:
        return [read_checkpoint(folder/f'param_{offset+i:04d}_seed_{seed:03d}.npz',pair,seed) for i,pair in enumerate(pairs)]
    model_file=folder/f'prepared_seed_{seed:03d}.pkl'
    with threadpool_limits(limits=args.threads):
        if model_file.exists():
            with model_file.open('rb') as f:model,preparation=pickle.load(f)
        else:
            start=time.perf_counter()
            model=PreparedMDGOD(X,seed=seed,layers=args.layers,eta=args.eta,block_rows=args.block_rows)
            preparation=time.perf_counter()-start
            temporary=model_file.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            with temporary.open('wb') as f:pickle.dump((model,preparation),f,protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary,model_file)
        spectral_cost={}
        for i,pair in enumerate(pairs):
            checkpoint=folder/f'param_{offset+i:04d}_seed_{seed:03d}.npz'
            if checkpoint.exists():
                records.append(read_checkpoint(checkpoint,pair,seed));continue
            sigma,alpha=pair
            if sigma not in spectral_cost:
                model.spectra.clear()
                start=time.perf_counter();model.prepare_sigma(sigma)
                spectral_cost={sigma:time.perf_counter()-start}
            start=time.perf_counter();scores=model.score(sigma,alpha)
            scoring=time.perf_counter()-start
            if scores.shape!=y.shape or not np.isfinite(scores).all():raise ValueError('异常分数维度或数值无效')
            cost=[preparation,spectral_cost[sigma],scoring];elapsed=sum(cost)
            a=float(roc_auc_score(y,scores));p=float(average_precision_score(y,scores))
            mode='shared_reconstructed_parallel' if args.workers>1 else 'shared_reconstructed_serial'
            atomic_npz(checkpoint,parameter=np.array(pair),seed=np.array(seed),scores=scores,
                auc=np.array(a),ap=np.array(p),seconds=np.array(elapsed),components=np.array(cost),
                worker_limit=np.array(args.workers),execution_mode=np.array(mode),pid=np.array(os.getpid()))
            records.append((a,p,elapsed,cost,args.workers,mode))
    return records


_WORKER_X=None
_WORKER_Y=None
_WORKER_ARGS=None


def worker_initialize(input_path,label_path,args):
    global _WORKER_X,_WORKER_Y,_WORKER_ARGS

    _WORKER_X=np.load(input_path,mmap_mode='r',allow_pickle=False)
    _WORKER_Y=np.load(label_path,mmap_mode='r',allow_pickle=False)
    _WORKER_ARGS=args
    with threadpool_limits(limits=args.threads):warmup()


def worker_stage(pairs,seed,folder,offset):
    return seed_stage(_WORKER_X,_WORKER_Y,pairs,seed,_WORKER_ARGS,folder,offset)


def parameter_runs(X,y,pairs,args,folder,offset,progress,executor=None,reported=None):
    shape=(len(pairs),len(args.seeds))
    arrays=[np.empty(shape) for _ in range(3)]
    components=np.empty(shape+(3,))
    def accept(j,records):
        for i,record in enumerate(records):
            for target,value in zip(arrays,record[:3]):target[i,j]=value
            components[i,j]=record[3]
        progress.update(last_completed_seed=args.seeds[j],stage_parameters=len(pairs))
        atomic_json(STATE_DIR/'status.json',progress)
    if executor is None:
        for j,seed in enumerate(args.seeds):accept(j,seed_stage(X,y,pairs,seed,args,folder,offset))
    else:
        futures={executor.submit(worker_stage,pairs,seed,str(folder),offset):j for j,seed in enumerate(args.seeds)}
        for future in as_completed(futures):accept(futures[future],future.result())
    if reported is None:reported=set()
    for i,pair in enumerate(pairs):
        key=tuple(pair)
        if key in reported:continue
        print(f'{progress["dataset"]} sigma={pair[0]:g} alpha={pair[1]:g} '
              f'n={len(args.seeds)} AUC={arrays[0][i].mean():.6f} '
              f'AP={arrays[1][i].mean():.6f} Time={arrays[2][i].mean():.6f}s',flush=True)
        reported.add(key)
    return (*[a.tolist() for a in arrays],components.tolist())


def report_best(dataset,pairs,auc,ap):
    auc_means=np.asarray(auc,dtype=float).mean(axis=1)
    ap_means=np.asarray(ap,dtype=float).mean(axis=1)
    auc_index=int(np.argmax(auc_means));ap_index=int(np.argmax(ap_means))
    auc_pair=pairs[auc_index];ap_pair=pairs[ap_index]
    print(f'{dataset} | Best AUC={auc_means[auc_index]:.6f} '
          f'(sigma={auc_pair[0]:g}, alpha={auc_pair[1]:g}) | '
          f'Best AP={ap_means[ap_index]:.6f} '
          f'(sigma={ap_pair[0]:g}, alpha={ap_pair[1]:g})',flush=True)


def archive_existing(destination):

    if not destination.exists():return
    archive=destination.parents[2]/'historical_results'/ALGORITHM/digest(destination)[:16]
    archive.mkdir(parents=True,exist_ok=True)
    for source in [destination,destination.with_suffix('.xlsx')]:
        if source.exists() and not (archive/source.name).exists():shutil.copy2(source,archive/source.name)


def publish(path,config,pairs,metrics,folder,args,y,batch_elapsed,startup_seconds):
    directory=args.result_dir/ALGORITHM;directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(path.stem+'.pkl')
    arrays=[np.asarray(a,dtype=np.float64) for a in metrics[:3]]
    shape=(len(pairs),len(args.seeds))
    if any(a.shape!=shape or not np.isfinite(a).all() for a in arrays):raise ValueError('结果矩阵无效')
    components=np.asarray(metrics[3],dtype=np.float64)
    if not np.allclose(components.sum(axis=2),arrays[2],rtol=0,atol=1e-8):raise ValueError('完整时间未包含全部算法阶段')
    provenance=[[read_checkpoint(folder/f'param_{i:04d}_seed_{seed:03d}.npz',pair,seed)[4:]
                 for seed in args.seeds] for i,pair in enumerate(pairs)]
    worker_limits=np.array([[entry[0] for entry in row] for row in provenance],dtype=int)
    run_modes=[[entry[1] for entry in row] for row in provenance]
    has_parallel=bool(np.any(worker_limits>1))
    has_serial=bool(np.any(worker_limits==1))
    environment=('mixed_serial_and_parallel_contention' if has_parallel and has_serial else
                 'parallel_contention' if has_parallel else 'serial')
    execution=('mixed_serial_and_process_parallel_spawn' if has_parallel and has_serial else
               'process_parallel_spawn' if has_parallel else 'serial')
    result=dict(Algorithm=ALGORITHM,Par=[tuple(p) for p in pairs],Seeds=list(args.seeds),
        AUC=arrays[0],AP=arrays[1],Time=arrays[2],TimingMode='reconstructed_independent',SchemaVersion=SCHEMA_VERSION,
        Labels=y,Config=config,TimingComponents=dict(Preparation=components[:,:,0],Spectral=components[:,:,1],Scoring=components[:,:,2]),
        ExecutionMode='shared_reconstructed_complete_cost; '+execution,WorkerCount=getattr(args,'workers',1),
        RunWorkerLimit=worker_limits,RunExecutionMode=run_modes,
        TimingEnvironment=environment,ContainsParallelRunTimes=has_parallel,
        TimingDefinition='reconstructed: measured seed Preparation + measured seed/sigma Spectral + measured parameter Scoring; every Time cell includes full shared costs once; excludes cache I/O, external preprocessing, metrics and saving',
        TimingCaveat='Time = measured per-seed shared Preparation + measured per-seed/sigma shared Spectral + this parameter Scoring. Shared costs are charged ONCE in EACH parameter record, never divided by parameter count. This reconstructs complete per-parameter cost; not a directly measured fresh independent run. Parallel contention applies. BatchWallSeconds is the actual current invocation duration including I/O and scheduling.',
        StartupWarmupSeconds=startup_seconds,BatchWallSeconds=batch_elapsed,
        BatchWallDefinition='current dataset invocation only, including worker startup, scheduling and checkpoint reads; excludes work completed in earlier invocations',
        ParameterNames=['sigma','alpha'])
    bytes_needed=len(pairs)*len(args.seeds)*len(y)*8
    external=args.external_scores or bytes_needed>args.inline_mb*1024**2
    def vectors_at(index):
        rows=[]
        for seed in args.seeds:
            with np.load(folder/f'param_{index:04d}_seed_{seed:03d}.npz',allow_pickle=False) as saved:
                rows.append(saved['scores'])
        vectors=np.stack(rows)
        if vectors.shape!=(len(args.seeds),len(y)) or not np.isfinite(vectors).all():
            raise ValueError('断点分数维度或数值无效')
        return vectors
    if external:

        generation=directory/'scores'/(path.stem+'_'+uuid.uuid4().hex)
        generation.mkdir(parents=True)
        score_paths=[]
        for index in range(len(pairs)):
            vectors=vectors_at(index)
            out=generation/f'param_{index}.npy';np.save(out,vectors,allow_pickle=False)
            score_paths.append(out.relative_to(directory).as_posix())
        result['scores_files']=score_paths
    else:
        result['OF']=np.stack([vectors_at(i) for i in range(len(pairs))])
    assert ('OF' in result)!=('scores_files' in result)
    temporary=directory/('.'+path.stem+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('wb') as f:pickle.dump(result,f,protocol=pickle.HIGHEST_PROTOCOL)
    archive_existing(destination)
    os.replace(temporary,destination)

    old_excel=destination.with_suffix('.xlsx')
    if old_excel.exists():
        old_archive=destination.parents[2]/'historical_results'/ALGORITHM/'legacy_excel'
        old_archive.mkdir(parents=True,exist_ok=True)
        os.replace(old_excel,old_archive/(path.stem+'_'+digest(old_excel)[:12]+'.xlsx'))
    return destination


def cleanup_completed_cache(folder,destination,expected_config,cache_root=None,dry_run=False):


    folder=Path(folder).resolve();root=Path(cache_root or STATE_DIR).resolve()
    destination=Path(destination).resolve()
    if folder==root or not folder.is_relative_to(root):
        raise ValueError('缓存目录不在指定运行目录内：'+str(folder))
    if not folder.is_dir() or not destination.is_file():return dict(status='skipped',reason='missing cache or result',bytes_removed=0)
    if destination.is_relative_to(folder):return dict(status='skipped',reason='result is inside cache',bytes_removed=0)
    try:
        with destination.open('rb') as f:result=pickle.load(f)
        if result.get('Algorithm')!=ALGORITHM or result.get('SchemaVersion')!=SCHEMA_VERSION:
            raise ValueError('结果版本不匹配')
        if not compatible_config(result.get('Config',{}),expected_config):raise ValueError('结果与缓存配置不匹配')
        labels=np.asarray(result['Labels']);seeds=list(result['Seeds']);parameters=result['Par']
        shape=(len(parameters),len(seeds) or 1)
        if seeds!=expected_config.get('Seeds'):raise ValueError('结果种子不完整')
        sigmas=expected_config.get('Sigmas',[]);alphas=expected_config.get('Alphas',[])
        expected_count=len(sigmas)*len(alphas) if expected_config.get('Search')=='grid' else len(sigmas)+len(alphas)
        if not sigmas or not alphas or shape[0]!=expected_count:raise ValueError('结果参数搜索未完整保存')
        if labels.ndim!=1 or not np.isfinite(labels).all():raise ValueError('标签不完整')
        for key in ['AUC','AP','Time']:
            values=np.asarray(result[key],dtype=float)
            if values.shape!=shape or not np.isfinite(values).all():raise ValueError('指标不完整：'+key)
            if key=='Time':
                if np.any(values<=0):raise ValueError('运行时间无效')
            elif np.any((values < -1e-12)|(values > 1+1e-12)):raise ValueError('指标越界')
        if ('OF' in result)==('scores_files' in result):raise ValueError('得分存储缺失或冲突')
        if 'OF' in result:
            values=np.asarray(result['OF'])
            if values.shape!=shape+(len(labels),) or not np.isfinite(values).all():raise ValueError('内嵌得分不完整')
        else:
            if len(result['scores_files'])!=shape[0]:raise ValueError('外部得分列表不完整')
            for name in result['scores_files']:
                score_path=(destination.parent/name).resolve()
                if score_path.is_relative_to(folder):raise ValueError('正式得分依赖缓存文件，保留缓存')
                values=np.load(score_path,mmap_mode='r',allow_pickle=False)
                valid=values.shape==(shape[1],len(labels)) and np.isfinite(values).all()
                del values
                if not valid:raise ValueError('外部得分不完整：'+str(score_path))
    except (OSError,ValueError,KeyError,TypeError,EOFError,pickle.UnpicklingError) as error:
        return dict(status='skipped',reason=str(error),bytes_removed=0)
    cleanup_lock=folder/'.cache_cleanup.lock'
    try:handle=os.open(cleanup_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:return dict(status='busy',bytes_removed=0)
    try:
        os.close(handle)
        candidates=[]
        for pattern in ['prepared_seed_*.pkl','param_*_seed_*.npz','input_features.npy','input_labels.npy']:
            candidates.extend(folder.glob(pattern))
        if dry_run:return dict(status='verified',result=str(destination),files=len(candidates),bytes=sum(p.stat().st_size for p in candidates),bytes_removed=0)
        removed=0;count=0;errors=[]
        for path in candidates:

            if path.parent.resolve()!=folder or path.is_symlink() or not path.resolve().is_relative_to(root):
                errors.append(str(path)+': unexpected path');continue
            try:
                size=path.stat().st_size;path.unlink();removed+=size;count+=1
            except OSError as error:errors.append(str(path)+': '+str(error))
        report=dict(status='cleaned' if not errors else 'partially_cleaned',result=str(destination),
            files_removed=count,bytes_removed=removed,errors=errors)
        if count or errors:
            try:atomic_json(folder/'cache_cleanup.json',report)
            except OSError as error:report['errors'].append('cleanup report: '+str(error))
        return report
    finally:cleanup_lock.unlink(missing_ok=True)


def run_dataset(path,args,startup_seconds=0.0):
    config=signature(path,args)
    destination=args.result_dir/ALGORITHM/(path.stem+'.pkl')
    if destination.exists() and not args.force:
        with destination.open('rb') as f:previous=pickle.load(f)
        if compatible_config(previous.get('Config',{}),config) and previous.get('SchemaVersion')==SCHEMA_VERSION:
            for saved in STATE_DIR.glob(path.stem+'_*/config.json'):
                try:
                    old=json.loads(saved.read_text(encoding='utf-8'))
                    if compatible_config(old,config):cleanup_completed_cache(saved.parent,destination,old)
                except (OSError,ValueError):pass
            report_best(path.stem,previous['Par'],previous['AUC'],previous['AP'])
            return destination
    folder=checkpoint_folder(path,config)
    folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'config.json').exists():atomic_json(folder/'config.json',config)
    history=folder/'execution_history';history.mkdir(exist_ok=True)
    atomic_json(history/(uuid.uuid4().hex+'.json'),config)
    X,y=load_data(path)
    progress=dict(status='running',dataset=path.stem,config=folder.name,result=str(destination),workers=getattr(args,'workers',1),planned_dataset_runs=(len(args.sigmas)*len(args.alphas) if args.search=='grid' else len(args.sigmas)+len(args.alphas))*len(args.seeds))
    batch_start=time.perf_counter()
    executor=None
    reported=set()
    try:
        if getattr(args,'workers',1)>1:
            input_path=folder/'input_features.npy';label_path=folder/'input_labels.npy'
            np.save(input_path,X,allow_pickle=False);np.save(label_path,y,allow_pickle=False)
            executor=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=worker_initialize,initargs=(str(input_path),str(label_path),args))
        if args.search=='grid':
            pairs=list(itertools.product(args.sigmas,args.alphas))
            metrics=parameter_runs(X,y,pairs,args,folder,0,progress,executor,reported)
        else:
            first=[(sigma,1.0) for sigma in args.sigmas]
            progress['stage']='sigma'
            metrics=parameter_runs(X,y,first,args,folder,0,progress,executor,reported)
            best=int(np.argmax(np.asarray(metrics[0]).mean(axis=1)))
            sigma=args.sigmas[best]
            second=[(sigma,alpha) for alpha in args.alphas]
            progress['stage']='alpha'
            extra=parameter_runs(X,y,second,args,folder,len(first),progress,executor,reported)
            metrics=tuple(a+b for a,b in zip(metrics,extra));pairs=first+second
            atomic_json(folder/'search_decision.json',dict(SelectedSigma=sigma,Selection='largest mean AUC over all seeds in sigma stage; first tie',FirstStageParameters=first))
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
    result=publish(path,config,pairs,metrics,folder,args,y,time.perf_counter()-batch_start,startup_seconds)
    atomic_json(folder/'completed.json',dict(result=str(result),parameter_count=len(pairs),seeds=args.seeds))
    cleanup_completed_cache(folder,result,config)
    report_best(path.stem,pairs,metrics[0],metrics[1])
    return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir',type=Path,help='覆盖顶部DATA_DIR：MAT输入文件夹')
    p.add_argument('--result-dir',type=Path,help='覆盖顶部RESULT_DIR：输出根目录，自动添加MDGOD子文件夹')
    p.add_argument('--dataset',nargs='+',default=DATASETS,help='覆盖顶部DATASETS：MAT文件名；支持一次指定多个')
    p.add_argument('--all-datasets',action='store_true',help='兼容旧命令；默认已遍历所选目录全部MAT')
    p.add_argument('--search',choices=['sequential','grid'],default=SEARCH_MODE)
    p.add_argument('--sigmas',nargs='+',type=float,default=SIGMAS)
    p.add_argument('--alphas',nargs='+',type=float,default=ALPHAS)
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--layers',type=int,default=LAYERS)
    p.add_argument('--eta',type=float,default=NOISE_STRENGTH)
    p.add_argument('--threads',type=int,default=THREADS)
    p.add_argument('--workers',type=int,default=WORKERS,help='种子间并行进程数；1为串行，同样复用参数间准备')
    p.add_argument('--block-rows',type=int,default=BLOCK_ROWS)
    p.add_argument('--inline-mb',type=float,default=INLINE_MB)
    p.add_argument('--external-scores',action='store_true',default=EXTERNAL_SCORES)
    p.add_argument('--force',action='store_true',help='重新发布结果；相同配置仍复用已完成的独立运行断点')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--status',action='store_true')
    a=p.parse_args(argv)
    a.data_dir=(a.data_dir or DATA_DIR).resolve()
    a.result_dir=(a.result_dir or RESULT_DIR).resolve()
    if a.layers<1 or not 0<=a.eta<=1 or a.threads<1 or a.workers<1 or a.block_rows<1 or a.inline_mb<0:p.error('无效运行配置')
    for values in [a.sigmas,a.alphas]:
        if len(set(values))!=len(values) or not all(np.isfinite(v) and 0<=v<=1 for v in values):p.error('参数需为0~1内互不重复的有限值')
    if len(set(a.seeds))!=len(a.seeds) or not all(0<=v<2**32 for v in a.seeds):p.error('种子需为互不重复的32位非负整数')
    return a


def main(argv=None):
    args=parse_args(argv)
    if args.status:
        path=STATE_DIR/'status.json'
        print(path.read_text(encoding='utf-8') if path.exists() else '尚未启动MDGOD专用main');return 0
    if not args.data_dir.is_dir():raise FileNotFoundError(args.data_dir)
    files=select_files(args.data_dir,args.dataset,args.all_datasets)
    if not files or any(not p.is_file() for p in files):raise FileNotFoundError('数据文件缺失：'+str([str(p) for p in files if not p.is_file()]))
    if args.dry_run:
        print('Input:',args.data_dir,'Output:',args.result_dir,'Datasets:',len(files),'Workers:',args.workers,'Threads per worker:',args.threads)
        for file in files:print(file.name)
        return 0
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    lock=STATE_DIR/'running.lock'
    try:
        handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        import psutil
        try:
            active=json.loads(lock.read_text(encoding='utf-8'))
            alive=psutil.pid_exists(active['pid']) and abs(psutil.Process(active['pid']).create_time()-active['created'])<1
        except (OSError,ValueError,KeyError,psutil.Error):alive=False
        if alive:raise RuntimeError('MDGOD已在运行；请使用--status查看进度')
        lock.unlink();handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    import psutil
    with os.fdopen(handle,'w',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    failures=[]
    try:
        start=time.perf_counter();warmup();startup=time.perf_counter()-start
        print(f'MDGOD: drop raw 0/1 columns then MinMax; seeds={args.seeds}; search={args.search}; workers={args.workers}; threads per worker={args.threads}; shared preparation; reconstructed complete per-parameter times; JIT warmup={startup:.3f}s',flush=True)
        if args.workers>1:print('Parallel Time records full run wall time under resource contention; use workers=1 for comparable serial runtime measurements.',flush=True)
        for path in sorted(files,key=lambda p:(p.stat().st_size,p.name)):
            try:run_dataset(path,args,startup)
            except Exception as error:
                import traceback
                failures.append(dict(dataset=path.stem,error=str(error)));traceback.print_exc()
        atomic_json(STATE_DIR/'status.json',dict(status='completed' if not failures else 'completed_with_errors',datasets=len(files),seeds=args.seeds,failures=failures,finished_at=datetime.now().isoformat()))
    finally:lock.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
