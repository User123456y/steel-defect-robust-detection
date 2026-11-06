#!/usr/bin/env python3
"""导出ONNX和TensorRT模型，测量CUDA推理并复核基准结果。"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ['YOLO_AUTOINSTALL'] = 'false'
from steel_defect.gpu import BACKENDS, WEIGHTS_SHA, GPUDetector, accuracy_gate, configure_cuda, latency_summary, validate_profile
from steel_defect.inference import file_sha256
from steel_defect.data import CLASSES
from steel_defect.evaluation import match_image, official_metrics, read_yolo_labels, summarize


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def export(args):
    profile_sha = validate_profile(args.profile)
    torch = configure_cuda()
    import onnx
    import tensorrt as trt
    from ultralytics import YOLO
    if not trt.__version__.startswith('10.'):
        raise RuntimeError('This builder requires TensorRT 10.x; 11.x removed precision flags')
    if file_sha256(args.weights) != WEIGHTS_SHA:
        raise ValueError('Frozen checkpoint mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    copied = args.output / 'frozen.pt'
    shutil.copy2(args.weights, copied)


    # 固定旧导出器以保持opset17协议；兼容PyTorch 2.9之后的默认行为变化。
    from functools import partial
    original_export = torch.onnx.export
    torch.onnx.export = partial(original_export, dynamo=False)
    try:
        onnx_path = Path(YOLO(str(copied)).export(format='onnx', imgsz=320, batch=1, dynamic=False,
                        half=False, simplify=False, opset=17, nms=False, device=0))
    finally:
        torch.onnx.export = original_export
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    metadata = {x.key: x.value for x in graph.metadata_props}
    for key in ('stride', 'batch', 'imgsz', 'names', 'args'):
        if key in metadata:
            metadata[key] = ast.literal_eval(metadata[key])
    metadata.update(batch=1, imgsz=[320, 320], task='detect')
    logger = trt.Logger(trt.Logger.INFO)
    for backend in ('trt_fp32', 'trt_fp16'):
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        config.clear_flag(trt.BuilderFlag.TF32)
        if backend == 'trt_fp16':
            if not builder.platform_has_fast_fp16:
                raise RuntimeError('GPU lacks fast FP16')
            config.set_flag(trt.BuilderFlag.FP16)
        tick = time.perf_counter()
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError('TensorRT build failed')
        path = args.output / f'{backend}.engine'
        header = json.dumps(metadata).encode()
        path.write_bytes(len(header).to_bytes(4, 'little', signed=True) + header + bytes(engine))
        write(path.with_suffix('.json'), dict(backend=backend, weights_sha256=WEIGHTS_SHA,
              profile_sha256=profile_sha, compute_capability=list(torch.cuda.get_device_capability(0)),
              engine_sha256=file_sha256(path), onnx_sha256=file_sha256(onnx_path),
              tensorrt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0), tf32=False, fp16_tactics=backend == 'trt_fp16',
              onnx_exporter='torchscript dynamo=False opset17', io='float32', shape=[1, 3, 320, 320], nms=False, workspace_bytes=2 << 30,
              build_seconds=time.perf_counter()-tick))


def val_records(root):
    manifest = root / 'manifest.json'
    data = json.loads(manifest.read_text())
    identity_path = Path(__file__).resolve().parents[1] / 'configs/validation_identity.json'
    expected = json.loads(identity_path.read_text())
    actual = [{k: r[k] for k in ('image_sha256', 'width', 'height', 'boxes')}
              for r in data['records'] if r['split'] == 'val']
    actual = [hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest() for r in actual]
    if actual != expected:
        raise ValueError('验证集身份或标注与原实测数据不一致')
    if data['classes'] != list(CLASSES):
        raise ValueError('Class mapping mismatch')
    records = [r for r in data['records'] if r['split'] == 'val']
    if len(records) != 269 or sum(len(r['boxes']) for r in records) != 612:
        raise ValueError('Expected frozen val: 269 images / 612 GT')
    evidence = []
    for r in records:
        image, label = [(root / r[k]).resolve() for k in ('output_image', 'output_label')]
        if not all(p.is_relative_to(root) for p in (image, label)):
            raise ValueError('Path escapes data root')
        if file_sha256(image) != r['image_sha256']:
            raise ValueError('Image hash mismatch')
        gt = read_yolo_labels(label, r['width'], r['height'], 6)
        if len(gt) != len(r['boxes']) or any(b['class_id'] != g[0] or any(abs(x-y)>1e-5 for x,y in zip(b['xyxy'],g[1:])) for b,g in zip(gt,r['boxes'])):
            raise ValueError('Label does not match frozen manifest')
        evidence.append(dict(width=r['width'],height=r['height'],path=str(image), image_sha256=file_sha256(image), label_sha256=file_sha256(label), ground_truth=gt))
    if len({r['path'] for r in evidence}) != len(evidence):
        raise ValueError('Duplicate validation images')
    return manifest, evidence


def run(args):
    import numpy as np
    import yaml
    import pynvml
    root = args.data.resolve()
    manifest, records = val_records(root)
    args.output.mkdir(parents=True, exist_ok=False)
    detector = GPUDetector(args.model, args.backend, args.profile)
    torch = detector.torch
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def memory():
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        matches=[p.usedGpuMemory for p in processes if p.pid==os.getpid()]
        return sum(matches) if matches else None
    report = dict(status='started', backend=args.backend, weights_sha256=WEIGHTS_SHA,
                  profile_sha256=detector.profile_sha256,
                  model_sha256=file_sha256(args.model), manifest_sha256=file_sha256(manifest),
                  versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics','numpy','onnx','Pillow','tensorrt-cu13')},
                  gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                  driver=pynvml.nvmlSystemGetDriverVersion(), batch=1, imgsz=320, rect=False,
                  confidence=.4, nms_iou=.7, preprocessing='none', tf32=False,
                  warmup=args.warmup, repeats=args.repeats, records=records,
                  code_sha256={str(p.name):file_sha256(p) for p in (Path(__file__),*(Path(__file__).resolve().parents[1]/'src/steel_defect'/n for n in ('gpu.py','inference.py','evaluation.py')))})
    write(args.output/'report.json',report)
    try:
        for i in range(args.warmup):
            detector.predict(records[i % len(records)]['path'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        peak_nvml = memory()
        device_baseline = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        device_peak = device_baseline
        report['gpu_start'] = dict(temperature_c=pynvml.nvmlDeviceGetTemperature(handle,0),
            graphics_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle,0), utilization_percent=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        forward_ms, request_ms = [], []

        # 前向循环不计预处理和CPU复制；完整请求循环计入全部推理步骤。
        for i in range(args.repeats):
            tensor, _ = detector.prepare(records[i % len(records)]['path'])
            torch.cuda.synchronize()
            tick = time.perf_counter()
            raw = detector.forward(tensor)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter()-tick)*1000)
            del raw, tensor
            observed=memory()
            if observed is not None: peak_nvml=max(peak_nvml or 0,observed)
            device_peak=max(device_peak,pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        for i in range(args.repeats):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            detector.predict(records[i % len(records)]['path'])
            torch.cuda.synchronize()
            request_ms.append((time.perf_counter()-tick)*1000)
            observed=memory()
            if observed is not None: peak_nvml=max(peak_nvml or 0,observed)
            device_peak=max(device_peak,pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        report['benchmark'] = dict(forward=latency_summary(forward_ms), request=latency_summary(request_ms),
              forward_samples_ms=forward_ms, request_samples_ms=request_ms,
              torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              nvml_process_sampled_peak_bytes=peak_nvml,
              nvml_device_baseline_bytes=device_baseline,nvml_device_sampled_peak_bytes=device_peak,
              nvml_process_note='null if process PID cannot be matched (e.g. container PID namespace); device memory includes all GPU users',
              memory_note='NVML sampled between requests; not an exact transient peak. Torch excludes TensorRT-owned allocation.')
        report['gpu_end'] = dict(temperature_c=pynvml.nvmlDeviceGetTemperature(handle,0),
            graphics_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle,0), utilization_percent=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        images, raw_outputs, input_hashes = [], [], []
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from ultralytics.utils.metrics import DetMetrics
        validator = DetectionValidator(args=dict(task='detect', plots=False))
        metrics = DetMetrics(names=dict(enumerate(CLASSES)))
        import hashlib
        for r in records:
            tensor, shape = detector.prepare(r['path'])
            if shape != (r['height'],r['width']):
                raise ValueError('Decoded dimensions differ from manifest')
            input_hashes.append(hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest())
            raw = detector.forward(tensor)
            raw_outputs.append(raw.detach().float().cpu().numpy())
            predictions = detector.postprocess(raw, shape)
            low = detector.postprocess(raw, shape, confidence=.001)
            pred_boxes = torch.tensor([p['xyxy'] for p in low], dtype=torch.float32).reshape(-1,4)
            pred_cls = torch.tensor([p['class_id'] for p in low], dtype=torch.float32)
            gt_boxes = torch.tensor([g['xyxy'] for g in r['ground_truth']], dtype=torch.float32).reshape(-1,4)
            gt_cls = torch.tensor([g['class_id'] for g in r['ground_truth']], dtype=torch.float32)
            tp = validator._process_batch(dict(bboxes=pred_boxes, cls=pred_cls),dict(bboxes=gt_boxes,cls=gt_cls))['tp']
            metrics.update_stats(dict(tp=tp,conf=np.array([p['confidence'] for p in low]),
                 pred_cls=pred_cls.numpy(),target_cls=gt_cls.numpy(),target_img=np.unique(gt_cls.numpy())))
            images.append(dict(path=r['path'], ground_truth=r['ground_truth'], detections=predictions,low_confidence_detections=low,
                               analysis=match_image(r['ground_truth'], predictions, .5)))
        stacked = np.concatenate(raw_outputs)
        if not np.isfinite(stacked).all():
            raise RuntimeError('Nonfinite raw predictions')
        np.save(args.output/'raw_outputs.npy',stacked)
        report['input_sha256'] = input_hashes
        report['fixed'] = summarize(images, CLASSES)
        write(args.output/'predictions.json',images)
        metrics.process(save_dir=args.output,plot=False)
        report['official']=official_metrics(metrics, CLASSES,[r['gt_count'] for r in report['fixed']['per_class']])
        report['official_pipeline']='Ultralytics 8.3.200 DetectionValidator matching + DetMetrics AP on identical Pillow/LetterBox/raw backend/common NMS at conf=.001'
        if args.backend=='torch_fp32':
            from steel_defect.inference import Detector
            legacy=Detector(args.model,'0')
            mismatches=[]
            for index,(record,current) in enumerate(zip(records,images,strict=True)):
                previous=legacy.predict([record['path']],confidence=.4,iou=.7,imgsz=320)['images'][0]
                if previous['status']!='success': raise RuntimeError('Legacy reference prediction failed')
                old=previous['detections']; new=current['detections']
                equal=len(old)==len(new) and all(a['class_id']==b['class_id'] and abs(a['confidence']-b['confidence'])<=1e-5 and all(abs(x-y)<=.001 for x,y in zip(a['xyxy'],b['xyxy'])) for a,b in zip(old,new))
                if not equal: mismatches.append(index)
            report['legacy_val_equivalence']=dict(passed=not mismatches,mismatch_indices=mismatches,scope='frozen square val only; not arbitrary non-square inputs',coordinate_atol=.001,confidence_atol=.00001)
            if mismatches: raise RuntimeError('New torch backend differs from legacy detector on frozen val')
        report['artifact_sha256']={n:file_sha256(args.output/n) for n in ('raw_outputs.npy','predictions.json')}
        if file_sha256(args.model)!=report['model_sha256'] or file_sha256(manifest)!=report['manifest_sha256']:
            raise RuntimeError('Model or manifest changed during run')
        for record in records:
            if file_sha256(record['path'])!=record['image_sha256']: raise RuntimeError('Image changed during run')
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(args.output/'report.json',report)
        pynvml.nvmlShutdown()


def compare(args):
    import numpy as np
    ref=json.loads((args.reference/'report.json').read_text())
    cand=json.loads((args.candidate/'report.json').read_text())
    if ref['status']!='completed' or cand['status']!='completed' or ref['backend']!='torch_fp32' or cand['backend'] not in ('trt_fp32','trt_fp16'):
        raise ValueError('Completed PyTorch reference and completed candidate required')
    for k in ('weights_sha256','profile_sha256','manifest_sha256','input_sha256','batch','imgsz','rect','confidence','nms_iou','preprocessing','tf32','gpu','driver','cuda','versions','code_sha256','warmup','repeats'):
        if ref[k]!=cand[k]:
            raise ValueError(f'Incomparable {k}')
    if [{k:r[k] for k in ('image_sha256','label_sha256')} for r in ref['records']] != [{k:r[k] for k in ('image_sha256','label_sha256')} for r in cand['records']]:
        raise ValueError('Different validation evidence')
    for directory, report in ((args.reference,ref),(args.candidate,cand)):
        for name,digest in report['artifact_sha256'].items():
            if file_sha256(directory/name)!=digest: raise ValueError('Result artifact hash mismatch')
        images=json.loads((directory/'predictions.json').read_text())
        if len(images)!=269 or len({i['path'] for i in images})!=269:
            raise ValueError('Incomplete prediction coverage')
        for image,record in zip(images,report['records'],strict=True):
            if image['path']!=record['path'] or image['ground_truth']!=record['ground_truth']:
                raise ValueError('Prediction identity/GT mismatch')
            image['analysis']=match_image(image['ground_truth'],image['detections'],.5)
        if summarize(images,CLASSES)!=report['fixed']: raise ValueError('Fixed metric replay mismatch')
        for kind in ('forward','request'):
            samples=report['benchmark'][kind+'_samples_ms']
            if len(samples)!=report['repeats'] or latency_summary(samples)!=report['benchmark'][kind]:
                raise ValueError('Benchmark sample/summary mismatch')
    a=np.load(args.reference/'raw_outputs.npy'); b=np.load(args.candidate/'raw_outputs.npy')
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1]!=10:
        raise ValueError('Unexpected raw output shape')
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    error=np.abs(a-b); error[:,:4,:]/=320
    box_q99=float(np.quantile(error[:,:4,:],.99)) if finite else None
    score_q99=float(np.quantile(error[:,4:,:],.99)) if finite else None
    q99=max(box_q99,score_q99) if finite else None
    tolerance=.01 if cand['backend']=='trt_fp16' else .001
    numeric=dict(finite=finite,box_error_q99=box_q99,score_error_q99=score_q99,
                 box_error_max=float(error[:,:4,:].max()) if finite else None,
                 score_error_max=float(error[:,4:,:].max()) if finite else None,tolerance=tolerance,
                 passed=finite and q99<=tolerance,
                 note='Coordinates divided by 320; scores absolute; raw tensor before common NMS')
    gate=accuracy_gate(ref,cand)
    result=dict(status='passed' if gate['passed'] and numeric['passed'] else 'rejected',accuracy_gate=gate,numeric_gate=numeric)
    if result['status']=='passed':
        result['speedup']={k:ref['benchmark'][k]['p50_ms']/cand['benchmark'][k]['p50_ms'] for k in ('forward','request')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        out.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


def predict(args):
    detector = GPUDetector(args.model,args.backend,args.profile)
    result = detector.predict(args.image)
    result['metadata'] = dict(source_weights_sha256=WEIGHTS_SHA, artifact_sha256=file_sha256(args.model),
          profile_sha256=detector.profile_sha256, backend=args.backend, batch=1, imgsz=320,
          confidence=.4, nms_iou=.7, rect=False, preprocessing='none')
    print(json.dumps(result,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('export'); e.add_argument('--weights',type=Path,required=True); e.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run'); r.add_argument('--model',type=Path,required=True); r.add_argument('--backend',choices=BACKENDS,required=True)
    r.add_argument('--data',type=Path,required=True,help='processed_v2 directory; val only'); r.add_argument('--output',type=Path,required=True)
    r.add_argument('--warmup',type=int,default=50); r.add_argument('--repeats',type=int,default=538)
    c=sub.add_parser('compare'); c.add_argument('--reference',type=Path,required=True); c.add_argument('--candidate',type=Path,required=True); c.add_argument('--output',type=Path,required=True)
    d=sub.add_parser('predict'); d.add_argument('--model',type=Path,required=True); d.add_argument('--backend',choices=BACKENDS,required=True); d.add_argument('--image',type=Path,required=True)
    for parser in (e,r,d):
        parser.add_argument('--profile',type=Path,required=True)
    args=p.parse_args()
    if args.action=='run' and (args.warmup<50 or args.repeats<300):
        p.error('At least 50 warmup and 300 measured iterations required')
    return {'export':export,'run':run,'compare':compare,'predict':predict}[args.action](args)
if __name__=='__main__':
    raise SystemExit(main())
