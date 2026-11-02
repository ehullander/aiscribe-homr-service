"""Offline structural inference, shared by desktop and the native iOS bridge.

Uses HOMR 0.7.0 staff/note/bar geometry, without its OCR or music transformer.
No downloads, API credentials, user corrections, or reference charts.
"""
import hashlib
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
from PIL import Image

MODEL_SHA256='6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d'
_verified_models=set()


def available():
    # Desktop validation opts in explicitly; production iOS has a built-in
    # module, rather than depending on an environment variable or network.
    return False  # CPU service never imports Apple native modules.


def _model_path():
    if available():
        return Path(__file__).resolve().parents[2]/'assets'/'omr'/'segnet.onnx'
    from homr.segmentation.config import segnet_path_onnx
    return Path(segnet_path_onnx)


def _verify_model(model):
    if model not in _verified_models:
        if hashlib.sha256(model.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise ValueError('Bundled staff model failed integrity verification')
        _verified_models.add(model)
    return model


def warmup():
    """Load the exact verified session used by the next phone inference."""
    started = perf_counter()
    if not available():
        return dict(available=False, cold_load=False)
    import _aiscribe_vision
    model=_verify_model(_model_path())
    verified = perf_counter()
    cold_load = bool(_aiscribe_vision.warmup(str(model)))
    finished = perf_counter()
    return dict(available=True, cold_load=cold_load,
                verification_seconds=round(verified-started, 6),
                session_seconds=round(finished-verified, 6),
                total_seconds=round(finished-started, 6))


def _predict(prepared, model):
    from homr.model import InputPredictions
    import cv2
    preprocessed=cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8)).apply(cv2.cvtColor(prepared,cv2.COLOR_BGR2GRAY))
    if available():
        import _aiscribe_vision
        from .native_cv import unpack
        infer=lambda batch:unpack(_aiscribe_vision.infer(str(model),batch))
    else:
        from .runtime import session_for
        session=session_for(model)
        infer=lambda batch:session.run(['output'],{'input':batch})[0]
    # Match HOMR's non-overlapping 320px grid, including repeated edge tiles
    # and its average-then-cast merge. Small batches bound phone peak memory.
    height,width=preprocessed.shape;window=320
    total=np.zeros((height,width),np.uint16);weights=np.zeros((height,width),np.uint8)
    tiles=[];positions=[]
    def flush():
        if not tiles:return
        output=infer(np.ascontiguousarray(np.stack(tiles),dtype=np.float32))
        for scores,(y,x,h,w) in zip(output,positions):
            total[y:y+h,x:x+w] += np.argmax(scores,axis=0)[:h,:w].astype(np.uint16)
            weights[y:y+h,x:x+w] += 1
        tiles.clear();positions.clear()
    for yy in range(0,max(height,window),window):
        y=max(0,min(yy,height-window))
        for xx in range(0,max(width,window),window):
            x=max(0,min(xx,width-window));h=min(window,height-y);w=min(window,width-x)
            tile=np.full((3,window,window),255,dtype=np.float32)
            tile[:,:h,:w]=preprocessed[y:y+h,x:x+w]
            tiles.append(tile);positions.append((y,x,h,w))
            if len(tiles)==2:flush()
    # Labels are nonnegative small integers. Integer floor division is exactly
    # the former average-then-cast rule, without page-sized float/int64 copies.
    flush();np.floor_divide(total,np.maximum(weights,1),out=total)
    merged=total.astype(np.uint8)
    return InputPredictions(original=prepared,preprocessed=preprocessed,
        notehead=(merged==2).astype(np.uint8),symbols=(merged==5).astype(np.uint8),
        staff=(merged==4).astype(np.uint8),clefs_keys=(merged==3).astype(np.uint8),
        stems_rest=(merged==1).astype(np.uint8))


def _staff_anchor_candidates(bars, fragments):
    """Look beside thick bar strokes without inventing extra barlines.

    HOMR's fixed ten-pixel anchor search can stop inside a thick terminal bar.
    Retry an unsupported anchor one staff spacing to either side, retaining
    HOMR's existing five-line, parallelism, and endpoint checks. These proxies
    are used only for staff assembly; exported bar positions remain original.
    """
    from homr.staff_detection import find_staff_anchors
    candidates=list(bars)
    for bar in bars:
        if find_staff_anchors(fragments,[bar],are_clefs=False):
            continue
        spacing=max(1,round(float(bar.size[1])/4))
        for offset in (-spacing,spacing):
            candidate=bar.move_to_x_horizontal_by(offset)
            if find_staff_anchors(fragments,[candidate],are_clefs=False):
                candidates.append(candidate)
                break
    return candidates


def detect(source: Path, *, model_path: Path | None=None):
    started=perf_counter()
    if available():
        from .native_cv import install
        install()
    from homr.resize import resize_image
    from homr.debug import Debug
    from homr.noise_filtering import filter_predictions
    from homr.bounding_boxes import create_bounding_ellipses,create_rotated_bounding_boxes
    from homr.bar_line_detection import prepare_bar_line_image,detect_bar_lines
    from homr.note_detection import combine_noteheads_with_stems
    from homr.staff_detection import break_wide_fragments,detect_staff,make_lines_stronger
    from .barline_fusion import ordered_staff_trace
    model=_verify_model(model_path or _model_path())
    with Image.open(source) as image:original=np.asarray(image.convert('RGB'))[:,:,::-1].copy()
    prepared=resize_image(original)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    prediction_started=perf_counter()
    prediction=_predict(prepared,model)
    prediction_seconds=perf_counter()-prediction_started
    geometry_started=perf_counter()
    debug=Debug(prediction.original,str(source),False)
    prediction=filter_predictions(prediction,debug)
    prediction.staff=make_lines_stronger(prediction.staff,(1,2))
    heads=create_bounding_ellipses(prediction.notehead,min_size=(4,4))
    fragments=break_wide_fragments(create_rotated_bounding_boxes(prediction.staff,skip_merging=True,min_size=(5,1),max_size=(10000,100)))
    clefs=create_rotated_bounding_boxes(prediction.clefs_keys,min_size=(20,40),max_size=(1000,1000))
    stems=create_rotated_bounding_boxes(prediction.stems_rest)
    candidates=create_rotated_bounding_boxes(prepare_bar_line_image(prediction.stems_rest),skip_merging=True,min_size=(1,5))
    notes=combine_noteheads_with_stems(heads,stems)
    if not notes:raise ValueError('No noteheads available to size staff geometry')
    unit=float(np.median([n.notehead.size[1] for n in notes]))
    noteheads=[n.notehead for n in notes];note_stems=[n.stem for n in notes if n.stem is not None]
    bars=detect_bar_lines([b for b in candidates if not b.is_overlapping_with_any(noteheads) and not b.is_overlapping_with_any(note_stems)],unit)
    anchor_candidates=_staff_anchor_candidates(bars,fragments)
    staffs=detect_staff(debug,prediction.staff,fragments,clefs,anchor_candidates)
    sx=original.shape[1]/prediction.staff.shape[1];sy=original.shape[0]/prediction.staff.shape[0]
    rows=[]
    for number,staff in enumerate(sorted(staffs,key=lambda s:s.min_y),1):
        xs=[]
        for candidate in bars:
            point=staff.get_at(candidate.center[0])
            if point is None:continue
            y0=min(candidate.top_left[1],candidate.top_right[1]);y1=max(candidate.bottom_left[1],candidate.bottom_right[1])
            tolerance=float(point.average_unit_size)*.65
            if abs(y0-point.y[0])<=tolerance and abs(y1-point.y[-1])<=tolerance:xs.append(float(candidate.center[0])*sx)
        groups=[]
        for x in sorted(xs):
            if groups and x-groups[-1][-1]<=staff.average_unit_size*sx*1.25:groups[-1].append(x)
            else:groups.append([x])
        rows.append(dict(system=number,top=float(staff.min_y)*sy,bottom=float(staff.max_y)*sy,
                         left=float(staff.min_x)*sx,right=float(staff.max_x)*sx,spacing=float(staff.average_unit_size)*sy,
                         staff_trace=ordered_staff_trace([dict(x=float(p.x)*sx,top=float(p.y[0])*sy,bottom=float(p.y[-1])*sy) for p in staff.grid]),
                         barlines=[sum(g)/len(g) for g in groups]))
    # Use notehead/stem labels only for notes anchored inside a traced staff.
    # A round letter in an inter-staff chord must never seed note erasure.
    from io import BytesIO
    import base64
    from PIL import ImageDraw
    supported=Image.new('1',(prediction.staff.shape[1],prediction.staff.shape[0]))
    draw=ImageDraw.Draw(supported);notation_notes=[]
    for n in notes:
        cx,cy=n.notehead.center
        in_staff=False
        for row in rows:
            x=float(cx)*sx;y=float(cy)*sy;trace=row['staff_trace']
            if not row['left']<=x<=row['right']:continue
            top=np.interp(x,[p['x'] for p in trace],[p['top'] for p in trace])
            bottom=np.interp(x,[p['x'] for p in trace],[p['bottom'] for p in trace])
            if top<=y<=bottom:in_staff=True;break
        if not in_staff:continue
        notation_notes.append(n)
        w,h=n.notehead.size
        draw.ellipse((cx-w*.6,cy-h*.6,cx+w*.6,cy+h*.6),fill=1)
        if n.stem is not None:
            draw.polygon([tuple(map(float,p)) for p in
                          (n.stem.top_left,n.stem.top_right,n.stem.bottom_right,n.stem.bottom_left)],fill=1)
    mask=(prediction.notehead>0)&np.asarray(supported,dtype=bool)
    stream=BytesIO();Image.fromarray(mask.astype(np.uint8)*255).save(stream,format='PNG')
    result=dict(source=str(source.resolve()),source_sha256=source_hash,
                note_ink_mask=dict(png_base64=base64.b64encode(stream.getvalue()).decode('ascii'),
                                   source_width=original.shape[1],source_height=original.shape[0]),
                engine='homr-structural',version='0.7.0',model_sha256=MODEL_SHA256,
                runtime='native-ios-cpu' if available() else 'desktop-cpu',systems=rows,
                staff_anchor_recovery_count=len(anchor_candidates)-len(bars),
                notes=[dict(head=dict(cx=float(n.notehead.center[0])*sx,cy=float(n.notehead.center[1])*sy,
                                      width=float(n.notehead.size[0])*sx,height=float(n.notehead.size[1])*sy),
                            stem=[[float(p[0])*sx,float(p[1])*sy] for p in
                                  (n.stem.top_left,n.stem.top_right,n.stem.bottom_right,n.stem.bottom_left)]
                                  if n.stem is not None else []) for n in notation_notes],
                notehead_boxes=[dict(cx=float(n.center[0])*sx,cy=float(n.center[1])*sy,width=float(n.size[0])*sx,height=float(n.size[1])*sy) for n in heads],
                elapsed_seconds=round(perf_counter()-started,3),api_calls=0,uses_user_edits=False,uses_rhythm_transformer=False,
                inference_cache_hit=False,prediction_seconds=round(prediction_seconds,3),geometry_seconds=round(perf_counter()-geometry_started,3))
    return result
