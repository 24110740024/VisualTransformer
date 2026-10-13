"""Frozen, threshold-free CPU diagnostics; no outcomes, training or inference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
sys.path.insert(0,str(NEW/'annotation_tools_deps'))
import shapely


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7),(a,b)


from rectangle_field import (field_from_rectangles,independent_difference,check_fields,footprint_from_cells,footprint_from_windows,region_metrics,center_metrics,covered_area,self_test)


def pair_auc(y,p):
    positive=p[y==1];negative=np.sort(p[y==0])
    if not len(positive) or not len(negative):return None
    return float(np.sum((np.searchsorted(negative,positive,'left')+np.searchsorted(negative,positive,'right'))*.5)/(len(positive)*len(negative)))


def centers_for_image(raw,image_name,source_sha):
    subset=raw[raw.image_name==image_name].copy()
    assert set(subset.source_sha256).issubset({source_sha})
    assert set(subset.category).issubset({'Lesion','Normal','Normal-Far','Normal-Red'})
    subset['y_lesion']=(subset.category=='Lesion').astype(int)
    if len(subset):assert subset.groupby(['center_x','center_y']).y_lesion.nunique().max()==1
    unique=subset.drop_duplicates(['center_x','center_y']).copy()
    return unique,len(subset)-len(unique)


def main(version):
    protocol=read(HERE/'SPATIAL_INPUT_BINDING.json')
    for path,digest in protocol['files_sha256'].items():assert sha(path)==digest,path
    assert version in protocol['versions']
    root=NEW/protocol['versions'][version]['root']
    heatdir=root/protocol['versions'][version]['heatmap_directory']
    ha=read(heatdir/'report_audit.json');assert ha['cases']==21 and ha['frozen_tiles']==132032
    state=read(root/'jobs/pipeline_status.json');assert state['phase']=='core_complete'
    aggregate=Path(state['aggregate']);ac=read(aggregate/'COMPLETE.json')
    assert sha(aggregate/'COMPLETE.json')==ha['source_aggregate_complete_sha256']
    frozen=read(NEW/'scale_candidate_v4/dense_scaled/FROZEN.json');assert sha(NEW/'scale_candidate_v4/dense_scaled/COMPLETE.json')==frozen['complete_sha256']
    assert sha(NEW/'scale_candidate_v4/dense_scaled/coordinates_all21.csv')==frozen['manifest_sha256']
    metas=read(NEW/'scale_candidate_v4/dense_scaled/COMPLETE.json')['cases']
    rf=read(NEW/'roi_reference_v1/FROZEN_REFERENCE.json')
    for path,digest in rf['files_sha256'].items():assert sha(path)==digest,path
    ra=read(rf['accepted_report']);refs=ra['dense_reference_map']
    center_path=NEW/protocol['center_audit_relative'];assert sha(center_path)==ra['png_center_checks_sha256']
    raw=pd.read_csv(center_path)
    keys=['case_id','source_sha256','x','y','crop_size']
    predictions=[]
    for run in ac['input_runs']:
        directory=Path(run['directory']);assert directory.is_relative_to(root/'predictions')
        path=directory/'tile_predictions.csv';assert sha(path)==run['predictions_sha256']
        predictions.append(pd.read_csv(path,float_precision='round_trip',usecols=keys+['seed','prob_lesion']))
    predictions=pd.concat(predictions,ignore_index=True)
    assert len(predictions)==396096 and not predictions.duplicated(keys+['seed']).any()
    wide=predictions.pivot(index=keys,columns='seed',values='prob_lesion')
    assert list(wide.columns)==[42,43,44] and len(wide)==132032 and not wide.isna().any().any()
    tiles=wide.mean(axis=1).rename('probability').reset_index()
    fc=pd.read_csv(NEW/'scale_candidate_v4/dense_scaled/coordinates_all21.csv',usecols=keys)
    assert tiles[keys].sort_values(keys).reset_index(drop=True).equals(fc.sort_values(keys).reset_index(drop=True))
    out=HERE/datetime.now(timezone.utc).strftime(f'{version}_canonical_%Y%m%dT%H%M%SZ');out.mkdir(exist_ok=False)
    objects=[];references=[];point_records=[];checks=[]
    for meta in metas:
        case=meta['case_id'];source=meta['source_sha256'];group=tiles[tiles.case_id==case]
        assert len(group)==meta['coordinate_count'] and set(group.source_sha256)=={source}
        assert meta['source_width']==meta['thumbnail_width']*64 and meta['source_height']==meta['thumbnail_height']*64
        coords=group[['x','y','crop_size']].to_numpy();values=group.probability.to_numpy()
        field=field_from_rectangles(coords,values,meta['source_width'],meta['source_height'])
        independent=independent_difference(coords,values,meta['source_width'],meta['source_height']);check_fields(field,independent)
        path=heatdir/case/'heatmap_arrays.npz'
        key=f'{case}\\heatmap_arrays.npz' if f'{case}\\heatmap_arrays.npz' in ha['outputs_sha256'] else f'{case}/heatmap_arrays.npz'
        assert sha(path)==ha['outputs_sha256'][key]
        old=np.load(path)
        saved={'x_edges':old['x_edges'],'y_edges':old['y_edges'],'probability':old['exact_probability'],'coverage':old['exact_coverage']}
        check_fields(saved,field);field=saved
        footprint=footprint_from_cells(field);raw_footprint=footprint_from_windows(coords)
        assert footprint.symmetric_difference(raw_footprint).area==0
        close(footprint.area,covered_area(field))
        checks.append(dict(case_id=case,raster_matches=True,raw_window_union_matches=True,covered_area_px2=covered_area(field)))
        for ref in [r for r in refs if r['case_id']==case]:
            assert ref['source_sha256']==source and ref['outer_fold']==meta['outer_fold']
            common={k:ref[k] for k in ['case_id','source_sha256','outer_fold','reference_version_index','reference_version_count','project_image_name']}
            included=[];geo=read(Path(ref['extraction_dir'])/'annotations.geojson')
            for feature in geo['features']:
                props=feature['properties']
                if props['excluded_by_saved_tiler']:continue
                path=Path(ref['extraction_dir'])/props['wkb_path'];assert sha(path)==props['wkb_sha256']
                geom=shapely.from_wkb(path.read_bytes());included.append(geom)
                m=region_metrics(geom,field,footprint)
                objects.append(dict(**common,annotation_index=props['annotation_index'],path_class=props['path_class'],unclassified_included_by_tiler=props['path_class'] is None,**m))
            assert len(included)==ref['included_annotation_count']
            union=shapely.union_all(included) if included else None
            spatial=region_metrics(union,field,footprint) if union is not None else {}
            centers,dropped=centers_for_image(raw,ref['project_image_name'],source)
            cm,values,scored=center_metrics(centers,field)
            references.append(dict(**common,roi_present=bool(included),included_objects=len(included),duplicate_export_centers_removed=dropped,**spatial,**cm))
            for row,pred,valid in zip(centers.to_dict('records'),values,scored):
                point_records.append(dict(**common,center_x=row['center_x'],center_y=row['center_y'],y_lesion=row['y_lesion'],covered=bool(valid),heatmap_probability=float(pred) if valid else None))
        print(f'{case}: reference diagnostics complete',flush=True)
    pd.DataFrame(objects).to_csv(out/'per_included_object.csv',index=False)
    pd.DataFrame(references).to_csv(out/'per_reference.csv',index=False)
    pd.DataFrame(point_records).to_csv(out/'unique_export_center_scores.csv',index=False)
    result=dict(status='numeric_complete',utc=datetime.now(timezone.utc).isoformat(),version=version,
                protocol_sha256=sha(HERE/'SPATIAL_INPUT_BINDING.json'),script_sha256=sha(__file__),self_test=self_test(),
                reference_versions=len(references),animals=len(metas),included_objects=len(objects),
                independent_raster_checks=checks,all_region_coverage_areas_independently_verified=True,
                source_aggregate_sha256=sha(aggregate/'COMPLETE.json'),source_heatmap_audit_sha256=sha(heatdir/'report_audit.json'),
                outcomes_read=False,threshold_selected=False,probabilities_changed=False,
                outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(out/'COMPLETE.json',result);print(json.dumps({'output':str(out),'status':'numeric_complete'}))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--version',choices=['v6'])
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    elif args.version:main(args.version)
    else:parser.error('Choose --self-test or --version')
