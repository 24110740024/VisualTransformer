"""Render every v6 OOF map and separate fixed QuPath reference versions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

VIS=Path(__file__).resolve().parent;NEW=VIS.parent;HERE=NEW/'balance_candidate_v6'
os.environ['MPLCONFIGDIR']=str(HERE/'reports/.matplotlib_cache')
sys.path.insert(0,str(NEW/'annotation_tools_deps'))
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def reference_panel(ax,rgb,meta,reference):
    ax.imshow(rgb,extent=(0,meta['source_width'],meta['source_height'],0))
    geo=read(Path(reference['extraction_dir'])/'annotations.geojson')
    if not geo['features']:
        ax.text(.5,.5,'NO SAVED ROI\nNot whole-slide normal truth',transform=ax.transAxes,ha='center',va='center',fontsize=9,
                color='darkred',bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
    for feature in geo['features']:
        annotation=feature['properties'];geom=shapely.from_wkb((Path(reference['extraction_dir'])/annotation['wkb_path']).read_bytes())
        parts=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
        color='#00b7d6' if annotation['excluded_by_saved_tiler'] else '#ff6b00' if annotation['path_class'] is None else '#e00058'
        for part in parts:
            for ring in [part.exterior,*part.interiors]:
                p=np.asarray(ring.coords);ax.plot(p[:,0],p[:,1],color=color,lw=.65,linestyle='--' if annotation['excluded_by_saved_tiler'] else '-')
    ax.set_xlim(0,meta['source_width']);ax.set_ylim(meta['source_height'],0)
    ax.set_title(f"QuPath reference {reference['reference_version_index']} / {reference['reference_version_count']}",fontsize=9)


def main(audit_dir):
    from prepare_visuals import validate_numeric_audit, validate_visual_protocol
    validate_visual_protocol()
    validate_numeric_audit(audit_dir)
    state=read(HERE/'jobs/pipeline_status.json');assert state['phase']=='core_complete'
    aggregate=Path(state['aggregate']);assert aggregate.is_relative_to(HERE/'aggregate')
    ac=read(aggregate/'COMPLETE.json');dense=NEW/'scale_candidate_v4/dense_scaled';dc=read(dense/'COMPLETE.json')
    assert sha(dense/'COMPLETE.json')==ac['frozen']['complete_sha256']
    assert sha(dense/'coordinates_all21.csv')==ac['frozen']['manifest_sha256']
    from rectangle_field import field_from_rectangles,display64,covered_area
    source=VIS/'rectangle_field.py'
    frames=[]
    for r in ac['input_runs']:
        folder=Path(r['directory']);assert folder.is_relative_to(HERE/'predictions')
        assert sha(folder/'tile_predictions.csv')==r['predictions_sha256']
        f=pd.read_csv(folder/'tile_predictions.csv',float_precision='round_trip',usecols=['seed','case_id','source_sha256','x','y','crop_size','prob_lesion'])
        assert (f.seed==r['seed']).all();frames.append(f)
    allp=pd.concat(frames,ignore_index=True);keys=['case_id','source_sha256','x','y','crop_size']
    assert len(allp)==396096 and not allp.duplicated(keys+['seed']).any()
    wide=allp.pivot(index=keys,columns='seed',values='prob_lesion')
    assert len(wide)==132032 and list(wide.columns)==[42,43,44] and not wide.isna().any().any()
    assert np.isfinite(wide.to_numpy()).all() and (wide.to_numpy()>=0).all() and (wide.to_numpy()<=1).all()
    tiles=wide.mean(axis=1).rename('prob_lesion_equal_seed').reset_index()
    frozen=pd.read_csv(dense/'coordinates_all21.csv',usecols=keys).sort_values(keys).reset_index(drop=True)
    assert frozen.equals(tiles[keys].sort_values(keys).reset_index(drop=True))
    scorefile=aggregate/'case_scores_equal_seed_mean.csv';assert sha(scorefile)==ac['outputs_sha256'][scorefile.name]
    scores=pd.read_csv(scorefile,float_precision='round_trip').set_index('case_id')
    rf=NEW/'roi_reference_v1/FROZEN_REFERENCE.json'
    assert sha(rf)=='99617f80f044f9334bd84d2c5fe764f17d5574448817bf0bb07d4a063d308a0e'
    reference_freeze=read(rf)
    for path,digest in reference_freeze['files_sha256'].items():assert sha(path)==digest,path
    reference_audit=read(reference_freeze['accepted_report']);references=reference_audit['dense_reference_map']
    out=HERE/'reports'/datetime.now(timezone.utc).strftime('heatmaps_v6_%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    qc=[];contacts=[]
    for meta in dc['cases']:
        case=meta['case_id'];g=tiles[tiles.case_id==case]
        assert len(g)==meta['coordinate_count'] and set(g.source_sha256)=={meta['source_sha256']}
        mean=float(g.prob_lesion_equal_seed.mean());assert math.isclose(mean,float(scores.loc[case,'oof_prob_mean_raw']),abs_tol=1e-12)
        thumbnail=NEW/'dense_v1/cases'/case/'thumbnail.png';assert sha(thumbnail)==meta['thumbnail_sha256']
        rgb=np.asarray(Image.open(thumbnail).convert('RGB'))
        field=field_from_rectangles(g[['x','y','crop_size']].to_numpy(),g.prob_lesion_equal_seed.to_numpy(),meta['source_width'],meta['source_height'])
        heat,count=display64(field)
        assert heat.shape==rgb.shape[:2]
        colors=plt.colormaps['viridis'](np.nan_to_num(heat))[:,:,:3];overlay=rgb.astype(float)/255;covered=count>0
        alpha=.5*count[covered,None]
        overlay[covered]=(1-alpha)*overlay[covered]+alpha*colors[covered]
        refs=[r for r in references if r['case_id']==case]
        assert refs and all(r['source_sha256']==meta['source_sha256'] for r in refs)
        caseout=out/case;caseout.mkdir()
        Image.fromarray(rgb).save(caseout/'he_thumbnail.png');Image.fromarray(np.uint8(np.clip(overlay,0,1)*255)).save(caseout/'overlay.png')
        np.savez_compressed(caseout/'heatmap_arrays.npz',probability=heat,coverage=count,x_edges=field['x_edges'],y_edges=field['y_edges'],exact_probability=field['probability'],exact_coverage=field['coverage'])
        fig,axes=plt.subplots(1,3+len(refs),figsize=(3.15*(3+len(refs)),5.2),constrained_layout=True)
        axes[0].imshow(rgb);axes[0].set_title('H&E thumbnail')
        im=axes[1].imshow(np.ma.masked_invalid(heat),vmin=0,vmax=1,cmap='viridis',interpolation='nearest');axes[1].set_title('OOF lesion probability')
        axes[2].imshow(overlay);axes[2].set_title('Overlay (alpha=.5 x coverage)')
        for ax,ref in zip(axes[3:],refs):reference_panel(ax,rgb,meta,ref)
        for ax in axes:ax.axis('off')
        fig.colorbar(im,ax=axes,fraction=.018,pad=.015,label='Mean lesion probability; shared 0–1 scale')
        fig.suptitle(f'Balanced-sampling ViT v6 | {case} | {len(g):,} frozen tiles | score = {mean:.4f}\n'
            'Equal three-seed OOF predictions. Uncovered pixels are unscored. References do not alter probabilities.\n'
            'Reference: magenta = included class; orange = unclassified (saved tiler includes it); cyan dashed = excluded Region/Ignore',fontsize=10)
        fig.savefig(caseout/'panels.png',dpi=170);plt.close(fig)
        contacts.append((case,overlay,len(g),mean))
        qc.append({'case_id':case,'source_sha256':meta['source_sha256'],'n_tiles':len(g),'animal_score':mean,
            'reference_versions':len(refs),'roi_absent':all(r['all_annotation_count']==0 for r in refs),
            'map_pixels_covered':int(covered.sum()),'coverage_max':int(field['coverage'].max()),'exact_covered_area_px2':covered_area(field),
            'mean_scored_map_pixels_not_animal_score':float(np.nanmean(heat))})
    for start in (0,7,14):
        fig,axes=plt.subplots(2,4,figsize=(12.5,8.4),constrained_layout=True)
        for ax in axes.flat:ax.axis('off')
        for ax,(case,overlay,n,mean) in zip(axes.flat,contacts[start:start+7]):
            ax.imshow(overlay);ax.set_title(f'{case}\n{n:,} tiles; score={mean:.3f}',fontsize=9)
        fig.suptitle('Balanced-sampling ViT v6 | all-animal OOF probability overlays\nShared 0–1 scale; alpha=.5 x covered fraction; uncovered regions unscored',fontsize=12)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='viridis'),ax=list(axes.flat),fraction=.015,pad=.015)
        fig.savefig(out/f'overview_{start//7+1}.png',dpi=170);plt.close(fig)
    pd.DataFrame(qc).to_csv(out/'case_heatmap_QC.csv',index=False)
    note='# v6全21动物热图与固定参考\n\n全部132032窗口均保留，每窗三种子等权。实际408/409原图像素窗口的所有左右/上下边界划分精确矩形网格；每位置取覆盖窗口概率均值，不把边缘四舍五入到64格。精确float64概率、覆盖次数和边界保存在NPZ，用于空间核验。\n\n64倍缩图仅显示：每显示格按已覆盖精确面积对概率加权，未覆盖部分不当0概率；叠加透明度为0.5乘该格实际覆盖比例。统一viridis 0–1。动物主评分仍为窗口等权均值，不能替换成空间面积均值。\n\n保存全部24参考版本，无ROI不当全正常，不合并不同版本，不按结果修改轮廓或模型。未证明轮廓穷尽全片，不报告全片Dice/IoU。\n\n本显示针对不对齐64格的新窗口作必要几何适配，与v2/v3的512对齐窗口显示不同；对齐输入时退化为原规则。不是像素级分割，没有读Area/IL6、平滑、阈值选择或修改预测。\n'
    (out/'热图说明.md').write_text(note,encoding='utf-8')
    report={'status':'numeric_pass_awaiting_visual_review','created_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(__file__),'raster_source_sha256':sha(source),'cases':21,'frozen_tiles':132032,'predictions':396096,
        'source_aggregate_complete_sha256':sha(aggregate/'COMPLETE.json'),'reference_freeze_sha256':sha(rf),
        'numeric_audit_directory':str(Path(audit_dir).resolve()),'numeric_audit_sha256':sha(Path(audit_dir)/'report_audit.json'),
        'visual_protocol_sha256':sha(VIS/'VISUAL_PROTOCOL.json'),'csv_probability_parsing':'round_trip',
        'display_downsample':64,'probability_scale':[0,1],'overlay_alpha':.5,'smooth_or_change_probabilities':False,'measurement_field':'exact rectangle-edge partition','display_weighting':'exact covered area; alpha=.5*covered fraction',
        'checks':['synthetic overlap/NaN','all nine prediction SHA','all frozen coordinate keys','all 3 seeds per tile','official animal means','thumbnail SHA','all 240 reference hashes'],
        'outputs_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'report_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'cases':len(qc),'numeric_audit':'pass'}))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--audit-dir',required=True);args=ap.parse_args();main(args.audit_dir)
