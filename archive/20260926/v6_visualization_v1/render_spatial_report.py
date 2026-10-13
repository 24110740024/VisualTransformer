"""Audit saved point diagnostics and render all reference versions, CPU only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)
def f(x,percent=False):return '缺失' if pd.isna(x) else f'{x*100:.2f}%' if percent else f'{x:.4f}'


def main(directory):
    directory=Path(directory).resolve();assert directory.is_relative_to(HERE)
    complete=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    assert complete['status']=='numeric_complete'
    assert complete['version']=='v6'
    assert complete['protocol_sha256']==sha(HERE/'SPATIAL_INPUT_BINDING.json')
    for path,digest in complete['outputs_sha256'].items():assert sha(directory/path)==digest,path
    refs=pd.read_csv(directory/'per_reference.csv',float_precision='round_trip');points=pd.read_csv(directory/'unique_export_center_scores.csv',float_precision='round_trip')
    assert len(refs)==24 and refs.case_id.nunique()==21
    assert not points.duplicated(['case_id','reference_version_index','center_x','center_y']).any()
    checks=[]
    for row in refs.to_dict('records'):
        p=points[(points.case_id==row['case_id'])&(points.reference_version_index==row['reference_version_index'])]
        assert len(p)==row['unique_centers']
        good=p[p.covered==True]
        assert good.heatmap_probability.notna().all() and p[p.covered==False].heatmap_probability.isna().all()
        for name,label in [('lesion',1),('normal',0)]:
            allclass=p[p.y_lesion==label];covered=good[good.y_lesion==label]
            assert len(allclass)==row[f'{name}_total_centers'] and len(covered)==row[f'{name}_covered_centers']
            if len(allclass):close(len(covered)/len(allclass),row[f'{name}_coverage_fraction'])
            if len(covered):close(math.fsum(covered.heatmap_probability)/len(covered),row[f'{name}_mean_probability_covered'])
        if good.y_lesion.nunique()==2:
            close(roc_auc_score(good.y_lesion,good.heatmap_probability),row['auc_at_covered_export_centers'])
        else:assert pd.isna(row['auc_at_covered_export_centers'])
        if row['roi_present']:
            close(row['covered_area_px2']/row['roi_area_px2'],row['coverage_fraction'])
            close(row['covered_area_px2']+row['unscored_area_px2'],row['roi_area_px2'])
        checks.append(dict(case_id=row['case_id'],reference_version=int(row['reference_version_index']),saved_points_verified=len(p)))
    refs=refs.sort_values(['case_id','reference_version_index']).reset_index(drop=True)
    labels=[f"{r.case_id}  [ref {r.reference_version_index}]" for r in refs.itertuples()]
    y=np.arange(len(refs));colors=['#0072B2' if n==1 else '#D55E00' for n in refs.reference_version_index]
    fig,axes=plt.subplots(1,3,figsize=(13.5,10.5),sharey=True)
    for ax,column,title in zip(axes,['coverage_fraction','mean_probability_covered','auc_at_covered_export_centers'],
            ['ROI area covered by frozen windows','Mean probability in covered ROI','AUC at covered labeled export centers']):
        values=refs[column].to_numpy();valid=np.isfinite(values)
        ax.scatter(values[valid],y[valid],c=np.array(colors)[valid],s=30)
        for yy in y[~valid]:ax.text(.5,yy,'No saved ROI / not estimable',ha='center',va='center',fontsize=7,color='.4')
        ax.set(xlim=(-.02,1.02),ylim=(len(refs)-.4,-.6),title=title,xlabel='Fraction' if column=='coverage_fraction' else 'Probability' if column=='mean_probability_covered' else 'Conditional descriptive AUC')
        ax.set_yticks(y);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',labelsize=8)
    axes[0].set_yticklabels(labels)
    fig.suptitle('v6 fixed-reference spatial diagnostics | 21 animals, all 24 reference versions',fontsize=14,y=.99)
    fig.text(.5,.044,'Blue: reference 1; orange: reference 2. Versions are retained separately, not independent animals.',ha='center',fontsize=9)
    fig.text(.5,.021,'Uncovered points/area reported separately. No threshold, whole-slide Dice, outcome tuning or probability repair.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.96))
    for ext in ('png','svg'):fig.savefig(directory/f'spatial_diagnostics.{ext}',dpi=180)
    plt.close(fig)
    lines=['# v6固定参考空间核验：描述性结果','',
      '已按冻结方案检查21只动物的全部24个参考版本。23个有纳入ROI的参考属于20只动物；30-40-80-N-3没有保存ROI。3只动物各两版独立列示，不跨版本合并，也不选更好的一版。诊断规则在v2结果及热图已知后制定，本次按实际408/409窗口边界单独绑定v6预测；64倍图仅作面积加权显示，空间数值不使用四舍五入的显示边界，属于描述性诊断。','',
      '全部132032个窗口的三种子等权概率通过二维差分算法独立重建，与已保存热图逐像素一致。每个ROI与实际408/409窗口边界构成的精确矩形单元相交计面积；覆盖面积还用原始窗口几何并集独立复核，全部一致。没有把未覆盖部分的概率赋0。','',
      '|动物 / 参考|ROI取样覆盖率|已覆盖ROI平均概率|Lesion点覆盖/总数|Normal点覆盖/总数|覆盖导出点AUC|',
      '|---|---:|---:|---:|---:|---:|']
    for r in refs.to_dict('records'):
        lines.append(f"|{r['case_id']} / {int(r['reference_version_index'])}|{f(r['coverage_fraction'],True)}|{f(r['mean_probability_covered'])}|{int(r['lesion_covered_centers'])}/{int(r['lesion_total_centers'])}|{int(r['normal_covered_centers'])}/{int(r['normal_total_centers'])}|{f(r['auc_at_covered_export_centers'])}|")
    lines+=['','## 怎样解读','',
      '- ROI取样覆盖率说明冻结窗口是否覆盖了保存参考区域，不是模型敏感度。覆盖范围由预测前冻结的组织取样决定，没有为本次结果扩大或缩小。',
      '- ROI平均概率是已覆盖位置的连续模型响应，不是正确率、病变面积比例、经过校准的置信度或原动物评分。低响应与取样缺口均保留。面积单位为源WSI像素平方，没有换算mm²。',
      '- 导出点AUC使用原始QuPath Lesion/Normal参考类别。Normal、Normal-Far、Normal-Red的相同中心确定性合并；独立工程中心二分类冲突为0。所有未覆盖点在分母和逐点表中保留，AUC仅条件于覆盖点，不能忽略这种选择限制。',
      '- 导出中心在原图连续坐标上按半开窗口边界定位；概率来自408/409窗口的重叠平均，和直接对原224 PNG重新推理不同，因此不能当作新增A6分类AUC。20只动物的对应模型均为外折模型，但仍属本研究内部交叉拟合材料，不是新外部测试。',
      '- 切块概率场不是像素级分割。没有证实轮廓穷尽全片，不将ROI外全部当正常；不报告全片Dice/IoU、二值正确率或patch独立p/CI。不同参考版本、多个ROI和导出点不增加动物样本量。',
      '- 本检查不读取Area/IL6，不改坐标、评分或模型。所有版本及弱响应都保留；不能据该诊断反向调参或只报告更好的版本。',
      '- 图中的颜色只标示参考版本遍历顺序，不代表旧/新时间关系；第三只30-30-120-N的一版未分类大对象仍按保存tiler保留。','',
      '原始数值：per_reference.csv、per_included_object.csv、unique_export_center_scores.csv。COMPLETE.json记录冻结协议和所有输入/输出身份；report_audit.json独立复核逐点覆盖、类别均值及AUC。视觉验收另记。',
      '本报告属于已完成的平衡采样候选v6。v2全部结果原样保留；不按r/AUC/热图择优隐瞒，也不将此比较称为新增外部验证。']
    (directory/'空间核验说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(status='independent_point_audit_pass_awaiting_visual_review',utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
      complete_sha256=sha(directory/'COMPLETE.json'),independent_auc='sklearn ROC-AUC vs pair-count source computation',checks=checks,
      outputs_sha256={name:sha(directory/name) for name in ['spatial_diagnostics.png','spatial_diagnostics.svg','空间核验说明.md']})
    (directory/'report_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'pass','output':str(directory),'references':len(refs),'point_records':len(points)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();main(args.directory)
