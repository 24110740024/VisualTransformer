"""Append v6 to all completed versions only after independent numerical acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,os,sys
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
os.environ.setdefault('MPLCONFIGDIR',str(NEW/'.matplotlib_cache'))
from input_gate import checked,need,sha,read,close
from figures import rows,figure_a5,figure_a6,np,plt,roc_curve
from sklearn.metrics import roc_auc_score
from verify_groups import verify_group
import groups

PRIOR=NEW/'reports/verified_v5_standard_20260926T085652_505320Z'
KEYS=['path','label','class','source_id','case_id','split','legacy_wsi','historical_file','derivation','source_category','pixel_sha256','source_png_sha256']


def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(p,records):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def append_associations(previous,results,holm):
    need(len(previous)==32 and {r['version'] for r in previous}=={'v2','v3','v4','v5'},'Wrong prior association versions')
    need(len(results)==8 and len(holm)==4,'Incomplete new associations')
    main=[r for r in results if r['analysis']=='primary21']
    need(len(main)==4 and {(r['outcome'],r['statistic']) for r in main}=={(o,s) for o in ('Area','IL6') for s in ('Pearson','Spearman')},'Wrong primary family')
    seen=set();out=[dict(r) for r in previous]
    for r in results:
        key=r['analysis'],r['outcome'],r['statistic'];need(key not in seen,'Duplicate association');seen.add(key)
        primary=r['analysis']=='primary21'
        need(r['analysis'] in ('primary21','secondary18_fixed_alias_history_exclusions'),'Unplanned analysis')
        need(r['n_animals']==(21 if primary else 18),'Wrong biological n')
        hh=[a for a in holm if a['outcome']==r['outcome'] and a['statistic']==r['statistic']]
        need(len(hh)==1,'Missing/duplicate Holm key');h=hh[0]
        ci=r.get('paired_animal_percentile95_CI',{})
        if primary:
            close(r['coefficient'],h['coefficient']);close(r['p_two_sided_nominal'],h['p_nominal'])
            need(set(ci)>={'low','high'},'Missing primary confidence interval')
        out.append(dict(version='v6',analysis=r['analysis'],outcome=r['outcome'],method=r['statistic'],n=r['n_animals'],
            coefficient=r['coefficient'],p_nominal=r['p_two_sided_nominal'],p_holm_family4=h['p_holm_family4'] if primary else '',
            ci_low=ci.get('low',''),ci_high=ci.get('high','')))
    need(len(seen)==8,'Missing association');return out


def all_roc(prior,metrics,evaluation,a6,out):
    specs=list(prior['prediction_files_sha256'].items())
    need(len(specs)==5 and len(metrics)==6,'Wrong classification-version count')
    specs.append((str(evaluation/'test_predictions.csv'),a6['prediction_files_sha256']['test']))
    first=None;records={};fig,ax=plt.subplots(figsize=(8.6,6.8));fig.subplots_adjust(top=.84,bottom=.2)
    for (path,digest),r,color in zip(specs,metrics,['#999999','#0072B2','#CC79A7','#D55E00','#009E73','#E69F00'],strict=True):
        path=Path(path);need(sha(path)==digest,'Test predictions changed')
        rr=rows(path);need(len(rr)==8704 and len({v['source_id'] for v in rr})==4,'Wrong A6 cohort')
        ident=[tuple(v[k] for k in KEYS) for v in rr]
        if first is None:first=ident
        else:need(ident==first,'Cross-version test identity mismatch')
        y=np.array([int(v['y_lesion']) for v in rr]);p=np.array([float(v['prob_lesion']) for v in rr])
        close(roc_auc_score(y,p),r['auc']);fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,c=color,lw=1.6,label=f"{r['version']}: {float(r['auc']):.4f}");records[str(path)]=digest
    ax.plot([0,1],[0,1],'--',color='#aaaaaa',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    fig.suptitle('A6 | All six classification pipelines',fontsize=14,y=.97)
    fig.text(.5,.90,'Same 8,704 patches / 4 already examined held-out WSI',ha='center',fontsize=10)
    fig.text(.5,.09,'Recipes differ in backbone, fitting, preprocessing and precision; all results retained.',ha='center',fontsize=9)
    fig.text(.5,.05,'Descriptive comparison; no architecture superiority claim or new external validation.',ha='center',fontsize=9)
    for ext in ('png','svg'):fig.savefig(out/f'a6_all_six_models.{ext}',dpi=250)
    plt.close(fig);return records


def run(audit_dir):
    bound=checked(audit_dir);statistics=bound['statistics'];ev=bound['evaluation'];a6=bound['a6']
    prior=read(PRIOR/'REPORT_COMPLETE.json');review=read(PRIOR/'REVIEW_COMPLETE.json')
    need(review['status']=='verified_figures_complete' and review['evidence_sha256'][str((PRIOR/'REPORT_COMPLETE.json').relative_to(NEW))]==sha(PRIOR/'REPORT_COMPLETE.json'),'Prior report not accepted')
    for n in ('a6_all_five_models.csv','all_associations_v2_v3_v4_v5.csv'):
        need(sha(PRIOR/n)==prior['outputs_sha256'][n],'Prior table changed')
    joined=rows(statistics/'joined21.csv');need(len(joined)==21 and len({r['case_id'] for r in joined})==21,'Wrong animal n')
    results=read(statistics/'association_results.json');holm=rows(Path(audit_dir)/'holm_primary4.csv')
    associations=append_associations(rows(PRIOR/'all_associations_v2_v3_v4_v5.csv'),results,holm)
    summary=read(ev/'evaluation_summary.json');m=summary['test'];ci=summary['bootstrap']['source_wsi_cluster']['percentile_95_ci']
    metrics=rows(PRIOR/'a6_all_five_models.csv')
    metrics.append(dict(version='Balanced-sampling ViT v6',auc=m['auc_lesion'],ci_low=ci[0],ci_high=ci[1],
        **{k:m.get(k) for k in ('accuracy','precision','recall','specificity','f1','TN','FP','FN','TP')}))
    out=NEW/'reports'/datetime.now(timezone.utc).strftime('verified_v6_standard_%Y%m%dT%H%M%S_%fZ');out.mkdir(exist_ok=False)
    try:
        plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
        figure_a5(joined,results,holm,out);figure_a6(ev,a6,out)
        hashes=all_roc(prior,metrics,ev,a6,out)
        group=groups.main('v6',audit_dir);group_audit=verify_group(group)
        write(out/'group_independent_audit.json',group_audit)
        table(out/'a6_all_six_models.csv',metrics);table(out/'all_associations_v2_v3_v4_v5_v6.csv',associations)
        table(out/'v6_all21_animal_scores.csv',joined)
        text=['# v6平衡采样ViT：完整结果与标准图表','',
            '本报告在生产全部完成和独立审计通过后生成。A6种子42，A5三外折×三种子全参数训练；各自train固定类别0.5/来源sqrt(n)采样，无权重交叉熵。保持408/409尺度与132032中心，396096预测先种子内评分再三种子等权，21动物为统计单位。','',
            '| 分类管线 | AUC | 4来源簇95%CI |','|---|---:|---|']
        for r in metrics:text.append(f"| {r['version']} | {float(r['auc']):.6f} | {float(r['ci_low']):.6f}–{float(r['ci_high']):.6f} |")
        text+=['','| v6主分析（21动物） | r/ρ | 名义p | Holm四检验p | 95%CI |','|---|---:|---:|---:|---|']
        for r in associations:
            if r['version']=='v6' and r['analysis']=='primary21':text.append(f"| {r['outcome']} {r['method']} | {float(r['coefficient']):.6f} | {float(r['p_nominal']):.6g} | {float(r['p_holm_family4']):.6g} | {float(r['ci_low']):.6f}–{float(r['ci_high']):.6f} |")
        text+=['','全部40项主/辅助相关见CSV，辅助18例不替代21例主分析；112项删除诊断均保留，不能据此删动物或组。组内去均值仅描述：']
        for r in rows(Path(audit_dir)/'within_group_descriptive.csv'):text.append(f"- {r['outcome']}: {r['coefficient'] or 'undefined'}；{r['inference']}。")
        text+=['',f'[分组图与原定全部四对比](<{(group/"分组结果与局限.md").as_posix()}>): WT2×2及n=3+3对比已独立复核，p未校正；O3/RH场景不是纯臭氧效应，G9仅描述。','',
            'a5_primary21_v6.png/svg为两项主相关；a6_classification_v6.png/svg为ROC/混淆矩阵；a6_all_six_models.png/svg保留所有六个分类方案。需另做实际视觉验收，不因脚本完成就宣称图件已看过。',
            '', '只有4个分类测试来源、21关联动物。追加方案在先前测试/结局结果已知后固定，不是全新盲法外部测试；不能按最终AUC/r/热图筛全局赢家后宣称每动物完全独立。不同配方不支持纯架构或纯预训练效应归因。区间以已拟合评分为条件，不包括重新训练不确定性。',
            '', '独立审计核对全部训练记录/采样方案、A6/Bootstrap、全部396096预测及63种子/21等权均值、八相关/四区间和固定稳健性。没有替换生产权重、重选阈值或改人工标签。',
            '', '热图/空间核验使用v6_visualization_v1另行绑定的实际408/409精确足迹，需全部21动物/24参考和三双版本/无ROI规则。该标准图表报告不自动证明热图空间核验完成，未标区域不作正常，无穷尽真值不报全片Dice/IoU。',
            '', 'Area校准、IL6单位及取材配对仍待作者记录；S7/S8归师兄，脂质ROC不计入ViT自动完成。正文/SI整合按fallback_gap_audit_20260926逐项销项，旧LASSO已退役，不为补旧图外观重启。',
            '', f'[Phikon许可及传播说明](<{(NEW/"external_assets_v1/phikon/LICENSE_USE_NOTE.md").as_posix()}>): Phikon比较项的非商业学术及传播条件随交付保留；v6使用已核验ImageNet初始化，未混入下载中的外部感染小鼠图片。']
        (out/'v6标准图表与范围.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
        write(out/'REPORT_COMPLETE.json',dict(status='numeric_complete_visual_pending',utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(HERE/'PROTOCOL.json'),numeric_audit_directory=str(Path(audit_dir).resolve()),
            numeric_audit_complete_sha256=sha(Path(audit_dir)/'COMPLETE.json'),statistics_directory=str(statistics),
            group_directory=str(group),all_six_test_identities_match=True,prediction_files_sha256=hashes,
            new_animal_scores=21,new_predictions=396096,seed_replicates=3,independent_oof_models_per_animal=3,
            outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
            remaining=['Actual standard/group image visual review','Separate full heatmaps/spatial review','Full manuscript integration and author records']))
        print(json.dumps({'output':str(out),'group':str(group)},ensure_ascii=False))
    except Exception as exc:
        write(out/'FAILED.json',dict(error=repr(exc),preserve_partial_outputs=True));raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--audit-dir',type=Path,required=True)
    run(ap.parse_args().audit_dir)
