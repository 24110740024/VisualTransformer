"""Candidate figures reuse established axes with explicit v6 single-head labels and inputs."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)

def figure_a5(joined,results,holm,out):
    groups=sorted({r['case_id'].rsplit('-',1)[0] for r in joined})
    palette=['#0072B2','#56B4E9','#D55E00','#CC79A7','#009E73','#E69F00','#333333']
    markers=['o','s','^','D','v','P','X']
    x=np.array([float(r['pathology_score']) for r in joined])
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.73,wspace=.29)
    fig.suptitle('Balanced-sampling ViT | v6 | A5 animal associations',fontsize=15,y=.975)
    fig.text(.5,.92,'All 21 animals; out-of-fold scores; seeds 42, 43 and 44 equally weighted',ha='center',fontsize=10)
    for ax,outcome in zip(axes,('Area','IL6')):
        y=np.array([float(r[outcome]) for r in joined])
        for group,color,marker in zip(groups,palette,markers):
            subset=[r for r in joined if r['case_id'].rsplit('-',1)[0]==group]
            assert len(subset)==3
            ax.scatter([float(r['pathology_score']) for r in subset],[float(r[outcome]) for r in subset],
                       c=color,marker=marker,s=60,label=group,edgecolor='white',linewidth=.6,zorder=3)
        slope,intercept=np.polyfit(x,y,1);span=np.array([x.min(),x.max()])
        ax.plot(span,slope*span+intercept,'--',c='#777777',lw=1)
        r=next(r for r in results if r['analysis']=='primary21' and r['outcome']==outcome and r['statistic']=='Pearson')
        h=next(h for h in holm if h['outcome']==outcome and h['statistic']=='Pearson')
        ci=r['paired_animal_percentile95_CI']
        ax.set_title(f"{'Area' if outcome=='Area' else 'IL-6'}: Pearson r = {r['coefficient']:.3f}\n"
            f"Nominal p = {r['p_two_sided_nominal']:.4f}; Holm p = {float(h['p_holm_family4']):.4f}\n"
            f"Paired-animal 95% CI: {ci['low']:.3f} to {ci['high']:.3f}",fontsize=10,pad=12)
        ax.set_xlabel('Out-of-fold mean lesion probability')
        ax.set_ylabel(('Area' if outcome=='Area' else 'IL-6')+' (source value)')
        ax.set_xlim(max(0,x.min()-.04),min(1,x.max()+.04));ax.margins(y=.10)
        ax.grid(axis='y',color='#e5e5e5',lw=.6)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,title='Recorded case-key groups; 3 animals per group',loc='lower center',
               bbox_to_anchor=(.5,.10),ncol=4,frameon=False,fontsize=9,title_fontsize=9)
    fig.text(.5,.056,'Holm family: Area / IL-6 × Pearson / Spearman (4 tests). Dashed line: descriptive fit.',ha='center',fontsize=9)
    fig.text(.5,.025,'Area calibration and IL-6 units remain unverified. Bootstrap conditions on the fitted scores.',ha='center',fontsize=9)
    for ext in ('png','svg'):fig.savefig(out/f'a5_primary21_v6.{ext}',dpi=250)
    plt.close(fig)


def figure_a6(evaluation,audit,out):
    summary=read(evaluation/'evaluation_summary.json');test=rows(evaluation/'test_predictions.csv')
    assert sha(evaluation/'test_predictions.csv')==audit['prediction_files_sha256']['test']
    y=np.array([int(r['y_lesion']) for r in test]);p=np.array([float(r['prob_lesion']) for r in test])
    m=summary['test'];ci=summary['bootstrap']['source_wsi_cluster']['percentile_95_ci']
    assert len(y)==8704 and len({r['source_id'] for r in test})==4
    for k,v in audit['recomputed_test_metrics'].items():
        if v is None:assert m[k] is None
        else:close(v,m[k])
    fpr,tpr,_=roc_curve(y,p)
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.4))
    fig.subplots_adjust(top=.79,bottom=.19,left=.07,right=.97,wspace=.33)
    fig.suptitle('Balanced-sampling ViT | v6 | A6 held-out patch classification',fontsize=14,y=.97)
    fig.text(.5,.90,'8,704 unique exported patches from 4 held-out source WSI; seed 42',ha='center',fontsize=10)
    ax=axes[0];ax.plot(fpr,tpr,c='#0072B2',lw=2,label=f"AUC = {m['auc_lesion']:.4f}")
    ax.plot([0,1],[0,1],'--',c='#aaaaaa',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate',title='A  Receiver operating characteristic')
    ax.legend(loc='lower right',frameon=False)
    ax=axes[1]
    if audit['validation_threshold'] is None:
        ax.axis('off');ax.text(.5,.5,'No finite validation threshold\nConfusion matrix undefined',ha='center',va='center')
        fig.text(.5,.07,'Only 4 test sources. All undefined threshold metrics remain explicitly undefined.',ha='center',fontsize=9)
        for ext in ('png','svg'):fig.savefig(out/f'a6_classification_v6.{ext}',dpi=250)
        plt.close(fig);return
    matrix=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]])
    ax.imshow(matrix,cmap='Blues',vmin=0)
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,f'{value:,}',ha='center',va='center',fontsize=17,color='white' if value>matrix.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Lesion'],yticklabels=['Normal','Lesion'],
           xlabel='Predicted class',ylabel='Fixed reference label',title='B  Confusion matrix')
    fig.text(.5,.08,f"WSI-cluster bootstrap 95% AUC CI: {ci[0]:.3f}–{ci[1]:.3f} | Validation-only threshold: {audit['validation_threshold']:.6f}",ha='center',fontsize=9)
    fig.text(.5,.035,'Only 4 independent test sources. Patch counts are not independent biological replicates.',ha='center',fontsize=9)
    for ext in ('png','svg'):fig.savefig(out/f'a6_classification_v6.{ext}',dpi=250)
    plt.close(fig)


