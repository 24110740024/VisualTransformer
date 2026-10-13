"""One-time CPU QA: real coordinate identities, synthetic probabilities only."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,sys
import numpy as np
import pandas as pd
from prepare_visuals import HERE,NEW,V6,read,sha,save,require,self_test,dependencies,validate_numeric_audit,validate_visual_protocol


def rejected(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Malformed prediction input accepted')


def main():
    require(not (HERE/'VISUAL_PROTOCOL.json').exists(),'Do not re-freeze')
    baseline=NEW/'v4_visualization_v1'
    require(sha(HERE/'rectangle_field.py')==sha(baseline/'rectangle_field.py'),'Geometry mathematics changed')
    qa=self_test()
    source=(HERE/'render_heatmaps.py').read_text(encoding='utf-8')
    original=(baseline/'render_heatmaps.py').read_text(encoding='utf-8')
    # Execute the actual renderer validation/aggregation block, with full-size synthetic values.
    block=source[source.index('    allp=pd.concat'):source.index('    scorefile=')]
    oldblock=original[original.index('    allp=pd.concat'):original.index('    scorefile=')]
    require(ast.dump(ast.parse('def check(frames,dense):\n'+block))==ast.dump(ast.parse('def check(frames,dense):\n'+oldblock)),'Three-seed aggregation changed')
    ns={'pd':pd,'np':np};exec(compile('def check(frames,dense):\n'+block+'    return tiles\n',str(HERE/'render_heatmaps.py'),'exec'),ns)
    dense=NEW/'scale_candidate_v4/dense_scaled'
    keys=['case_id','source_sha256','x','y','crop_size']
    coordinates=pd.read_csv(dense/'coordinates_all21.csv',usecols=keys)
    probability=.1+(np.arange(len(coordinates))%100)*.005
    frames=[coordinates.assign(seed=seed,prob_lesion=probability+.05*i) for i,seed in enumerate((42,43,44))]
    combined=ns['check'](frames,dense)
    expected=coordinates.assign(expected=probability+.05)
    paired=combined.merge(expected,on=keys,validate='one_to_one')
    require(len(paired)==132032 and np.allclose(paired.prob_lesion_equal_seed,paired.expected,rtol=0,atol=2e-16),'Independent arithmetic mean disagrees')
    corruptions=['missing','duplicate','seed','nan','outside','source','coordinate']
    for corruption in corruptions:
        altered=[frames[0].copy(),frames[1],frames[2]]
        if corruption=='missing':altered[0]=altered[0].iloc[:-1]
        elif corruption=='duplicate':altered[0]=pd.concat([altered[0].iloc[:-1],altered[0].iloc[[0]]],ignore_index=True)
        elif corruption=='seed':altered[0].loc[0,'seed']=41
        elif corruption=='nan':altered[0].loc[0,'prob_lesion']=np.nan
        elif corruption=='outside':altered[0].loc[0,'prob_lesion']=1.5
        elif corruption=='source':altered[0].loc[0,'source_sha256']='wrong-source'
        else:altered[0].loc[0,'x']+=1
        rejected(lambda:ns['check'](altered,dense))
    state=read(V6/'jobs/pipeline_status.json')
    require(state['phase']!='core_complete','Preparation must not inspect actual results')
    rejected(lambda:validate_numeric_audit(NEW/'v6_postprocessing_v1/nonexistent'))
    qa.update(utc=datetime.now(timezone.utc).isoformat(),synthetic_probability_rows=396096,synthetic_tile_means=132032,
        malformed_predictions_rejected=corruptions,aggregation_ast_unchanged=True,geometry_byte_identical=True,
        actual_incomplete_state_refused=True,actual_candidate_probabilities_or_outcomes_read=False)
    save(HERE/'CPU_QA.json',qa)
    save(HERE/'VISUAL_PROTOCOL.json',dict(protocol='V6_EXACT_RECTANGLE_VISUALIZATION_V1',utc=datetime.now(timezone.utc).isoformat(),
        files_sha256=dependencies(),cpu_qa_sha256=sha(HERE/'CPU_QA.json'),measurement='Unchanged v4 exact rectangle mathematics; independent v6 input binding',
        geometry_readonly_source=str(dense),prior_results_known=True,selection=False))
    validate_visual_protocol()
    rejected(lambda:validate_numeric_audit(NEW/'v6_postprocessing_v1/nonexistent'))
    require('torch' not in sys.modules,'GPU library imported')
    save(HERE/'POSTFREEZE_CHECK.json',dict(status='passed',protocol_sha256=sha(HERE/'VISUAL_PROTOCOL.json'),candidate_phase=state['phase'],
        incomplete_state_refused=True,gpu_imported=False,actual_candidate_probabilities_or_outcomes_read=False))
    print(json.dumps({'status':'prepared_and_frozen','protocol_sha256':sha(HERE/'VISUAL_PROTOCOL.json')}))


if __name__=='__main__':main()
