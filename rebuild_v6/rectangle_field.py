"""Exact half-open rectangle overlap field, with area-weighted display only.

Actual crop edges partition level0 coordinates. Measurements never use a rounded
64px footprint. No model or biological outcomes are imported by this module.
"""
from pathlib import Path
import math,sys
import numpy as np
import shapely

def require(ok,message):
    if not ok:raise ValueError(message)

def layout(coords,values,width,height):
    a=np.asarray(coords);p=np.asarray(values,dtype=np.float64)
    require(a.ndim==2 and a.shape[1]==3 and len(a)>0 and p.shape==(len(a),),'Invalid rectangle arrays')
    require(np.isfinite(a).all() and np.array_equal(a,a.astype(np.int64)),'Noninteger coordinates')
    a=a.astype(np.int64)
    require(np.all(a[:,:2]>=0) and np.all(a[:,2]>0) and width>0 and height>0,'Invalid footprint')
    require(np.all(a[:,0]+a[:,2]<=width) and np.all(a[:,1]+a[:,2]<=height),'Out-of-image rectangle')
    require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid probability')
    xs=np.unique(np.r_[0,width,a[:,0],a[:,0]+a[:,2]]);ys=np.unique(np.r_[0,height,a[:,1],a[:,1]+a[:,2]])
    x0=np.searchsorted(xs,a[:,0]);x1=np.searchsorted(xs,a[:,0]+a[:,2])
    y0=np.searchsorted(ys,a[:,1]);y1=np.searchsorted(ys,a[:,1]+a[:,2])
    return a,p,xs,ys,x0,x1,y0,y1

def field_from_rectangles(coords,values,width,height):
    a,p,xs,ys,x0,x1,y0,y1=layout(coords,values,width,height)
    total=np.zeros((len(ys)-1,len(xs)-1),dtype=np.float64);count=np.zeros(total.shape,dtype=np.int32)
    for l,r,t,b,v in zip(x0,x1,y0,y1,p,strict=True):
        total[t:b,l:r]+=v;count[t:b,l:r]+=1
    avg=np.divide(total,count,out=np.full(total.shape,np.nan),where=count>0)
    return dict(x_edges=xs,y_edges=ys,probability=avg,coverage=count)

def independent_difference(coords,values,width,height):
    a,p,xs,ys,x0,x1,y0,y1=layout(coords,values,width,height)
    total=np.zeros((len(ys),len(xs)),dtype=np.float64);count=np.zeros(total.shape,dtype=np.int64)
    for x,y,sign in [(x0,y0,1),(x1,y0,-1),(x0,y1,-1),(x1,y1,1)]:
        np.add.at(total,(y,x),sign*p);np.add.at(count,(y,x),sign)
    total=total.cumsum(0).cumsum(1)[:-1,:-1];count=count.cumsum(0).cumsum(1)[:-1,:-1]
    require(np.all(count>=0),'Negative coverage')
    return dict(x_edges=xs,y_edges=ys,probability=np.divide(total,count,out=np.full(total.shape,np.nan),where=count>0),coverage=count)

def check_fields(first,second):
    for k in ['x_edges','y_edges','coverage']:require(np.array_equal(first[k],second[k]),'Field identity/count differs: '+k)
    require(np.allclose(first['probability'],second['probability'],rtol=0,atol=1e-12,equal_nan=True),'Field probability difference')

def areas(field):return np.diff(field['y_edges'])[:,None]*np.diff(field['x_edges'])[None,:]
def covered_area(field):return float(areas(field)[field['coverage']>0].sum())

def display64(field):
    xs,ys=field['x_edges'],field['y_edges'];width=int(xs[-1]);height=int(ys[-1])
    dx=np.r_[np.arange(0,width,64),width];dy=np.r_[np.arange(0,height,64),height]
    xo=np.maximum(0,np.minimum(dx[1:,None],xs[None,1:])-np.maximum(dx[:-1,None],xs[None,:-1]))
    yo=np.maximum(0,np.minimum(dy[1:,None],ys[None,1:])-np.maximum(dy[:-1,None],ys[None,:-1]))
    valid=(field['coverage']>0).astype(float)
    numerator=yo@np.nan_to_num(field['probability'],nan=0.)@xo.T;denominator=yo@valid@xo.T
    average=np.divide(numerator,denominator,out=np.full(denominator.shape,np.nan),where=denominator>0)
    area=np.diff(dy)[:,None]*np.diff(dx)[None,:];fraction=denominator/area
    require(np.all((fraction>=0)&(fraction<=1)),'Display covered fraction invalid')
    require(math.isclose(float(denominator.sum()),covered_area(field),rel_tol=0,abs_tol=1e-7),'Display loses covered area')
    return average,fraction

def footprint_from_cells(field):
    xs,ys=field['x_edges'],field['y_edges'];pieces=[]
    for j,row in enumerate(field['coverage']>0):
        delta=np.diff(np.r_[False,row,False].astype(np.int8));left=np.where(delta==1)[0];right=np.where(delta==-1)[0]
        for a,b in zip(left,right,strict=True):pieces.append(shapely.box(xs[a],ys[j],xs[b],ys[j+1]))
    footprint=shapely.union_all(pieces)
    require(math.isclose(float(footprint.area),covered_area(field),rel_tol=0,abs_tol=1e-7),'Cell union area mismatch')
    return footprint

def footprint_from_windows(coords):
    a=np.asarray(coords);return shapely.union_all(shapely.box(a[:,0],a[:,1],a[:,0]+a[:,2],a[:,1]+a[:,2]))

def region_metrics(geometry,field,footprint):
    require(geometry.is_valid and not geometry.is_empty and geometry.area>0,'Invalid reference ROI')
    xs,ys=field['x_edges'],field['y_edges'];l,t,r,b=geometry.bounds
    inside=(xs[:-1]<r)&(xs[1:]>l);vertical=(ys[:-1]<b)&(ys[1:]>t)
    yy,xx=np.nonzero((field['coverage']>0)&vertical[:,None]&inside[None,:])
    cells=shapely.box(xs[xx],ys[yy],xs[xx+1],ys[yy+1]);weights=shapely.area(shapely.intersection(cells,geometry))
    area=float(weights.sum());independent=float(geometry.intersection(footprint).area)
    require(math.isclose(area,independent,rel_tol=1e-10,abs_tol=1e-7),'ROI exact partition/union disagreement')
    require(area<=geometry.area*(1+1e-10),'ROI overcoverage')
    mean=float(np.dot(weights,field['probability'][yy,xx])/area) if area else None
    return dict(roi_area_px2=float(geometry.area),covered_area_px2=area,coverage_fraction=area/geometry.area,
                mean_probability_covered=mean,unscored_area_px2=max(0.,float(geometry.area)-area),
                cells_intersecting_covered_roi=int(np.count_nonzero(weights)),independent_footprint_area_px2=independent)

def sample_centers(x,y,field):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float);xs,ys=field['x_edges'],field['y_edges']
    require(x.shape==y.shape and x.ndim==1 and np.isfinite(x).all() and np.isfinite(y).all(),'Invalid reference centers')
    inbounds=(x>=0)&(y>=0)&(x<xs[-1])&(y<ys[-1]);ix=np.searchsorted(xs,x,side='right')-1;iy=np.searchsorted(ys,y,side='right')-1
    scored=np.zeros(len(x),dtype=bool);scored[inbounds]=field['coverage'][iy[inbounds],ix[inbounds]]>0
    probability=np.full(len(x),np.nan);probability[scored]=field['probability'][iy[scored],ix[scored]]
    return probability,scored,inbounds

def center_metrics(centers,field):
    p,scored,inbounds=sample_centers(centers.center_x,centers.center_y,field);labels=centers.y_lesion.to_numpy()
    result=dict(unique_centers=len(p),out_of_map_centers=int((~inbounds).sum()))
    for name,label in [('lesion',1),('normal',0)]:
        allclass=labels==label;good=allclass&scored;n=int(allclass.sum());nc=int(good.sum())
        result.update({f'{name}_total_centers':n,f'{name}_covered_centers':nc,f'{name}_uncovered_centers':n-nc,
                       f'{name}_coverage_fraction':nc/n if n else None,f'{name}_mean_probability_covered':float(p[good].mean()) if nc else None})
    positive=p[scored&(labels==1)];negative=np.sort(p[scored&(labels==0)])
    result['auc_at_covered_export_centers']=float(np.sum((np.searchsorted(negative,positive,'left')+np.searchsorted(negative,positive,'right'))*.5)/(len(positive)*len(negative))) if len(positive) and len(negative) else None
    return result,p,scored

def self_test():
    # The irregular edges are the actual source408/409 crop edges, not rounded bins.
    c=np.array([[51,52,409],[307,52,408],[51,308,408]])
    f=field_from_rectangles(c,[.2,.8,.4],1024,1024);other=independent_difference(c,[.2,.8,.4],1024,1024);check_fields(f,other)
    footprint=footprint_from_cells(f);raw=footprint_from_windows(c)
    require(footprint.symmetric_difference(raw).area==0,'Exact raw/cell window union mismatch')
    p,good,bounds=sample_centers([50.999,51,459.999,460,714.999,715,1024],[52,52,60,60,60,60,60],f)
    require(np.array_equal(good,[False,True,True,True,True,False,False]),'Half-open edge membership')
    require(np.allclose(p[good],[.2,.5,.8,.8]),'Boundary field values')
    roi=shapely.box(0,0,1024,1024);m=region_metrics(roi,f,footprint)
    require(math.isclose(m['covered_area_px2'],raw.area,rel_tol=0,abs_tol=1e-9),'Coverage area mismatch')
    hole=shapely.box(51,52,460,461).difference(shapely.box(100,100,200,200))
    require(region_metrics(hole,f,footprint)['coverage_fraction']==1,'ROI hole wrong')
    missing=region_metrics(shapely.box(900,900,1000,1000),f,footprint)
    require(missing['coverage_fraction']==0 and missing['mean_probability_covered'] is None,'Unscored ROI filled')
    display,fraction=display64(f);require(np.any((fraction>0)&(fraction<1)),'No fractional edge display')
    # An aligned original-grid case agrees exactly with its 64px raster semantics.
    aligned=field_from_rectangles([[0,0,128],[64,0,128]],[.2,.8],256,128)
    a,fr=display64(aligned)
    require(np.allclose(a[:,:3],[[.2,.5,.8]]*2) and np.isnan(a[:,3]).all(),'Aligned display regression')
    require(np.array_equal(fr,[[1,1,1,0]]*2),'Aligned coverage regression')
    # Independent source-pixel ground truth for small nonaligned rectangles.
    small=np.array([[1,2,4],[3,1,5]]);values=np.array([.17,.83]);g=field_from_rectangles(small,values,10,9)
    yy,xx=np.indices((9,10));total=np.zeros((9,10));count=np.zeros((9,10))
    for (x,y,s),value in zip(small,values):
        mask=(xx>=x)&(xx<x+s)&(yy>=y)&(yy<y+s);count+=mask;total+=mask*value
    exact=np.divide(total,count,out=np.full((9,10),np.nan),where=count>0)
    samples,valid,_=sample_centers((xx+.5).ravel(),(yy+.5).ravel(),g)
    require(np.allclose(samples.reshape(9,10),exact,atol=1e-15,rtol=0,equal_nan=True),'Pixel ground truth differs')
    grid,frac=display64(g);require(math.isclose(grid[0,0],float(np.nanmean(exact)),abs_tol=1e-15),'Area display mean differs')
    require(math.isclose(frac[0,0],float(np.count_nonzero(count))/90,abs_tol=1e-15),'Clipped display cell area wrong')
    return dict(status='pass',checks=['409/408 nonaligned edges','independent prefix sum','raw rectangles vs partition union',
        'half-open point boundaries','ROI holes','zero-coverage NaN','fractional display coverage','aligned64 regression',
        'native pixel brute force','last clipped display cell'],study_probabilities_or_outcomes_read=False)

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
