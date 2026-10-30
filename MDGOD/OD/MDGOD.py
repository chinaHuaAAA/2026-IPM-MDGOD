from time import perf_counter
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.linalg import eigh
from .GB import granular_balls

@njit(cache=True)
def _cvm(a,na,b,nb):

    i=0;j=0;previous_h=0.0;previous_g=0.0;first=True;area=0.0
    while i<na or j<nb:
        if j>=nb or (i<na and a[i]<=b[j]):value=a[i]
        else:value=b[j]
        while i<na and a[i]<=value:i+=1
        while j<nb and b[j]<=value:j+=1
        fa=i/na;fb=j/nb
        h=(na*fa+nb*fb)/(na+nb)
        g=(fa-fb)**2
        if first:

            previous_g=g;first=False
        area+=.5*(g+previous_g)*(h-previous_h)
        previous_h=h;previous_g=g
    return np.sqrt(max(0.0,area*na*nb/(na+nb)**2))


@njit(cache=True)
def _spectral_matrix(values,lengths):
    n=len(lengths);out=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            out[i,j]=_cvm(values[i],lengths[i],values[j],lengths[j])
            out[j,i]=out[i,j]
    return out


def warmup():

    _spectral_matrix(np.array([[0.,1.],[0.,2.]]),np.array([2,2]))


def normalize(matrix):
    low=float(matrix.min());spread=float(matrix.max()-low)

    scale=spread if spread>=10*np.finfo(float).eps else 1.0
    result=(matrix-low)/scale
    np.fill_diagonal(result,0)
    return result


def _spectral_values(points,sigma):
    n=len(points)
    if n<2:return np.array([0.])
    distances=squareform(pdist(points)) if n>100 else cdist(points,points)
    adjacency=(distances<=sigma).astype(float)
    np.fill_diagonal(adjacency,0)
    if not adjacency.any():
        positive=distances[distances>0]
        if positive.size:
            adjacency=(distances<=positive.min()).astype(float)
            np.fill_diagonal(adjacency,0)
    degrees=adjacency.sum(axis=1)
    inverse=np.zeros(n);mask=degrees>0
    inverse[mask]=1/np.sqrt(degrees[mask])
    laplacian=np.eye(n)-(inverse[:,None]*adjacency)*inverse[None,:]
    if n<=10:values=np.linalg.eigvalsh(laplacian)
    else:


        k=min(n-1,50)
        values=eigh(laplacian,subset_by_index=(0,k-1),eigvals_only=True,
                    driver='evr',check_finite=False,overwrite_a=True)
    return np.clip(np.sort(values),0,2)


class PreparedMDGOD:
    def __init__(self,data,seed=0,layers=5,eta=.05,block_rows=256):


        self.seed=int(seed);self.layers=[];self.spectra={}
        rng=np.random.RandomState(seed)
        std=float(np.std(data,axis=0).mean()) or 1.0
        levels=eta*np.linspace(0,1,layers)
        self.weights=np.exp(-2*levels);self.weights/=self.weights.sum()
        for layer in range(layers):
            x=data.copy() if layer==0 else data+rng.randn(*data.shape)*(levels[layer]*std)
            balls=granular_balls(x)
            centers=np.array([b.center for b in balls]);radii=np.array([b.radius_mean for b in balls])
            geometry=squareform(pdist(centers))+radii[:,None]+radii[None,:]
            np.fill_diagonal(geometry,0)
            distances=cdist(x,centers)
            local_sigma=float(np.median(distances))
            similarity=np.exp(-distances**2/(2*local_sigma**2)) if local_sigma>0 else (distances==0).astype(float)
            self.layers.append(dict(data=x,balls=balls,geometry=normalize(geometry),similarity=similarity))

    def prepare_sigma(self,sigma):
        sigma=float(sigma)
        if sigma in self.spectra:return self.spectra[sigma]
        matrices=[]
        for layer in self.layers:
            spectra=[_spectral_values(layer['data'][ball.indices],sigma) for ball in layer['balls']]
            lengths=np.array([len(v) for v in spectra])
            values=np.zeros((len(spectra),int(lengths.max())))
            for i,v in enumerate(spectra):values[i,:len(v)]=v
            matrices.append(normalize(_spectral_matrix(values,lengths)))
        self.spectra[sigma]=matrices
        return matrices


    def score(self,sigma,alpha,damping=.1):
        spectral=self.prepare_sigma(sigma)
        output=np.zeros(len(self.layers[0]['data']))
        for weight,layer,spec in zip(self.weights,self.layers,spectral):
            fused=alpha*spec+(1-alpha)*layer['geometry']
            fused=(fused+fused.T)/2;np.fill_diagonal(fused,0)
            sums=fused.sum(axis=1);sums[sums==0]=1e-10
            transition=fused/sums[:,None]
            n=len(fused);phi=np.ones(n)/n;previous=np.ones(n);iteration=0
            while np.linalg.norm(previous-phi,1)>1e-4 and iteration<1000:
                previous=phi;phi=damping+(1-damping)*phi@transition;iteration+=1
            membership=(phi-phi.min())/(phi.max()-phi.min()+1e-8)
            similarity=layer['similarity']
            lower=np.min(np.maximum(1-similarity,membership),axis=1)
            upper=np.max(np.minimum(similarity,membership),axis=1)
            ratio=np.divide(lower,upper,out=np.zeros_like(lower),where=upper>0)
            output+=weight*ratio
        spread=float(output.max()-output.min())
        return (output-output.min())/spread if spread>0 else output


def MDGOD(data,sigma,alpha,random_seed=0,**kwargs):
    """计算每个样本的MDGOD异常得分。

    Parameters
    ----------
    data : numpy.ndarray，形状为(n_samples, n_features)
        仅包含特征的有限数值矩阵，一行一个样本，一列一个特征。
        调用前应完成逐列Min-Max归一化到[0, 1]，不能包含标签列。
        本函数不执行外部特征删除或归一化；实验main已实现论文预处理。
    sigma : float
        构建粒球内部邻接图的欧氏距离阈值。论文搜索范围为[0, 1]。
    alpha : float，范围[0, 1]
        谱距离的融合权重：alpha*谱距离 + (1-alpha)*几何距离。
        alpha=1仅使用谱距离，alpha=0仅使用几何距离。
    random_seed : int，默认0
        内部扰动的随机种子；固定输入、参数和种子可复现结果。
        论文重复实验使用0～9。
    layers : int，默认5，通过kwargs传入
        尺度总数，至少为1，包含一个无扰动的原始尺度。
        layers=1时只使用原始尺度。
    eta : float，默认0.05，通过kwargs传入
        基础扰动强度，0.05表示5%。第l层的高斯噪声标准差为
        eta*l/(layers-1)*mean(std(data, axis=0))；l从0开始。
        当全部特征标准差为0时，算法沿用基准尺度1；单尺度不加噪声。
    block_rows : int，默认256，通过kwargs传入
        保留的兼容接口；当前近似粒球实现不使用该值。

    Returns
    -------
    scores : numpy.ndarray，形状为(n_samples,)
        每个输入样本对应一个浮点异常得分，顺序与data的行顺序一致。
        得分越大越异常。非恒定得分被归一化到[0, 1]；恒定得分原样返回。
        AUC、AP和二值预测标签需在函数外另行计算。

    Example
    -------
    scores = MDGOD(X, sigma=0.5, alpha=0.8, random_seed=0,
                   layers=5, eta=0.05)
    """
    return PreparedMDGOD(data,seed=random_seed,**kwargs).score(sigma,alpha)


def run_independent(data,sigma,alpha,seed,**kwargs):
    """单参数独立运行；返回得分、耗时（秒）及计时分项。"""
    start=perf_counter()
    model=PreparedMDGOD(data,seed=seed,**kwargs)
    prepared_at=perf_counter()
    model.prepare_sigma(sigma)
    spectrum_at=perf_counter()
    scores=model.score(sigma,alpha)
    end=perf_counter()
    return scores,end-start,dict(Preparation=prepared_at-start,Spectral=spectrum_at-prepared_at,Scoring=end-spectrum_at)
