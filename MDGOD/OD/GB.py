from dataclasses import dataclass
import numpy as np


@dataclass
class Ball:
    indices: np.ndarray
    center: np.ndarray
    radius_sum: float
    radius_mean: float
    radius_max: float


def _ball(data,indices):
    points=data[indices];center=points.mean(axis=0)
    distances=np.sqrt(np.sum((points-center)**2,axis=1))
    return Ball(indices,center,float(distances.sum()),float(distances.mean()),float(distances.max()))


def _squared_distances_to_point(X,point,x_squared_norms=None):
    if x_squared_norms is None:x_squared_norms=np.einsum('ij,ij->i',X,X)
    point_squared_norm=np.dot(point,point)
    return np.maximum(x_squared_norms-2.0*np.dot(X,point)+point_squared_norm,0.0)


def _balanced_fallback_split(X):
    axis=int(np.argmax(np.var(X,axis=0)))
    order=np.argsort(X[:,axis],kind='stable')
    labels=np.zeros(len(X),dtype=np.int8)
    labels[order[len(X)//2:]]=1
    return labels


def _split_labels(X):
    if len(X)<2:raise ValueError('a ball needs at least two samples to split')
    X=np.asarray(X,dtype=float)
    norms=np.einsum('ij,ij->i',X,X)
    centroid=X.mean(axis=0)
    first=int(np.argmax(_squared_distances_to_point(X,centroid,norms)))
    first_distances=_squared_distances_to_point(X,X[first],norms)
    second=int(np.argmax(first_distances))
    if first==second or first_distances[second]<=np.finfo(float).eps:
        labels=_balanced_fallback_split(X)
    else:
        second_distances=_squared_distances_to_point(X,X[second],norms)

        labels=(second_distances<first_distances).astype(np.int8)
        if not np.any(labels==0) or not np.any(labels==1):labels=_balanced_fallback_split(X)
    return labels


def spilt_ball_2(data):
    labels=_split_labels(data[:,:-1])
    return [data[labels==0],data[labels==1]]


def _split(data,ball):


    if not hasattr(ball, '_children'):
        labels=_split_labels(data[ball.indices])
        ball._children=(_ball(data,ball.indices[labels==0]),_ball(data,ball.indices[labels==1]))
    return ball._children


def granular_balls(data):
    data=np.asarray(data,dtype=np.float64)
    if data.ndim!=2 or not len(data) or not data.shape[1] or not np.isfinite(data).all():
        raise ValueError('data must be a nonempty finite feature matrix')
    n=len(data)
    threshold=64 if np.sqrt(n)>64 else (8 if np.sqrt(n)>8 else int(np.sqrt(n)))
    balls=[_ball(data,np.arange(n))]
    while True:
        updated=[]
        for ball in balls:
            if len(ball.indices)>=max(2,threshold):
                children=_split(data,ball);left,right=children
                parent_density=ball.radius_sum or len(ball.indices)
                left_density=left.radius_sum or len(left.indices)
                right_density=right.radius_sum or len(right.indices)
                weighted=len(left.indices)/len(ball.indices)*left_density+len(right.indices)/len(ball.indices)*right_density
                if weighted<parent_density:
                    updated.extend(children);continue
            updated.append(ball)
        if len(updated)==len(balls):break
        balls=updated
    radii=[b.radius_max for b in balls if len(b.indices)>=2]
    detect=max(float(np.median(radii)),float(np.mean(radii))) if radii else 0.0
    while True:
        updated=[]
        for ball in balls:
            if len(ball.indices)>=2 and ball.radius_max>2*detect:updated.extend(_split(data,ball))
            else:updated.append(ball)
        if len(updated)==len(balls):break
        balls=updated
    return balls


def getGranularBall(data):

    data=np.asarray(data,dtype=np.float64)
    balls=granular_balls(data)
    return (np.array([b.center for b in balls]),
            [np.column_stack([data[b.indices],b.indices]) for b in balls],
            np.array([len(b.indices) for b in balls]),
            np.array([b.radius_mean for b in balls]),
            np.array([b.radius_max for b in balls]))
