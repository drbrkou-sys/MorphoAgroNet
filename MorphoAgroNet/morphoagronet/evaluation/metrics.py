import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,classification_report,confusion_matrix

def classification_metrics(y_true,y_pred,prob=None):
    y=np.asarray(y_true);p=np.asarray(y_pred)
    prec,rec,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    out={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(prec),'macro_recall':float(rec),'macro_f1':float(f1)}
    if prob is not None:
        try:
            prob=np.asarray(prob);classes=np.arange(prob.shape[1]);onehot=np.eye(len(classes))[y]
            out['macro_roc_auc']=float(roc_auc_score(onehot,prob,average='macro'))
        except ValueError:out['macro_roc_auc']=None
    return out

def classwise(y,p,labels):return classification_report(y,p,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)

def confusion(y,p,classes):return confusion_matrix(y,p,labels=list(range(classes)))
