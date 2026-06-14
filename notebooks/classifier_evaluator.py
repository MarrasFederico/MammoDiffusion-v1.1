#classi e procedure per misurare le prestazioni dei classificatori
from __future__ import annotations
from dataclasses import dataclass, field #field - per i default che possono essere dict/list nei dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  #per evitare crash su server senza display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support, roc_auc_score,
)


#raccoglie le metriche del classificatore, confusion matrix e path delle figure
@dataclass
class EvaluationResult:
    #metriche di base
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    roc_auc: Optional[float] = None #serve y_prob altrimenti è None
    #solo per classificazione binaria
    sensitivity: Optional[float] = None  #recall sulla classe malata
    specificity: Optional[float] = None  #recall sulla classe sana

    per_class: dict = field(default_factory=dict) #metriche per singola classe
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    figure_paths: list = field(default_factory=list) #path dei file PNG salvati

    #costruisce la stringa aggiungendo le righe opzionali solo se calcolate davvero
    def __str__(self) -> str:
        extra = ""
        if self.sensitivity is not None:
            extra += f"Sensitivity (pos): {self.sensitivity:.4f}\n"
        if self.specificity is not None:
            extra += f"Specificity (neg): {self.specificity:.4f}\n"
        if self.roc_auc is not None:
            extra += f"ROC-AUC:   {self.roc_auc:.4f}\n"
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1-Score:  {self.f1:.4f}\n"
            f"{extra}"
            f"---\n{self.classification_report}"
        )


    #costruisce il formato del dizionario da passare al logger per WandB
    #sono sempre presenti le 4 chiavi delle metriche base mentre le altre vengono aggiunte solo se calcolabili
    def to_dict(self) -> dict:
        d = {
            "Accuracy":  self.accuracy,
            "Precision": self.precision,
            "Recall":    self.recall,
            "F1":        self.f1,
        }
        if self.roc_auc is not None:
            d["ROC_AUC"] = self.roc_auc
        if self.sensitivity is not None:
            d["Sensitivity"] = self.sensitivity
        if self.specificity is not None:
            d["Specificity"] = self.specificity
        return d


#raccoglie l'esito della valutazione bilanciata a bootstrap (media +/- dev std su piu' round)
@dataclass
class BalancedBootstrapResult:
    n_rounds: int = 0
    n_positives: int = 0            #positivi usati ogni round (tutti)
    n_negatives_per_round: int = 0  #negativi campionati ogni round (= n_positives)
    replace: bool = False           #campionamento negativi con reimmissione (se ne mancano)
    means: dict = field(default_factory=dict)      #media per metrica
    stds: dict = field(default_factory=dict)       #deviazione standard per metrica
    per_round: list = field(default_factory=list)  #lista di dict (una riga per round)
    figure_paths: list = field(default_factory=list)

    def __str__(self) -> str:
        righe = [
            f"Bootstrap bilanciato: {self.n_rounds} round, "
            f"{self.n_positives} pos + {self.n_negatives_per_round} neg per round"
            + (" (neg con reimmissione)" if self.replace else "")
        ]
        for k in self.means:
            righe.append(f"  {k:14s}: {self.means[k]:.4f} +/- {self.stds[k]:.4f}")
        return "\n".join(righe)

    #chiavi piatte tipo "F1_mean"/"F1_std" comode per il logging
    def to_dict(self) -> dict:
        d = {}
        for k in self.means:
            d[f"{k}_mean"] = self.means[k]
            d[f"{k}_std"]  = self.stds[k]
        return d


"""
-calcola le metriche per valutare i classificatori, salvando la confusion matrix come PNG
-il parametro average controlla l'aggregazione delle metriche tra classi
-funziona per classificatori binari e multiclasse ma nel caso binario
 aggiunge sensitivity e specificity rispetto alla classe "positiva" (malata)
 inoltre la metrica ROC-AUC viene calcolata solo se si ricevono le y_prob
"""
class ClassifierEvaluator:

    FIGURES_SUBDIR = "figures" #cartella dove salvare le figure

    def __init__(self, output_dir: str | Path = "results", average: str = "weighted") -> None:
        self.output_dir  = Path(output_dir)
        self.figures_dir = self.output_dir / self.FIGURES_SUBDIR
        self.average     = average
        self.figures_dir.mkdir(parents=True, exist_ok=True)


    """
    -calcola tutte le metriche e salva la confusion matrix
    -pos_label - indica quale etichetta numerica corrisponde alla classe "malata"
     pos_label=0 dato che nel dataset sklearn breast cancer 0=malignant 
     ma nel nostro progetto la classe malata è posta a 1 quindi anche pos_label=1.
    """
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
                             label: str = "model", class_names: Optional[list] = None, pos_label: int = 1) -> EvaluationResult:

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true e y_pred devono avere la stessa lunghezza.")

        #ricava i label unendo y_true e y_pred così da avere l'ordine corretto
        #nella confusion matrix anche se y_pred non copre tutte le classi
        #per evitare errori su batch piccoli o classi rare
        labels = np.unique(np.concatenate([y_true, y_pred]))

        #calcolo metriche
        acc    = accuracy_score(y_true, y_pred)
        prec   = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        rec    = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        f1     = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        
        per_class             = self.per_class_metrics(y_true, y_pred, labels, class_names)#metriche per ogni classe
        sensitivity, specificity = self.binary_sens_spec(cm, labels, pos_label)
        roc_auc               = self.safe_roc_auc(y_true, y_prob, labels, pos_label)

        fig_path = self.plot_confusion_matrix(cm, label=label, class_names=class_names) #salva la CM e ritorna il path

        return EvaluationResult(
            accuracy=acc, precision=prec, recall=rec, f1=f1,
            roc_auc=roc_auc, sensitivity=sensitivity, specificity=specificity,
            per_class=per_class,
            confusion_matrix=cm, classification_report=report,
            figure_paths=[str(fig_path)],
        )


    
    #calcola precision/recall/f1/support per ogni singola classe
    def per_class_metrics(self, y_true, y_pred, labels, class_names: Optional[list]) -> dict:

        #calcola tutte le metriche assieme
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        #se class_names non è allineato con labels o non è passato, usa le labels numeriche
        nomi = class_names if class_names and len(class_names) == len(labels) \
            else [str(l) for l in labels]
        return {
            nome: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(s[i])}
            for i, nome in enumerate(nomi)
        }


    #calcola sensitivity e specificity dalla confusion matrix
    def binary_sens_spec(self, cm: np.ndarray, labels, pos_label: int):

        #sensitivity e specificity si usano solo per classificatori binari, se la cm non è 2x2 rende (None,None)
        if cm.shape != (2, 2) or pos_label not in labels:
            return None, None

        idx_pos = int(np.where(labels == pos_label)[0][0]) #posizione della classe positiva in labels
        idx_neg = 1 - idx_pos  #poichè le label sono 2

        #metriche relative alla CM
        tp = cm[idx_pos, idx_pos]
        fn = cm[idx_pos, :].sum() - tp
        tn = cm[idx_neg, idx_neg]
        fp = cm[idx_neg, :].sum() - tn

        #esegue un controllo per evitare divisioni per zero
        sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
        return sensitivity, specificity
 

    #calcola la ROC-AUC dalle probabilità (binario e multiclasse) e rende None se non calcolabile
    def safe_roc_auc(self, y_true, y_prob, labels, pos_label: int) -> Optional[float]:

        if y_prob is None:
            return None
        try:
            y_prob = np.asarray(y_prob, dtype=float)
            if len(labels) == 2:
                #roc_auc_score di sklearn assume che il positivo sia il valore maggiore
                #ma se pos_label=0 (malignant su dataset sklearn) bisogna binarizzare a mano
                #altrimenti AUC viene calcolata su classe sbagliata e vale ~1 - AUC_reale
                idx_pos = int(np.where(labels == pos_label)[0][0]) \
                    if pos_label in labels else 1
                pos    = labels[idx_pos]
                score  = y_prob[:, idx_pos] if y_prob.ndim == 2 else y_prob #probabilita' della classe positiva
                y_bin  = (np.asarray(y_true) == pos).astype(int) #1 dove y_true e' la positiva e 0 altrove
                return float(roc_auc_score(y_bin, score))

            #caso multiclasse con one-vs-rest e la stessa strategia di media delle altre metriche
            avg = self.average if self.average in {"macro", "weighted"} else "macro"
            return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average=avg))
        except (ValueError, IndexError):
            return None


    #disegna confusion matrix come heatmap e salva PNG
    def plot_confusion_matrix(self, cm: np.ndarray, label: str, class_names: Optional[list] = None) -> Path:
        
        n = cm.shape[0]
        if class_names is None:
            class_names = [str(i) for i in range(n)]

        fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Classi predette", fontsize=12)
        ax.set_ylabel("Classi reali",    fontsize=12)
        ax.set_title(f"Matrice di confusione - {label}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        safe = label.replace(" ", "_").replace("/", "-")
        path = self.figures_dir / f"confusion_matrix_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path.resolve()


    """
    -valutazione BILANCIATA a bootstrap per test set sbilanciato (nel nostro caso ~5:1).
    -idea (dagli appunti del gruppo): si tengono TUTTI i positivi e ad ogni round si
     campionano n=positivi negativi a caso, si calcolano le metriche e si ripete
     n_rounds volte, poi si media (media +/- dev std). Cosi' accuracy & co. non sono
     gonfiate dalla maggioranza dei negativi.
    -binario: usa pos_label come classe "malata"; metriche per-round su sottoinsieme 50/50.
    -y_prob opzionale (matrice (N,2) o vettore P(positivo)) -> aggiunge ROC_AUC per round.
    -NON sostituisce evaluate_predictions (single-pass): e' complementare.
    """
    def evaluate_balanced_bootstrap(self, y_true, y_pred, y_prob=None,
                                    pos_label: int = 1, n_rounds: int = 10,
                                    random_state: int = 42, label: str = "bootstrap",
                                    make_plot: bool = True) -> BalancedBootstrapResult:

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true e y_pred devono avere la stessa lunghezza.")

        labels = np.unique(y_true)
        if labels.shape[0] != 2 or pos_label not in labels:
            raise ValueError("il bootstrap bilanciato vale solo per il caso binario con pos_label valido.")
        neg_label = int(labels[labels != pos_label][0])

        pos_idx = np.where(y_true == pos_label)[0]
        neg_idx = np.where(y_true == neg_label)[0]
        n_pos = int(pos_idx.shape[0])
        if n_pos == 0 or neg_idx.shape[0] == 0:
            raise ValueError("servono almeno un positivo e un negativo per il bootstrap bilanciato.")
        replace = neg_idx.shape[0] < n_pos  #se i negativi non bastano, reimmissione

        #colonna delle probabilita' della classe positiva (per ROC-AUC), se disponibile
        score_pos = None
        if y_prob is not None:
            yp = np.asarray(y_prob, dtype=float)
            if yp.ndim == 2:
                idx_pos = int(np.where(labels == pos_label)[0][0])  #colonna allineata a labels ordinate
                score_pos = yp[:, idx_pos]
            else:
                score_pos = yp

        rng = np.random.default_rng(random_state)
        metric_keys = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
        if score_pos is not None:
            metric_keys.append("ROC_AUC")

        per_round: list[dict] = []
        for _ in range(n_rounds):
            sampled_neg = rng.choice(neg_idx, size=n_pos, replace=replace)
            idx = np.concatenate([pos_idx, sampled_neg])
            yt, yp_pred = y_true[idx], y_pred[idx]
            row = {
                "Accuracy":    float(accuracy_score(yt, yp_pred)),
                "Precision":   float(precision_score(yt, yp_pred, pos_label=pos_label, average="binary", zero_division=0)),
                "Sensitivity": float(recall_score(yt, yp_pred, pos_label=pos_label, average="binary", zero_division=0)),
                "Specificity": float(recall_score(yt, yp_pred, pos_label=neg_label, average="binary", zero_division=0)),
                "F1":          float(f1_score(yt, yp_pred, pos_label=pos_label, average="binary", zero_division=0)),
            }
            if score_pos is not None:
                y_bin = (yt == pos_label).astype(int)
                try:
                    row["ROC_AUC"] = float(roc_auc_score(y_bin, score_pos[idx]))
                except ValueError:
                    row["ROC_AUC"] = float("nan")  #round degenere (una sola classe nel sample)
            per_round.append(row)

        means, stds = {}, {}
        for k in metric_keys:
            vals = np.array([r[k] for r in per_round], dtype=float)
            means[k] = float(np.nanmean(vals))
            stds[k]  = float(np.nanstd(vals))

        res = BalancedBootstrapResult(
            n_rounds=n_rounds, n_positives=n_pos, n_negatives_per_round=n_pos,
            replace=replace, means=means, stds=stds, per_round=per_round,
        )
        if make_plot:
            res.figure_paths = [str(self.plot_bootstrap(per_round, metric_keys, label))]
        return res


    #boxplot della distribuzione delle metriche sui round del bootstrap, salvato come PNG
    def plot_bootstrap(self, per_round: list, metric_keys: list, label: str) -> Path:
        data = [[r[k] for r in per_round] for k in metric_keys]
        fig, ax = plt.subplots(figsize=(max(6, len(metric_keys) * 1.2), 5))
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(metric_keys) + 1))
        ax.set_xticklabels(metric_keys, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("valore")
        ax.set_title(f"Bootstrap bilanciato - {label} ({len(per_round)} round)")
        plt.tight_layout()

        safe = label.replace(" ", "_").replace("/", "-")
        path = self.figures_dir / f"bootstrap_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path.resolve()