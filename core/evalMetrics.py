import numpy as np

class EvalMetrics:
    def __init__(self):
        pass

    def confusion_matrix(self,y_pred, y_real, normalize=None):
        """Compute confusion matrix.

        Args:
            y_pred (list[int] | np.ndarray[int]): Prediction labels.
            y_real (list[int] | np.ndarray[int]): Ground truth labels.
            normalize (str | None): Normalizes confusion matrix over the true
                (rows), predicted (columns) conditions or all the population.
                If None, confusion matrix will not be normalized. Options are
                "true", "pred", "all", None. Default: None.

        Returns:
            np.ndarray: Confusion matrix.
        """
        if normalize not in ['true', 'pred', 'all', None]:
            raise ValueError("normalize must be one of {'true', 'pred', "
                             "'all', None}")

        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
            if y_pred.dtype == np.int32:
                y_pred = y_pred.astype(np.int64)
        if not isinstance(y_pred, np.ndarray):
            raise TypeError(
                f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
        if not y_pred.dtype == np.int64:
            raise TypeError(
                f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

        if isinstance(y_real, list):
            y_real = np.array(y_real)
            if y_real.dtype == np.int32:
                y_real = y_real.astype(np.int64)
        if not isinstance(y_real, np.ndarray):
            raise TypeError(
                f'y_real must be list or np.ndarray, but got {type(y_real)}')
        if not y_real.dtype == np.int64:
            raise TypeError(
                f'y_real dtype must be np.int64, but got {y_real.dtype}')

        label_set = np.unique(np.concatenate((y_pred, y_real)))
        num_labels = len(label_set)
        max_label = label_set[-1]
        label_map = np.zeros(max_label + 1, dtype=np.int64)
        for i, label in enumerate(label_set):
            label_map[label] = i

        y_pred_mapped = label_map[y_pred]
        y_real_mapped = label_map[y_real]

        confusion_mat = np.bincount(
            num_labels * y_real_mapped + y_pred_mapped,
            minlength=num_labels**2).reshape(num_labels, num_labels)

        with np.errstate(all='ignore'):
            if normalize == 'true':
                confusion_mat = (
                    confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
            elif normalize == 'pred':
                confusion_mat = (
                    confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
            elif normalize == 'all':
                confusion_mat = (confusion_mat / confusion_mat.sum())
            confusion_mat = np.nan_to_num(confusion_mat)

        return confusion_mat


    def mean_class_accuracy(self,scores, labels):
        """Calculate mean class accuracy.

        Args:
            scores (list[np.ndarray]): Prediction scores for each class.
            labels (list[int]): Ground truth labels.

        Returns:
            np.ndarray: Mean class accuracy.
        """
        pred = np.argmax(scores, axis=1)
        labels = np.argmax(labels, axis=1)
        cf_mat = self.confusion_matrix(pred, labels).astype(float)

        cls_cnt = cf_mat.sum(axis=1)
        cls_hit = np.diag(cf_mat)

        mean_class_acc = np.mean(
            [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

        return mean_class_acc

    def per_class_accuracy(self,scores, labels):
        pred = np.argmax(scores, axis=1)
        cf_mat = self.confusion_matrix(pred, labels).astype(float)

        cls_cnt = cf_mat.sum(axis=1)
        cls_hit = np.diag(cf_mat)
        hit_ratio = np.array(
            [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])
        return hit_ratio, cf_mat
    def top_k_accuracy(self,scores, labels, topk=(1, )):
        """Calculate top k accuracy score.

        Args:
            scores (list[np.ndarray]): Prediction scores for each class.
            labels (list[int]): Ground truth labels.
            topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

        Returns:
            list[float]: Top k accuracy score for each k.
        """
        res = []
        labels = np.array(labels.argmax(-1))[:, np.newaxis]
        for k in topk:
            max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
            match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
            topk_acc_score = match_array.sum() / match_array.shape[0]
            res.append(topk_acc_score)

        return res