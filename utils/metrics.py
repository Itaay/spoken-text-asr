import editdistance


def wer(preds, targets, use_cer=False):
    lev_dis = 0.0
    word_count = 0
    for pred, target in zip(preds, targets):
        if use_cer:
            pred_list = list(pred)
            target_list = list(target)
        else:
            pred_list = pred.split()
            target_list = target.split()
        word_count += len(target_list)
        lev_dis += editdistance.eval(pred_list, target_list)
    if word_count != 0:
        wer = 100.0 * lev_dis / word_count
    else:
        wer = float('inf')
    return wer
