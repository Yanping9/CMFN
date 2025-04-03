from .box_ops import calculate_iou
def select_heatmap(pred, action=None, obj_id=None, triplet=None, tgt_obj=None, tgt_hum=None, set_iou=0.5):
    obj_keep = []
    hum_keep = []
    act_keep = []
    tri_keep = []
    valid_obj_bbox = []
    valid_hum_bbox = []
    for k in tgt_obj.keys():
        if '_{}'.format(obj_id) in k:
            valid_obj_bbox.append(tgt_obj[k])
    for k in tgt_hum.keys():
        if '_0' in k:
            valid_hum_bbox.append(tgt_hum[k])
    for index, obj in enumerate(pred['predictions'][64:]):
        obj_index = index + 64
        if obj['category_id'] == obj_id:
            for i, tgt in enumerate(valid_obj_bbox):
                iou = calculate_iou(obj['bbox'], tgt)
                if iou >= set_iou:
                    obj_keep.append(obj_index)
    for i, obj_index in enumerate(obj_keep):
        hum_index = obj_index - 64
        hum = pred['predictions'][hum_index]
        for j, tgt_ in enumerate(valid_hum_bbox):
            iou = calculate_iou(hum['bbox'], tgt_)
            if iou >= set_iou:
                hum_keep.append(obj_index)

    for _, hoi in enumerate(pred['hoi_prediction']):
        if hoi['category_id'] == action:
            act_keep.append(hoi['object_id'])
    for i, tri in enumerate(pred['triplet_prediction']):
        obj_in = i + 64
        for j in tri['hoi_label']:
            if j == triplet:
                tri_keep.append(obj_in)
    set1 = set(hum_keep)
    set2 = set(act_keep)
    set3 = set(tri_keep)
    assert len(hum_keep) != 0, 'no matched pair for <{},{}>'
    print('hum_keep:{}'.format(hum_keep))
    if len(tri_keep) == 0:
        keep = set1 & set2
    else:
        keep = set1 & set2 & set3
    # for i in keep:

    obj_list = list(keep)
    assert len(obj_list) != 0, 'no matched pair for <{},{}>'
    print('keep{}'.format(obj_list))
    hum_list = []
    for i in obj_list:
        hum_list.append([i-64])
    #hum_list = [x - 64 for x in obj_list]
    return hum_list