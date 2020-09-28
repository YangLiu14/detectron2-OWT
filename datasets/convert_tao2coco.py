import json


def coco_id2tao_id():
    # Loading json files
    val_annot_path = "/home/kloping/OpenSet_MOT/data/TAO/annotations/validation.json"
    with open(val_annot_path, 'r') as f:
        val_annot_dict = json.load(f)

    with open("coco/coco2synset.json") as f:
        coco2synset = json.load(f)

    # list all the categories of TAO (in the style of synset)
    TAO_categories = list()
    TAO_ids = list()

    for c in val_annot_dict['categories']:
        id = c['id']
        TAO_ids.append(c['id'])
        TAO_categories.append(c['synset'])

    synset_cls = [v['synset'] for k, v in coco2synset.items()]
    coco_ids = [v['coco_cat_id'] for k, v in coco2synset.items()]

    coco2tao = dict()
    coco_leftout = list()
    for s, coco_id in zip(synset_cls, coco_ids):
        if s in TAO_categories:
            i = TAO_categories.index(s)
            coco2tao[coco_id] = TAO_ids[i]
        else:
            coco_leftout.append((coco_id, s))

    print("The following synset categories are left out")
    print(coco_leftout)

    return coco2tao




def check_coco_tao_intersection(tao_category_dict, split='TRAIN'):
    # list all the categories of TAO
    TAO_categories = list()
    for c in tao_category_dict:
        # class_name = c['name']
        class_name = c['synset']
        TAO_categories.append(class_name)


    # list all the categories of COCO
    with open('coco/coco_classes.json', 'r') as f:
        coco_dict = json.load(f)

    coco_classes = [cname for k, cname in coco_dict.items()]

    coco_leftout = list()

    # Replace coco naming conversions with synset
    with open("coco/coco2synset.json") as f:
        coco2synset = json.load(f)

    # synset_cls = [v['synset'][:-5] for k, v in coco2synset.items()]
    synset_cls = [v['synset'] for k, v in coco2synset.items()]
    coco_ids = [v['coco_cat_id'] for k, v in coco2synset.items()]

    TAO_categories_string = " ".join(TAO_categories)
    for coco_c in synset_cls:
        if coco_c not in TAO_categories:
        # if coco_c not in TAO_categories_string:
            coco_leftout.append(coco_c)

    print("Number of COCO categories that are not covered in the TAO_{}: ".format(split), len(coco_leftout))
    print(coco_leftout)


# def convert_to_coco(tao_dict):
#     """
#     Covert TAO validation set annotation to coco format
#     """
#     videos = tao_dict['videos']
#     annotations = tao_dict['annotations']
#     tracks = tao_dict['tracks']
#     images = tao_dict['images']
#     info = tao_dict['info']
#     categories = tao_dict['categories']
#
#     # Load coco_val_2017
#     with open("coco/annotations/instances_val2017.json", "r") as f:
#         coco_annot = json.load(f)
#
#     coco_annot['images']
#     coco_annot['annotations']
#     coco_annot['categories']
#
#     # categories, annotations, 可以直接挪过来
#
#     print("")

def check_train_val_diff(train_dict, val_dict):
    test = (train_dict == val_dict)

    train_categories = set()
    for c in train_dict:
        if c['instance_count'] > 0:
            train_categories.add(c['name'])
    train_categories = list(train_categories)
    train_categories.sort()
    print("Number of classes in train_categories:", len(train_categories))

    val_categories = set()
    for c in val_dict:
        if c['instance_count'] > 0:
            val_categories.add(c['name'])
    val_categories = list(val_categories)
    val_categories.sort()
    print("Number of classes in val_categories:", len(val_categories))

    knowns = list()
    unknowns = list()
    for val_cls in val_categories:
        if val_cls in train_categories:
            knowns.append(val_cls)
        else:
            unknowns.append(val_cls)

    # print("Number of knowns:", len(knowns))
    # print(knowns)
    # print("Number of unknowns:", len(unknowns))
    # print(unknowns)


if __name__ == "__main__":
    coco_id2tao_id()

    # The followings are for test
    train_annot_path = "/home/kloping/OpenSet_MOT/data/TAO/annotations/train.json"
    with open(train_annot_path, 'r') as f:
        train_annot_dict = json.load(f)

    val_annot_path = "/home/kloping/OpenSet_MOT/data/TAO/annotations/validation.json"
    with open(val_annot_path, 'r') as f:
        val_annot_dict = json.load(f)

    test = (train_annot_dict == val_annot_dict)

    # TEST 1: Check how many instances of coco already exists in TAO VAL dataset
    check_coco_tao_intersection(train_annot_dict['categories'], split='TRAIN')
    check_coco_tao_intersection(val_annot_dict['categories'], split='VAL')

    # TEST 2: Check how many classes are covered in TRAIN and how many covered in VAL
    # check_train_val_diff(train_annot_dict['categories'], val_annot_dict['categories'])
    # print("###################################")

    # Conversion (No need)
    # convert_to_coco(annot_dict)

    print("Done")
