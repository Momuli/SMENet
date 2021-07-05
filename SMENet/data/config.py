HOME ='/media/newuser/03daf96d-2a41-4fef-b562-0833955aee6f/lina/SMENet'
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (86, 91, 82)

voc = {
    'num_classes': 11,
    'lr_steps': (7740, 17145),
    'max_iter': 28500,
    'feature_maps': [50, 25, 13, 7, 5, 3],  # scale of feature map
    'min_dim': 400,   # input size
    'steps': [8, 16, 30, 57, 80, 133],  # The mapping relationship between the feature map  and the input img
    'min_sizes': [25, 65, 116, 167, 218, 269],
    'max_sizes': [65, 116, 167, 218, 269, 320],
    'aspect_ratios': [[2, 0.65], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
