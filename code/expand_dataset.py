from PIL import Image

labels = open('star_tracker_dataset/labels.txt', 'r').readlines()
new_labels = open('star_tracker_dataset/labels_expanded.txt', 'w')

NUM_IMAGES = 2592
for i in range(NUM_IMAGES):
    print(i)
    new_labels.write(labels[i])
    filename = 'star_tracker_dataset/images/stars_{0:03d}.png'.format(i)
    img = Image.open(filename)
    for j in range(15, 360, 15):
        new_labels.write(labels[i])
        img_rot = img.rotate(j, expand=True)
        img_rot.save(filename.rstrip('.png') + '_rot_{}.png'.format(j))
