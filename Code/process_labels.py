class Labels(dict):
    def __getitem__(self, item):
        for key in self.keys():
            if item in key:
                return super().__getitem__(key)
        raise KeyError(item)


# label_ra = Labels({
#     range(0, 15): 0,
#     range(15, 30): 1,
#     range(30, 45): 2,
#     range(45, 60): 3,
#     range(60, 75): 4,
#     range(75, 90): 5,
#     range(90, 105): 6,
#     range(105, 120): 7,
#     range(120, 135): 8,
#     range(135, 150): 9,
#     range(150, 165): 10,
#     range(165, 180): 11,
#     range(180, 195): 12,
#     range(195, 210): 13,
#     range(210, 225): 14,
#     range(225, 240): 15,
#     range(240, 255): 16,
#     range(255, 270): 17,
#     range(270, 285): 18,
#     range(285, 300): 19,
#     range(300, 315): 20,
#     range(315, 330): 21,
#     range(330, 345): 22,
#     range(345, 360): 23
# })
#
# label_dec = Labels({
#     range(-90, -75): 0,
#     range(-75, -60): 1,
#     range(-60, -45): 2,
#     range(-45, -30): 3,
#     range(-30, -15): 4,
#     range(-15, 0): 5,
#     range(0, 15): 6,
#     range(15, 30): 7,
#     range(30, 45): 8,
#     range(45, 60): 9,
#     range(60, 75): 10,
#     range(75, 90): 11
# })

# label_ra = Labels({
#     range(0, 45): 0,
#     range(45, 90): 1,
#     range(90, 135): 2,
#     range(135, 180): 3,
#     range(180, 225): 4,
#     range(225, 270): 5,
#     range(270, 315): 6,
#     range(315, 360): 7
# })
#
# label_dec = Labels({
#     range(-90, -45): 0,
#     range(-45, 0): 1,
#     range(0, 45): 2,
#     range(45, 90): 3,
# })

label_ra = Labels({
    range(0, 45): 0,
    range(45, 90): 1,
    range(90, 135): 2,
    range(135, 180): 3,
    range(180, 225): 4,
    range(225, 270): 5,
    range(270, 315): 6,
    range(315, 360): 7
})

label_dec = Labels({
    range(-90, -45): 0,
    range(-45, 0): 1,
    range(0, 45): 2,
    range(45, 90): 3,
})

data = open('star_tracker_dataset/ra_dec.txt', 'r').readlines()
labels = open('star_tracker_dataset/labels.txt', 'w')

for line in data:
    line = line.strip()
    ra, dec = line.split('_')
    ra, dec = int(ra), int(dec)
    if dec > 0:
        if ra < 180:
            label = 0
        else:
            label = 1
    else:
        if ra < 180:
            label = 2
        else:
            label = 3
    # label = 0 if dec > 0 else 1
    labels.write('{}\n'.format(label))
