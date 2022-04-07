class Labels(dict):
    def __getitem__(self, item):
        for key in self.keys():
            if item in key:
                return super().__getitem__(key)
        raise KeyError(item)


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
    labels.write('{}\n'.format(label))
