data = open('star_tracker_dataset/ra_dec.txt', 'r').readlines()
labels = open('star_tracker_dataset/labels.txt', 'w')

for line in data:
    line = line.strip()
    ra, dec = line.split('_')
    ra, dec = int(ra), int(dec)
    if dec > 0:
        if ra < 180:
            label = 0  # North-East
        else:
            label = 1  # North-West
    else:
        if ra < 180:
            label = 2  # South-East
        else:
            label = 3  # South-West
    labels.write('{}\n'.format(label))
