# -*- coding: utf-8 -*-
print('i m here')

from collections import Counter
import sys

output_dir = sys.argv[1]

word_list = []
for l in sys.stdin:
    word_list += [w.strip().lower() for w in l.split()]

wordcount = Counter(word_list).most_common()

with open(output_dir + '/wordcount.txt', 'w') as f:
    for count in wordcount:
        f.write('{}\n'.format(count))

print("i'm done")
