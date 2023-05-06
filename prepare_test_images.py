import os

# images
# filename = ""
# os.mkdir("datasets/images/test2020")
# for f in os.listdir("datasets/images/test2019/"):
#   if f.isnumeric():
#     for f_seq in os.listdir("datasets/images/test2019/" + f + '/'):
#       os.rename("datasets/images/test2019/" + f + '/' + f_seq, "datasets/images/test2020/" + f + '-' + f_seq)

# labels
# filename = ""
# os.mkdir("datasets/labels/test2020")
# for f in os.listdir("datasets/labels/test2019/"):
#   if f.isnumeric():
#     for f_seq in os.listdir("datasets/labels/test2019/" + f + '/'):
#       os.rename("datasets/labels/test2019/" + f + '/' + f_seq, "datasets/labels/test2020/" + f + '-' + f_seq)

# validation
filename = ""
os.mkdir("datasets/images/val2020")
for f in os.listdir("datasets/images/val2019/"):
  if f.isnumeric():
    for f_seq in os.listdir("datasets/images/val2019/" + f + '/'):
      os.rename("datasets/images/val2019/" + f + '/' + f_seq, "datasets/images/val2020/" + f + '-' + f_seq)
os.mkdir("datasets/labels/val2020")
for f in os.listdir("datasets/labels/val2019/"):
  if f.isnumeric():
    for f_seq in os.listdir("datasets/labels/val2019/" + f + '/'):
      os.rename("datasets/labels/val2019/" + f + '/' + f_seq, "datasets/labels/val2020/" + f + '-' + f_seq)