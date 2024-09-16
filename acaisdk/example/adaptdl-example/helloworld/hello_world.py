import adaptdl.env
import os
import time

with open(os.path.join(adaptdl.env.share_path(), "foo.txt"), "w") as f:
    f.write("Hello, world!")

time.sleep(100)
