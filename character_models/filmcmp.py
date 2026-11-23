"""Film frames next to our SlipperyT renders: python3 filmcmp.py out.png"""
import sys
import numpy as np
from PIL import Image
U = '/root/.claude/uploads/041a3851-cc56-5eec-b608-dd5f8b64d61c/'
refs = [Image.open(U + '40244350-image.webp').convert('RGB').crop((285, 40, 470, 400)),
        Image.open(U + '43302085-image.webp').convert('RGB').crop((330, 50, 560, 330))]
H = 720
out = [r.resize((int(r.width * H / r.height), H), Image.LANCZOS) for r in refs]
for f in ('front', 'q3', 'side', 'back'):
    o = Image.open(f'out/renders/Jenny_Nude_SlipperyT_{f}.png').convert('RGB')
    a = np.array(o).astype(int); fg = np.abs(a - a[2, 2]).sum(-1) > 30
    ys, xs = np.nonzero(fg); o = o.crop((xs.min(), ys.min(), xs.max(), ys.max()))
    out.append(o.resize((int(o.width * H / o.height), H), Image.LANCZOS))
s = Image.new('RGB', (sum(i.width for i in out) + 20 * len(out), H), 'white'); x = 0
for i in out:
    s.paste(i, (x, 0)); x += i.width + 20
s.save(sys.argv[1])
