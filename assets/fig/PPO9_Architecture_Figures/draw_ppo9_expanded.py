#!/usr/bin/env python3
"""Draw the source-verified PPO9 paired network as a publication-style vector PDF.

Based on profiles/ppo9.json, utils/mirror_policy.py, utils/mirror_rlgames.py,
and run.py in the supplied Racer-HitchOpen-RLGPU repository.
All geometric coordinates below are top-left based design units.
"""
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT = Path(__file__).resolve().parent
W, H = 2400, 1450
C = dict(ink='#183047', muted='#65768A', blue='#367FBE', paleBlue='#EEF5FC',
         teal='#218A86', paleTeal='#EBF6F3', coral='#D46C58', paleCoral='#FFF1EC',
         amber='#B88728', paleAmber='#FFF6DF', purple='#8065B3', palePurple='#F4F0FA',
         line='#D9E1EA', white='#FFFFFF')
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DV', str(FONT_DIR / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVB', str(FONT_DIR / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DVI', str(FONT_DIR / 'DejaVuSans-Oblique.ttf')))
c = canvas.Canvas(str(OUT / 'PPO9_Network_Expanded.pdf'), pagesize=(W*.5,H*.5))
c.setTitle('PPO9 | Expanded shared-mirror actor and privileged critic')
c.setAuthor('Source-verified architecture drawing')
c.scale(.5,.5)

def col(key): return HexColor(C.get(key,key))
def rect(x,y,w,h,fill='white',stroke='line',radius=15,lw=1.8):
    c.setLineWidth(lw); c.setFillColor(col(fill)); c.setStrokeColor(col(stroke))
    c.roundRect(x,H-y-h,w,h,radius,fill=1,stroke=1)
def text(x,y,s,size=24,color='ink',bold=False,align='left',italic=False):
    c.setFont('DVB' if bold else ('DVI' if italic else 'DV'),size)
    c.setFillColor(col(color))
    getattr(c, {'left':'drawString','center':'drawCentredString','right':'drawRightString'}[align])(x,H-y,s)
def line(points,color='ink',lw=2.5,dash=None):
    c.saveState();c.setStrokeColor(col(color));c.setLineWidth(lw)
    c.setLineJoin(1);c.setLineCap(1)
    if dash:c.setDash(*dash)
    p=c.beginPath();p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]:p.lineTo(x,H-y)
    c.drawPath(p);c.restoreState()
def arrow(points,color='ink',lw=2.5,head=9,dash=None):
    line(points,color,lw,dash)
    (x0,y0),(x,y)=points[-2:]
    a=math.atan2(y-y0,x-x0);d=math.pi/6
    pts=[(x,y),(x-head*math.cos(a-d),y-head*math.sin(a-d)),(x-head*math.cos(a+d),y-head*math.sin(a+d))]
    c.setFillColor(col(color));p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
    for px,py in pts[1:]:p.lineTo(px,H-py)
    p.close();c.drawPath(p,fill=1,stroke=0)
def circle(x,y,r,fill='white',stroke='line',lw=2):
    c.setLineWidth(lw);c.setStrokeColor(col(stroke));c.setFillColor(col(fill));c.circle(x,H-y,r,stroke=1,fill=1)
def dot(x,y,color='ink',r=4):circle(x,y,r,color,color,1)
def label_box(x,y,w,s,color='purple',fill='palePurple',size=22):
    rect(x-w/2,y-19,w,36,fill,fill,9,0.5);text(x,y+7,s,size,color,align='center')
def recurrent_icon(x,y,color='blue'):
    c.saveState();c.setStrokeColor(col(color));c.setLineWidth(2.7)
    c.arc(x-19,H-y-19,x+19,H-y+19,22,285)
    c.restoreState()
    arrow([(x+12,y+14),(x+18,y+12)],color,2.5,8)
    circle(x,y,5,'white',color,1.7)
def layer(x,y,w,h,color,light):
    # A front slab and slight offset back slab read as a dense layer, not a 3D chart.
    rect(x-w/2+5,y-h/2-5,w,h,light,color,5,1.2)
    rect(x-w/2,y-h/2,w,h,light,color,5,1.5)
    for yy in (-.27,0,.27):circle(x,y+yy*h,3.5,color,color,0.7)
def action_mlp(y):
    x=1430;rect(x,y-91,455,186,'white','line',15)
    text(x+227.5,y-60,'Action MLP',25,'blue',True,'center')
    xs=[1480,1580,1680,1780];hs=[73,61,49,35];units=['256','124','64','3']
    for j in range(3):arrow([(xs[j]+26,y),(xs[j+1]-27,y)],'muted',1.6,6)
    for i,xx in enumerate(xs):
        layer(xx,y,42,hs[i], 'blue' if i<3 else 'coral','paleBlue' if i<3 else 'paleCoral')
        text(xx,y+62,units[i],23,'ink',True,'center')
    text(1580,y+86,'ELU hidden layers',22,'muted',align='center')
    text(1780,y+86,'linear',22,'coral',align='center')
def gaussian_icon(x,y):
    pts=[]
    for k in range(61):
        xx=-49+k*98/60
        pts.append((x+xx,y-30*math.exp(-.5*(xx/17)**2)))
    line([(x-53,y),(x+53,y)],'muted',1.5)
    line(pts,'coral',2.8)

# Header.
text(64,69,'PPO9 | Inside the shared-mirror network',44,'ink',True)
text(67,111,'A source-verified view of the default recurrent policy and its independent privileged critic',24,'muted')
line([(1770,98),(1817,98)],'ink',2.5)
text(1828,106,'forward path',22,'muted')
line([(2050,98),(2097,98)],'purple',2,(6,6))
text(2108,106,'tied weights',22,'muted')

# Actor container.
rect(50,145,2300,845,'paleBlue','line',22)
text(86,189,'A',28,'blue',True)
text(129,189,'MIRROR-SHARED RECURRENT ACTOR',28,'ink',True)
text(2311,189,'51-D observations   →   3-D Gaussian policy',24,'muted',align='right')

# Current normalized observations and explicit reflection.
for yy,lab,sub,accent in [(345,'x⁺ = Norm(oₜ)','51 dimensions','blue'),(765,'x⁻ = Mₒ x⁺','51 dimensions','teal')]:
    rect(85,yy-53,200,106,'white',accent,14,2)
    text(185,yy-9,lab,24,accent,True,'center')
    text(185,yy+28,sub,22,'muted',align='center')
arrow([(185,398),(185,712)],'teal',2.5)
circle(185,548,31,'white','teal',2)
text(185,557,'Mₒ',28,'teal',True,'center')
text(185,611,'reflection',22,'teal',align='center')

# Observation bypass: current measurements are concatenated alongside memory.
arrow([(300,345),(300,225),(1305,225),(1305,280)],'blue',2)
text(805,215,'51-D observation bypass',22,'blue',align='center')
arrow([(300,765),(300,903),(1305,903),(1305,830)],'teal',2)
text(800,934,'51-D observation bypass',22,'teal',align='center')

# Shared GRU blocks, distinct recurrent histories.
for yy,accent,fill,sign in [(345,'blue','paleBlue','⁺'),(765,'teal','paleTeal','⁻')]:
    arrow([(285,yy),(350,yy)],accent,2.5)
    dot(300,yy,accent)
    rect(350,yy-55,210,110,'white',accent,14,2)
    recurrent_icon(388,yy-1,accent)
    text(475,yy-5,'GRU',28,accent,True,'center')
    text(475,yy+30,'51 → 128',23,'muted',align='center')
    arrow([(560,yy),(1220,yy)],accent,2.6)
    text(640,yy-19,'h'+sign+'  ·  128-D',23,accent)
    dot(625,yy,accent)
text(455,263,'h⁺ₜ₋₁',25,'blue',align='center')
arrow([(455,270),(455,290)],'blue',2.1)
text(455,856,'h⁻ₜ₋₁',25,'teal',align='center')
arrow([(455,841),(455,820)],'teal',2.1)
line([(455,400),(455,710)],'purple',1.8,(6,6))
label_box(455,543,170,'shared θGRU')
rect(352,568,206,68,'paleBlue','paleBlue',2,0.1)
text(455,592,'separate',22,'muted',align='center')
text(455,620,'histories',22,'muted',align='center')

# Auxiliary head on each recurrent output, with the exact antisymmetric assembly.
for yy,from_y,accent,sign in [(460,345,'blue','⁺'),(650,765,'teal','⁻')]:
    arrow([(625,from_y),(625,yy),(700,yy)],'amber',2.3)
    rect(700,yy-39,185,78,'paleAmber','amber',12,1.8)
    text(792.5,yy-5,'β head',25,'amber',True,'center')
    text(792.5,yy+25,'128 → 64 → 1',22,'ink',align='center')
    end=530 if sign=='⁺' else 610
    arrow([(885,yy),(1047.5,yy),(1047.5,end)],'amber',2.3)
    text(918,yy-13,'b'+sign,25,'amber',True)
line([(792.5,499),(792.5,611)],'purple',1.8,(6,6))
label_box(792.5,551,138,'shared θβ')
rect(925,530,245,80,'paleAmber','amber',14,2)
text(1047.5,560,'β̂ = ½ (b⁺ − b⁻)',27,'amber',True,'center')
text(1047.5,591,'sideslip estimate',22,'muted',align='center')
arrow([(1170,570),(1190,570)],'amber',2.3,7)
circle(1215,570,25,'white','amber',2)
text(1215,578,'sg',22,'amber',True,'center')
line([(1240,570),(1305,570)],'amber',2.3)
dot(1305,570,'amber')
arrow([(1305,570),(1305,410)],'amber',2.3)
arrow([(1305,570),(1305,700)],'amber',2.3)
text(1217,619,'stop-grad',22,'amber',align='center')
text(1327,478,'+ sg(β̂)',23,'amber')
text(1327,663,'− sg(β̂)',23,'amber')

# Concatenation: recurrent feature, current measurement, signed detached estimate.
for yy,accent,sign in [(345,'blue','⁺'),(765,'teal','⁻')]:
    rect(1220,yy-65,170,130,'white',accent,14,2)
    circle(1305,yy-18,17,'white',accent,1.6)
    text(1305,yy-9,'⊕',26,accent,align='center')
    text(1305,yy+28,'Concat',24,accent,True,'center')
    text(1305,yy+55,'180-D',23,'muted',align='center')
    arrow([(1390,yy),(1430,yy)],accent,2.5)
    action_mlp(yy)
    arrow([(1885,yy),(2070,yy)],accent,2.5)
    if sign=='⁺':text(1908,yy-18,'m'+sign,26,accent,True)

# The MLP has one set of weights, evaluated in two branches.
line([(1635,440),(1635,674)],'purple',1.8,(6,6))
label_box(1635,551,147,'shared θπ')
rect(1457,580,366,36,'paleBlue','paleBlue',2,0.1)
text(1640,605,'180 → 256 → 124 → 64 → 3',22,'muted',align='center')

# Restore action orientation, then average the two raw branch means.
rect(1910,710,160,110,'paleCoral','coral',14,2)
arrow([(1885,765),(1910,765)],'teal',2.5,8)
text(1940,693,'m⁻',26,'teal',True)
text(1990,751,'Dₐ',30,'coral',True,'center')
text(1990,788,'−1, +1, +1',22,'ink',align='center')
text(1990,853,'action reflection',22,'coral',align='center')
arrow([(2070,345),(2070,531)],'blue',2.5)
arrow([(2070,765),(2070,609)],'teal',2.5)
circle(2070,570,39,'white','coral',2.4)
text(2070,580,'½Σ',29,'coral',True,'center')
text(1960,638,'align + average',22,'muted',align='center')
arrow([(2109,570),(2160,570)],'coral',2.5)
text(2135,548,'μₜ',25,'coral',True,'center')

# State-independent learned variance, not a constant or an observation-dependent head.
rect(2125,367,202,98,'white','coral',14,1.8)
text(2226,402,'Learned log σ',23,'coral',True,'center')
text(2226,434,'global 3-vector',22,'muted',align='center')
arrow([(2245,465),(2245,510)],'coral',2.2)
text(2265,497,'exp',22,'coral')
rect(2160,510,170,120,'paleCoral','coral',15,2)
gaussian_icon(2245,561)
text(2245,605,'Gaussian',24,'coral',True,'center')
arrow([(2245,630),(2245,685)],'coral',2.5)
text(2245,722,'aₜ ~ πθ',28,'coral',True,'center')
text(2245,758,'3-D action',23,'muted',align='center')
text(2225,845,'σ init',22,'muted',align='center')
text(2225,879,'[.35, .45, .25]',22,'coral',align='center')

# Thin footer within actor: essential semantic distinctions, not hyperparameter clutter.
text(90,970,'β supervision: simulator labels during training.  sg: stop-gradient.  Deterministic evaluation uses μₜ; action clipping and rescaling follow.',22,'muted')

# Independent critic: batch the two mirrored states, evaluate a single MLP, average.
rect(50,1032,2300,321,'paleTeal','line',22)
text(86,1078,'B',28,'teal',True)
text(129,1078,'INDEPENDENT PRIVILEGED CRITIC',28,'ink',True)
text(2310,1078,'Separate weights φ  ·  Feedforward only',24,'muted',align='right')
rect(85,1160,210,98,'white','teal',13,2)
text(190,1197,'Norm(sₜ)',27,'teal',True,'center')
text(190,1233,'82 dimensions',22,'muted',align='center')
arrow([(295,1209),(355,1209)],'teal',2.5)
rect(355,1137,225,145,'white','teal',13,1.8)
rect(373,1151,189,49,'paleTeal','paleTeal',7,.5)
rect(373,1214,189,49,'paleTeal','paleTeal',7,.5)
text(467,1184,'s⁺',26,'teal',True,'center')
text(467,1248,'s⁻ = Mₛ s⁺',24,'teal',True,'center')
arrow([(580,1209),(680,1209)],'teal',2.5)
text(631,1188,'paired',22,'muted',align='center')
rect(680,1119,760,186,'white','line',15,1.8)
text(1060,1150,'Shared value MLP',25,'teal',True,'center')
xs=[790,970,1150,1330]; hs=[78,64,50,34]
for j in range(3):arrow([(xs[j]+31,1209),(xs[j+1]-31,1209)],'muted',1.7,7)
for j,(xx,hh,unit) in enumerate(zip(xs,hs,['256','128','64','1'])):
    layer(xx,1209,48,hh,'teal','paleTeal')
    text(xx,1275,unit,24,'ink',True,'center')
text(1060,1298,'ELU hidden layers; linear value head',22,'muted',align='center')
line([(1440,1209),(1482,1209)],'teal',2.4)
dot(1482,1209,'teal')
for yy,lab in [(1168,'v⁺'),(1250,'v⁻')]:
    arrow([(1482,1209),(1482,yy),(1555,yy)],'teal',2.3)
    circle(1585,yy,30,'white','teal',1.8)
    text(1585,yy+8,lab,25,'teal',True,'center')
    arrow([(1615,yy),(1730,yy),(1730,1172 if yy<1209 else 1246)],'teal',2.3)
circle(1730,1209,37,'white','teal',2.2)
text(1730,1219,'½Σ',28,'teal',True,'center')
arrow([(1767,1209),(1840,1209)],'teal',2.5)
rect(1840,1156,220,106,'white','teal',14,2)
text(1950,1201,'Vφ(sₜ)',34,'teal',True,'center')
text(1950,1241,'scalar value',22,'muted',align='center')
text(2200,1192,'Reflection invariant',22,'teal',True,'center')
text(2200,1236,'V(s) = V(Mₛs)',25,'ink',align='center')
text(90,1333,'The actor and critic have independent parameters. Within each network, mirrored evaluations reuse one parameter set.',22,'muted')

# Figure footer / provenance.
text(67,1404,'Default PPO9 · 9 preview points · Actor core: 164,227 parameters · Critic core: 62,465 parameters',22,'muted')
text(2335,1404,'Racer-HitchOpen-RLGPU',22,'ink',True,align='right')
c.showPage();c.save()
print(OUT / 'PPO9_Network_Expanded.pdf')
