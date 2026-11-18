#!/usr/bin/env python3
"""Deterministic vector figure for the supplied default PPO9 source snapshot.

Requires reportlab.  Run this file, then render its PDF with Poppler:
  pdftoppm -png -singlefile -scale-to-x 4800 -scale-to-y -1 PPO9_Overview.pdf PPO9_Overview
No repository files or model weights are modified.
"""
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color

ROOT = Path(__file__).resolve().parent
W,H = 2400,1420
SCALE = .5
C = {
    'ink':'#183047','muted':'#65768A','blue':'#367FBE','paleBlue':'#EEF5FC',
    'teal':'#218A86','paleTeal':'#EBF6F3','coral':'#D46C58','paleCoral':'#FFF1EC',
    'amber':'#B88728','paleAmber':'#FFF6DF','purple':'#8065B3','palePurple':'#F4F0FA',
    'line':'#D9E1EA','white':'#FFFFFF','soft':'#FAFBFD','road':'#E5EBF1',
}
FONTROOT=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('Regular','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTROOT/file)))
cv=canvas.Canvas(str(ROOT/'PPO9_Overview.pdf'),pagesize=(W*SCALE,H*SCALE),pageCompression=1)
cv.setTitle('PPO9 - Observable recurrent control architecture')
cv.setAuthor('Racer-HitchOpen-RLGPU')
cv.setSubject('Default PPO9 profile; source-verified network, observations, actions and independent training branches.')
cv.scale(SCALE,SCALE)

def color(v):return HexColor(C.get(v,v))
def line(x1,y1,x2,y2,stroke='ink',width=2,dash=None):
    cv.saveState();cv.setStrokeColor(color(stroke));cv.setLineWidth(width);cv.setLineCap(1)
    if dash:cv.setDash(dash)
    cv.line(x1,H-y1,x2,H-y2);cv.restoreState()
def path(points,stroke='ink',width=2,fill=None,dash=None,close=False):
    cv.saveState();cv.setStrokeColor(color(stroke) if stroke else color('white'));cv.setLineWidth(width);cv.setLineJoin(1);cv.setLineCap(1)
    if fill:cv.setFillColor(color(fill))
    if dash:cv.setDash(dash)
    p=cv.beginPath()
    for cmd in points:
        if cmd[0]=='M':p.moveTo(cmd[1],H-cmd[2])
        elif cmd[0]=='L':p.lineTo(cmd[1],H-cmd[2])
        elif cmd[0]=='C':p.curveTo(cmd[1],H-cmd[2],cmd[3],H-cmd[4],cmd[5],H-cmd[6])
    if close:p.close()
    cv.drawPath(p,fill=int(bool(fill)),stroke=int(bool(stroke)));cv.restoreState()
def poly(points,stroke='ink',width=2,fill=None):
    path([('M',*points[0])]+[('L',*p) for p in points[1:]],stroke,width,fill,close=True)
def rect(x,y,w,h,fill='white',stroke='line',r=13,width=1.8,dash=None):
    cv.saveState();cv.setLineWidth(width)
    if fill:cv.setFillColor(color(fill))
    if stroke:cv.setStrokeColor(color(stroke))
    if dash:cv.setDash(dash)
    cv.roundRect(x,H-y-h,w,h,r,stroke=int(bool(stroke)),fill=int(bool(fill)));cv.restoreState()
def circle(x,y,r,fill='white',stroke='ink',width=2):
    cv.saveState();cv.setLineWidth(width)
    if fill:cv.setFillColor(color(fill))
    if stroke:cv.setStrokeColor(color(stroke))
    cv.circle(x,H-y,r,stroke=int(bool(stroke)),fill=int(bool(fill)));cv.restoreState()
def text(x,y,s,size=22,fill='ink',font='Regular',anchor='left'):
    cv.saveState();cv.setFont(font,size);cv.setFillColor(color(fill))
    if anchor=='center':cv.drawCentredString(x,H-y,s)
    elif anchor=='right':cv.drawRightString(x,H-y,s)
    else:cv.drawString(x,H-y,s)
    cv.restoreState()
def arrow(points,stroke='ink',width=2.4,dash=None,head=10):
    path([('M',*points[0])]+[('L',*p) for p in points[1:]],stroke,width,dash=dash)
    x,y=points[-1];xp,yp=points[-2];a=math.atan2(y-yp,x-xp)
    q=[(x,y),(x-head*math.cos(a)+head*.48*math.sin(a),y-head*math.sin(a)-head*.48*math.cos(a)),(x-head*math.cos(a)-head*.48*math.sin(a),y-head*math.sin(a)+head*.48*math.cos(a))]
    poly(q,stroke,1,stroke)
def pill(x,y,w,label,fill='paleBlue',fg='blue',size=20,h=34):
    rect(x,y,w,h,fill,None,r=h/2);text(x+w/2,y+h*.70,label,size,fg,'Bold','center')
def section(x,y,num,title,accent='blue'):
    pill(x,y-26,42,num,'pale'+accent.title() if 'pale'+accent.title() in C else 'paleBlue',accent,18,34)
    text(x+56,y,title,28,'ink','Bold')
def dot(x,y,fill='ink',r=3.2):circle(x,y,r,fill,None)

def chip(x,y,s=1,accent='blue'):
    rect(x-14*s,y-14*s,28*s,28*s,'white',accent,r=4*s,width=2)
    rect(x-6*s,y-6*s,12*s,12*s,'paleBlue',None,r=2)
    for a in (-8,0,8):
        line(x+a*s,y-20*s,x+a*s,y-14*s,accent,2)
        line(x+a*s,y+14*s,x+a*s,y+20*s,accent,2)
        line(x-20*s,y+a*s,x-14*s,y+a*s,accent,2)
        line(x+14*s,y+a*s,x+20*s,y+a*s,accent,2)
def wheel(x,y,s=1,accent='blue'):
    circle(x,y,17*s,'white',accent,2.5);circle(x,y,5*s,'white',accent,2)
    for i in range(5):
        a=2*math.pi*i/5;line(x+5*s*math.cos(a),y+5*s*math.sin(a),x+14*s*math.cos(a),y+14*s*math.sin(a),accent,2)
def commands(x,y,s=1,accent='blue'):
    for i in range(3):
        xx=x+(i-1)*12*s;line(xx,y-16*s,xx,y+16*s,accent,2)
        yy=y+(-7 if i==0 else 7 if i==1 else 0)*s;circle(xx,yy,4*s,'white',accent,2)
def target(x,y,s=1,accent='blue'):
    circle(x,y,14*s,None,accent,2);circle(x,y,6*s,None,accent,2)
    line(x-20*s,y,x-10*s,y,accent,2);line(x+10*s,y,x+20*s,y,accent,2)
    line(x,y-20*s,x,y-10*s,accent,2);line(x,y+10*s,x,y+20*s,accent,2)
def route_icon(x,y,s=1,accent='blue'):
    path([('M',x-17*s,y+13*s),('C',x+23*s,y+13*s,x-23*s,y-14*s,x+17*s,y-14*s)],accent,2.5)
    circle(x-17*s,y+13*s,3.7*s,'white',accent,2);circle(x+17*s,y-14*s,3.7*s,'white',accent,2)
def steering(x,y,s=1,accent='blue'):
    circle(x,y,31*s,None,accent,3.2);circle(x,y,7*s,accent,None)
    for a in (0,math.pi,math.pi/2):
        line(x+9*s*math.cos(a),y+9*s*math.sin(a),x+27*s*math.cos(a),y+27*s*math.sin(a),accent,3.5)
def pedal(x,y,s=1,accent='teal',wide=False):
    w=32 if wide else 22
    poly([(x-w*s,y-25*s),(x+w*s,y-25*s),(x+(w-4)*s,y+26*s),(x-(w-4)*s,y+26*s)],accent,2.3,'white')
    for i in range(4):line(x-(w-7)*s,y+(-15+i*10)*s,x+(w-7)*s,y+(-15+i*10)*s,accent,2)
    line(x,y+28*s,x+8*s,y+42*s,accent,3)
def car(cx,cy,s=1,angle=0,accent='blue'):
    # Original vector illustration: top view of a blue/white racing coupe.
    a=math.radians(angle)
    def tr(p):
        x,y=p;return(cx+s*(x*math.cos(a)-y*math.sin(a)),cy+s*(x*math.sin(a)+y*math.cos(a)))
    def pg(p,f,st='ink',wd=1.8):poly([tr(v) for v in p],st,wd,f)
    for xx in (-31,22):
        for yy in (-38,31):pg([(xx,yy),(xx+9,yy),(xx+9,yy+22),(xx,yy+22)],'ink',None)
    pg([(-22,-65),(22,-65),(29,-48),(29,45),(23,61),(-23,61),(-29,45),(-29,-48)],accent,'ink',2)
    pg([(-4,-62),(4,-62),(4,58),(-4,58)],'white',None)
    pg([(-21,-24),(21,-24),(18,-1),(-18,-1)],'#B5D7EB','ink',1.3)
    pg([(-18,3),(18,3),(21,29),(-21,29)],'#183047','ink',1.3)
    pg([(-18,-2),(18,-2),(17,4),(-17,4)],'white',None)
    pg([(-34,47),(34,47),(34,55),(-34,55)],'white','ink',1.8)
    pg([(-22,-59),(-10,-59),(-10,-54),(-23,-54)],'white',None)
    pg([(10,-59),(22,-59),(23,-54),(10,-54)],'white',None)

def gaussian(x,y,w,h):
    line(x,y,x+w,y,'muted',1.8)
    for sigma,accent,peak in [(w/7,'blue',h),(w/5,'coral',h*.78)]:
        pts=[]
        for i in range(100):
            xx=x+w*i/99;yy=y-peak*math.exp(-.5*((xx-x-w/2)/sigma)**2);pts.append((xx,yy))
        path([('M',*pts[0])]+[('L',*p)for p in pts[1:]],accent,3)
    line(x+w/2,y-h-8,x+w/2,y+7,'muted',1.5,[3,5])
    text(x+w/2,y+31,'μ',24,'ink','Italic','center')

# Header.
rect(0,0,W,H,'white',None,0)
rect(56,49,8,58,'coral',None,3)
text(83,97,'PPO9',48,'ink','Bold')
text(280,94,'Learning to drift from observable history',34,'ink')
text(2344,72,'Racer-HitchOpen-RLGPU',20,'muted',anchor='right')
text(2344,102,'RECURRENT POLICY  /  ASYMMETRIC TRAINING',16,'muted',anchor='right')
line(56,130,2344,130,'line',1.6)

section(56,188,'01','Observable inputs','blue')
section(540,188,'02','Shared-mirror recurrent actor','coral')
section(1835,188,'03','Actions','blue')
rect(56,215,430,580,'paleBlue','line',16)
rect(540,215,1240,580,'white','line',16)
rect(1835,215,509,580,'soft','line',16)

# Input card: the path preview is a geometric reference, not future state.
text(80,251,'VEHICLE SIGNALS + ROAD GEOMETRY',17,'blue','Bold')
path([('M',99,414),('C',170,428,196,357,260,340),('C',331,321,332,274,449,279)],'road',26)
path([('M',99,414),('C',170,428,196,357,260,340),('C',331,321,332,274,449,279)],'blue',2.8,[ ] if False else None)
for i,(xx,yy) in enumerate([(205,370),(229,351),(257,341),(288,329),(313,312),(338,295),(373,285),(410,280),(447,279)]):
    circle(xx,yy,5.4,'teal','white',1.5)
car(157,395,.54,42)
text(422,323,'9 × 4',26,'teal','Bold','center')
text(271,458,'Body-frame preview · 0.25–3.0 s',21,'muted',anchor='center')
rows=[('IMU', '4',chip),('Wheel encoders','4',wheel),('Previous commands','3',commands),('Tracking + speed cues','4',target),('Road preview','36',route_icon)]
for i,(label,dim,icon) in enumerate(rows):
    yy=500+i*43
    icon(99,yy-6,.67,'blue')
    text(126,yy,label,21,'ink')
    pill(399,yy-25,57,dim,'white','blue',19,31)
    if i<4:line(126,yy+13,457,yy+13,'#D7E5F1',1)
rect(79,731,384,46,'blue',None,10)
text(271,762,'oₜ : 15 + 9 × 4 = 51 dimensions',21,'white','Bold','center')

# Actor forward schematic.  The companion figure expands both shared calls.
text(565,252,'ONE PARAMETER SET, EVALUATED ON ORIGINAL + REFLECTED INPUTS',17,'coral','Bold')
rect(570,455,170,100,'paleBlue','blue',12)
text(655,491,'Symmetric',24,'blue','Bold','center')
text(655,523,'normalization',22,'blue',anchor='center')
text(674,584,'xₜ · 51-D',22,'muted')
arrow([(486,505),(570,505)],'blue')
arrow([(740,505),(800,505)],'blue')

rect(807,447,190,100,'#FBE0D7',None,12)
rect(800,455,190,100,'paleCoral','coral',12)
text(895,495,'GRU',29,'coral','Bold','center')
text(895,533,'128 hidden units',21,'coral',anchor='center')
arrow([(850,455),(850,413),(944,413),(944,455)],'coral',2.1)
text(896,397,'recurrent memory',20,'muted',anchor='center')
text(895,589,'Two 128-D histories',20,'muted',anchor='center')
text(895,616,'reset on done',19,'muted',anchor='center')

rect(1055,310,320,110,'paleAmber','amber',12)
text(1215,343,'Sideslip estimator',25,'amber','Bold','center')
text(1215,375,'shared head: 128 → 64 → 1',20,'ink',anchor='center')
text(1215,404,'β̂ = (b⁺ − b⁻) / 2',23,'amber',anchor='center')
arrow([(990,505),(1020,505),(1020,366),(1055,366)],'amber',2.2)
dot(1020,505,'coral')
arrow([(990,505),(1130,505)],'coral')
text(1081,529,'h⁺ / h⁻',19,'muted',anchor='center')
arrow([(1215,420),(1215,470)],'amber',2.4)
pill(1054,433,128,'stop-grad','paleAmber','amber',17,28)
text(1237,451,'±β̂',19,'amber')

rect(1130,470,120,95,'paleCoral','coral',12)
text(1190,515,'⊕',39,'coral','Bold','center')
text(1190,548,'180-D',21,'coral','Bold','center')
arrow([(655,555),(655,663),(1190,663),(1190,565)],'blue',2.2)
text(916,653,'51-D normalized observation bypass',21,'blue',anchor='center')
text(1200,692,'128 + 51 + 1',19,'muted',anchor='center')

arrow([(1250,505),(1315,505)],'coral')
text(1475,423,'Action MLP',25,'coral','Bold','center')
for i,(xx,hh,lab) in enumerate([(1315,116,'256'),(1400,88,'124'),(1485,62,'64')]):
    yy=505-hh/2
    rect(xx+5,yy-5,40,hh,'#F3C5B7',None,5)
    rect(xx,yy,40,hh,'paleCoral','coral',5)
    for k in range(4):circle(xx+20,yy+hh*(k+1)/5,3.0,'coral',None)
    text(xx+20,603,lab,22,'coral','Bold','center')
    if i<2:arrow([(xx+40,505),(xx+85,505)],'coral',2,head=8)
text(1422,637,'ELU activations',20,'muted',anchor='center')
arrow([(1525,505),(1570,505)],'coral',2.2)
rect(1570,467,72,76,'paleCoral','coral',9)
text(1606,515,'3',31,'coral','Bold','center')
text(1606,580,'raw means',19,'muted',anchor='center')
arrow([(1642,505),(1683,505)],'coral',2.2)
circle(1712,505,29,'paleBlue','blue',2.3)
text(1712,514,'½Σ',23,'blue','Bold','center')
text(1712,580,'Align +',20,'blue',anchor='center')
text(1712,607,'average',20,'blue',anchor='center')
arrow([(1741,505),(1808,505),(1808,345),(1889,345)],'blue',2.6)
text(1771,487,'μₜ',23,'blue',anchor='center')
rect(565,720,1190,51,'soft',None,9)
text(1160,752,'Mirror fusion:  μₜ = ½ (m⁺ + Dₐm⁻)     Dₐ = diag(−1, +1, +1)',23,'ink',anchor='center')

# Policy and physical action channels.
text(2090,255,'Gaussian policy',27,'ink','Bold','center')
gaussian(1915,367,342,82)
text(2090,427,'aₜ ~ N(μₜ, diag σ²)',25,'ink',anchor='center')
rect(1862,448,455,72,'paleAmber',None,10)
text(2089,477,'Learned log σ · state independent',21,'amber','Bold','center')
text(2089,504,'Initial σ = (0.35, 0.45, 0.25)',20,'ink',anchor='center')
text(2090,556,'Train: sample  ·  Evaluate: mean μₜ',20,'muted',anchor='center')
for xx in (1920,2090,2260):
    line(xx,575,xx,589,'line',1.5)
steering(1920,622,.72,'blue');pedal(2090,616,.72,'teal');pedal(2260,616,.72,'coral',True)
for xx,label,rg,acc in [(1920,'Steering','±0.45 rad','blue'),(2090,'Throttle','0 to 1','teal'),(2260,'Brake','0 to 1','coral')]:
    text(xx,678,label,22,acc,'Bold','center');text(xx,708,rg,20,'muted',anchor='center')
rect(1862,738,455,38,'paleBlue',None,8)
text(2089,764,'Clip / scale → actuator model',21,'blue','Bold','center')

# Lower row, deliberately separated from deployable actor inference.
section(56,895,'04','Training-only privileged information','purple')
section(1835,895,'05','Vehicle & environment','blue')
rect(56,925,1724,320,'#FCFBFE','purple',16,1.8,[7,7])
rect(1835,925,509,320,'paleBlue','line',16)
arrow([(2290,795),(2290,925)],'blue',2.5)
text(2270,850,'physical commands',21,'blue',anchor='right')

rect(80,972,330,220,'paleTeal',None,12)
text(245,1010,'Privileged state sₜ',24,'teal','Bold','center')
text(245,1050,'82-D = 37 + 9 × 5',25,'teal','Bold','center')
text(245,1088,'True dynamics + tire forces',18,'ink',anchor='center')
text(245,1119,'Reference preview + βref',18,'ink',anchor='center')
pill(104,1145,282,'βtrue, speed: supervision','white','teal',18,32)

rect(460,972,460,135,'paleTeal','teal',12)
text(690,1008,'Independent mirrored critic',24,'teal','Bold','center')
text(690,1054,'82 → 256 → 128 → 64 → 1',25,'ink',anchor='center')
text(690,1088,'Shared MLP calls · average scalar outputs',18,'teal',anchor='center')
arrow([(410,1040),(460,1040)],'teal')
arrow([(920,1040),(946,1040)],'teal')
circle(975,1040,29,'paleTeal','teal',2)
text(975,1049,'V',26,'teal','Italic','center')
arrow([(1004,1040),(1040,1040)],'teal')

rect(1040,980,230,127,'palePurple','purple',12)
text(1155,1020,'GAE',28,'purple','Bold','center')
text(1155,1058,'γ .999 · λ .97',21,'purple',anchor='center')
text(1155,1089,'+ rewards / done',18,'muted',anchor='center')
arrow([(1270,1040),(1330,1040)],'purple')
text(1300,1017,'Âₜ',23,'purple',anchor='center')

rect(1330,972,420,135,'palePurple','purple',12)
text(1540,1009,'Actor update',26,'purple','Bold','center')
text(1540,1048,'Clipped PPO (ε = 0.2) + bounds',21,'ink',anchor='center')
text(1540,1082,'+ 0.1 Lβ',24,'purple',anchor='center')
arrow([(1580,972),(1580,840),(1470,840),(1470,795)],'purple',2.3,[7,6])
text(1348,864,'update actor θ',21,'purple')

rect(460,1142,460,70,'white','teal',11)
text(690,1186,'Clipped value regression → critic only',20,'teal','Bold','center')
arrow([(690,1142),(690,1107)],'teal',2,[5,5],head=8)
arrow([(1155,1107),(1155,1125),(960,1125),(960,1177),(920,1177)],'purple',1.7,[5,5],head=8)
text(1025,1119,'returns R̂ₜ',16,'purple',anchor='center')
rect(1040,1142,710,70,'paleAmber','amber',11)
text(1395,1171,'Sideslip supervision: Lβ(β̂, βtrue; speed)',22,'amber','Bold','center')
text(1395,1199,'Wrapped Huber loss · low-speed weighting',19,'ink',anchor='center')
arrow([(1600,1142),(1600,1107)],'amber',2,[5,5],head=8)
arrow([(245,1192),(245,1230),(1080,1230),(1080,1212)],'amber',1.8,[5,5],head=8)

# Environment icon, a small racing scene made from vectors.
text(2090,962,'FastPlant / CarSim',26,'ink','Bold','center')
path([('M',1872,1113),('C',1924,1034,2055,1138,2157,1060),('C',2220,1007,2275,1030,2314,989)],'road',29)
path([('M',1872,1113),('C',1924,1034,2055,1138,2157,1060),('C',2220,1007,2275,1030,2314,989)],'blue',2.1,dash=[8,8])
car(2100,1060,.79,48)
for xx,yy in [(1961,1065),(2239,1040)]:
    poly([(xx-9,yy+13),(xx,yy-12),(xx+9,yy+13)],'coral',1.2,'paleCoral')
    line(xx-12,yy+14,xx+12,yy+14,'coral',2)
text(2090,1170,'20 Hz control · 100 Hz physics',22,'blue','Bold','center')
text(2090,1207,'Training: 32,768 parallel environments',18,'muted',anchor='center')

# Feedback channels.  State data uses its own dashed training path.
arrow([(1935,1245),(1935,1273),(204,1273),(204,1192)],'purple',1.8,[6,6],head=8)
rect(1090,1256,600,30,'white',None,0)
text(1390,1278,'Privileged simulator state · training only',19,'purple',anchor='center')
arrow([(2090,1245),(2090,1315),(25,1315),(25,505),(56,505)],'blue',2.1)
rect(636,1295,760,35,'white',None,0)
text(1016,1320,'Vehicle response → next observable input oₜ₊₁',23,'blue',anchor='center')

# Legend + compact default-profile context, outside network panels.
line(56,1351,2344,1351,'line',1.4)
arrow([(62,1380),(109,1380)],'blue',2.2,head=8);text(126,1387,'Online forward path',19,'muted')
arrow([(416,1380),(463,1380)],'purple',2,[6,5],head=8);text(480,1387,'Training-only signals / updates',19,'muted')
text(1160,1387,'Default PPO9 · no attention encoder',18,'muted',anchor='center')
text(2344,1387,'Ranges shown before actuator processing',18,'muted',anchor='right')
cv.showPage();cv.save()
print(ROOT/'PPO9_Overview.pdf')
