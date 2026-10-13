// QuPath 0.4+ | tiles_v2 (Lesion / Normal / Normal-Far / Normal-Red)
// 背景安全 + 组织覆盖率 + 边框屏蔽 + 边缘伪影过滤 + 预扫/回退/兜底 + 安全读取
// 直接在已打开的 WSI 上运行；输出到 Project/tiles_v2/<WSI名>/*

import qupath.lib.objects.PathObject
import qupath.lib.roi.interfaces.ROI
import qupath.lib.regions.RegionRequest
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.text.SimpleDateFormat
import java.util.Collections
import static java.lang.Math.*

// ===== 基本参数 =====
def EXCLUDE_PREFIXES = ['Region', 'Ignore']   // 这些前缀的注释不当 Lesion
int    TILE = 224
double UM_PER_PX = 0.5                        // 目标分辨率（~20x）
double NEG_POS = 1.0                          // Normal : Lesion
int    PAD = 6000

boolean WRITE_BASE_POS = true
boolean WRITE_BASE_NEG = true
String  RUN_TAG = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date())

// ===== 难负样本目标与阈值（可调）=====
boolean ENABLE_FAR  = true
boolean ENABLE_HARD = true
double FAR_MULT     = 0.50                    // 目标 = 正样本 * 0.5
double HARD_MULT    = 0.50
double MIN_DIST_UM  = 2000                    // Far/Hard 与病灶最小距离（µm）
double A_Q          = 0.65                    // 偏红/粉 分位阈值
double T_Q          = 0.50                    // 纹理std 分位阈值

int    MAX_BACKOFF_ROUNDS = 3                 // 回退最多轮次
double BACKOFF_DIST_FACTOR = 0.7              // 每轮距离 ×0.7
double BACKOFF_AQ          = 0.05             // 每轮 a* 阈值 ↓0.05
double BACKOFF_TQ          = 0.05             // 每轮 std 阈值 ↓0.05
double MIN_DIST_UM_FLOOR   = 1200             // 距离下限
double A_Q_FLOOR           = 0.50
double T_Q_FLOOR           = 0.45

// ===== 预扫密度 =====
int    SAMPLE_STEP   = 4
double PREPASS_STRIDE_FACTOR = 0.75           // 0.75=较密, 1.0=常规

// ===== 画幅边框/组织/伪影过滤 =====
boolean SKIP_BORDER = true
double  BORDER_UM = 1500                      // 画幅边距 <1.5mm 一律丢弃
double  MIN_TISSUE_FRAC = 0.15                // 组织覆盖率阈值（基于低OD）

// ============ 背景与组织覆盖率 ============
// 返回 [isBg(boolean), tissueFrac(double)]
def analyzeBackground = { BufferedImage img ->
  def ras = img.getRaster(); int w = img.getWidth(), h = img.getHeight()
  int step = 2; long n = 0
  long whiteCnt = 0, lowODCnt = 0
  double gSum = 0, gSq = 0
  int[] rgb = new int[3]
  for (int yy = 0; yy < h; yy += step)
    for (int xx = 0; xx < w; xx += step) {
      ras.getPixel(xx, yy, rgb)
      int r = rgb[0], g = rgb[1], b = rgb[2]
      n++
      if (r >= 245 && g >= 245 && b >= 245) whiteCnt++
      double gray = 0.299*r + 0.587*g + 0.114*b
      gSum += gray; gSq += gray*gray
      double odR = -Math.log((r + 1) / 256.0)
      double odG = -Math.log((g + 1) / 256.0)
      double odB = -Math.log((b + 1) / 256.0)
      if ((odR + odG + odB) < 0.25) lowODCnt++
    }
  double m = gSum / Math.max(1, n)
  double std = Math.sqrt(Math.max(0.0, gSq / Math.max(1, n) - m*m))
  double fracWhite = whiteCnt / (double)n
  double fracLowOD = lowODCnt / (double)n
  boolean isBg = (fracWhite > 0.60) || (m > 220 && std < 12) || (fracLowOD > 0.85)
  double tissueFrac = 1.0 - fracLowOD
  return [isBg, tissueFrac]
}

// ============ 边缘伪影（黑条/玻片边）检测 ============
def hasEdgeArtifact = { BufferedImage img ->
  def ras = img.getRaster(); int w = img.getWidth(), h = img.getHeight()
  int band = Math.max(1, (int)Math.round(h * 0.10)) // 上下各10%
  int[] rgb = new int[3]
  double sumTop = 0, sumBot = 0; long nTop = 0, nBot = 0
  for (int yy=0; yy<band; yy++) {
    for (int xx=0; xx<w; xx+=2) { ras.getPixel(xx, yy, rgb); sumTop += (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]); nTop++ }
  }
  for (int yy=h-band; yy<h; yy++) {
    for (int xx=0; xx<w; xx+=2) { ras.getPixel(xx, yy, rgb); sumBot += (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]); nBot++ }
  }
  double meanTop = (nTop>0)?(sumTop/nTop):255
  double meanBot = (nBot>0)?(sumBot/nBot):255
  return (meanTop < 140) || (meanBot < 140)    // 阈值可调：140→160 更严格
}

// ============ 其他工具 ============
def isExcluded = { name ->
  if (name == null) return false
  def n = name.replace('*','')
  return EXCLUDE_PREFIXES.any { n.startsWith(it) }
}
def inAny = { double x, double y, List<PathObject> objs ->
  for (po in objs) { ROI r = po.getROI(); if (r!=null && r.contains(x,y)) return true }
  false
}
def minDistToLesionsPx = { double cx, double cy, List<PathObject> objs ->
  double best = Double.POSITIVE_INFINITY
  for (po in objs) {
    def r = po.getROI(); if (r==null) continue
    if (r.contains(cx, cy)) return 0.0
    def bx=r.getBoundsX(), by=r.getBoundsY(), bw=r.getBoundsWidth(), bh=r.getBoundsHeight()
    double dx = Math.max(Math.max(bx - cx, 0), cx - (bx + bw))
    double dy = Math.max(Math.max(by - cy, 0), cy - (by + bh))
    best = Math.min(best, Math.hypot(dx, dy))
  }
  return best
}
def tileFeatures = { BufferedImage img, int step ->
  def ras = img.getRaster(); int w = img.getWidth(), h = img.getHeight()
  double aSum=0, graySum=0, graySq=0; long n=0
  for (int yy=0; yy<h; yy+=step)
    for (int xx=0; xx<w; xx+=step) {
      int[] rgb = new int[3]; ras.getPixel(xx,yy,rgb)
      int r=rgb[0], g=rgb[1], b=rgb[2]
      def f = { double u -> u<=0.04045 ? u/12.92 : Math.pow((u+0.055)/1.055, 2.4) }
      double R=f(r/255.0), G=f(g/255.0), B=f(b/255.0)
      double X=0.4124*R+0.3576*G+0.1805*B, Y=0.2126*R+0.7152*G+0.0722*B, Z=0.0193*R+0.1192*G+0.9505*B
      double xr=X/0.95047, yr=Y/1.00000, zr=Z/1.08883
      def gfun={ double t -> t>Math.pow(6.0/29,3)? Math.cbrt(t): (t/(3*Math.pow(6.0/29,2))+4.0/29) }
      double fx=gfun(xr), fy=gfun(yr), fz=gfun(zr)
      double a = 500*(fx - fy); aSum += a
      double gray = 0.299*r + 0.587*g + 0.114*b; graySum += gray; graySq += gray*gray
      n++
    }
  double aMean = aSum / Math.max(1,n)
  double gMean = graySum / Math.max(1,n)
  double gStd  = Math.sqrt(Math.max(0.0, (graySq/Math.max(1,n) - gMean*gMean)))
  return [aMean, gStd]
}
def quantile = { List<Double> xs, double q ->
  if (xs==null || xs.isEmpty()) return 0.0
  def ys = new ArrayList<Double>(xs); java.util.Collections.sort(ys)
  double pos = (ys.size()-1) * q
  int lo = (int)Math.floor(pos), hi = (int)Math.ceil(pos)
  if (lo==hi) return ys[lo]
  double w = pos - lo
  return ys[lo]*(1.0-w) + ys[hi]*w
}
def safeReadRegion = { serverObj, double downsample, int x, int y, int w, int h ->
  int tries = 0
  while (tries < 3) {
    try {
      def region = RegionRequest.createInstance(serverObj.getPath(), downsample, x, y, w, h)
      BufferedImage img = serverObj.readRegion(region)
      return img
    } catch (Throwable t) {
      tries++
      try { Thread.sleep(15L * tries) } catch (InterruptedException ie) {}
    }
  }
  return null
}

// ===== 环境与路径 =====
def imageData = getCurrentImageData()
def server = imageData.getServer()
double base = server.getPixelCalibration()?.getAveragedPixelSizeMicrons(); if (Double.isNaN(base)) base = 0.25
double down = Math.max(UM_PER_PX / base, 1e-6)
int    tileBase = Math.max(1, (int)Math.round(TILE * down))
double px_per_um = 1.0 / base
int    MIN_DIST_PX_INIT  = (int)Math.round(MIN_DIST_UM       * px_per_um)
int    MIN_DIST_PX_FLOOR = (int)Math.round(MIN_DIST_UM_FLOOR * px_per_um)
int    BORDER_PX         = (int)Math.round(BORDER_UM * px_per_um)

def imgName = getProjectEntry()?.getImageName() ?: server.getMetadata().getName()
def outRoot = buildFilePath(PROJECT_BASE_DIR, 'tiles_v2', imgName)
def outPos  = buildFilePath(outRoot, 'Lesion')
def outNeg  = buildFilePath(outRoot, 'Normal')
def outFar  = buildFilePath(outRoot, 'Normal-Far')
def outHard = buildFilePath(outRoot, 'Normal-Red')
mkdirs(outPos); mkdirs(outNeg); mkdirs(outFar); mkdirs(outHard)

println "=== QuPath tiler v2 (border-safe + artifact filter) ==="
println "Image: ${imgName}"
println String.format("Base=%.4f um/px  Target=%.4f  down=%.6f  tileBase=%d  borderPx=%d", base, UM_PER_PX, down, tileBase, BORDER_PX)
println "Output: " + outRoot
println "RUN_TAG: " + RUN_TAG

// ===== ROI =====
def lesions = getAnnotationObjects().findAll { po -> !isExcluded(po.getPathClass()?.getName()) }
if (lesions.isEmpty()) { println "[SKIP] No annotations (after excluding ${EXCLUDE_PREFIXES})."; return }

// 画幅边框禁区（按中心点）
def inBorder = { double cx, double cy ->
  if (!SKIP_BORDER) return false
  return (cx < BORDER_PX || cy < BORDER_PX ||
          cx > server.getWidth()  - BORDER_PX ||
          cy > server.getHeight() - BORDER_PX)
}

// ===== 1) Lesion（ROI 内 + 非背景）=====
int ps=0, pb=0, pe=0
for (obj in lesions) {
  def roi = obj.getROI(); if (roi==null) continue
  int bx=(int)Math.floor(roi.getBoundsX()), by=(int)Math.floor(roi.getBoundsY())
  int bw=(int)Math.ceil(roi.getBoundsWidth()), bh=(int)Math.ceil(roi.getBoundsHeight())
  for (int y=by; y<=by+bh-tileBase; y+=tileBase)
    for (int x=bx; x<=bx+bw-tileBase; x+=tileBase) {
      double cx=x+tileBase/2.0, cy=y+tileBase/2.0
      if (!roi.contains(cx,cy)) continue
      if (inBorder(cx,cy)) continue
      BufferedImage img = safeReadRegion(server, down, x, y, tileBase, tileBase)
      if (img==null) { pe++; continue }
      if (img.getWidth()!=TILE || img.getHeight()!=TILE) { pe++; continue }
      def bgan = analyzeBackground(img); if (bgan[0]) { pb++; continue }
      ImageIO.write(img, 'PNG', new File(outPos, String.format('P_x%07d_y%07d_%s.png',x,y,RUN_TAG))); ps++
    }
}
println String.format("[Lesion] saved=%d, background=%d, edge=%d", ps,pb,pe)
if (ps==0) { println "[WARN] zero positives; stop."; return }

// ===== 2) Normal（ROI 外 + 非背景 + 覆盖率 + 非边框 + 无伪影）=====
int ns=0, nb=0, nl=0, ne=0, na=0
int minX=Integer.MAX_VALUE,minY=Integer.MAX_VALUE,maxX=Integer.MIN_VALUE,maxY=Integer.MIN_VALUE
for (obj in lesions) {
  def r=obj.getROI()
  double x0=r.getBoundsX(), y0=r.getBoundsY(), x1=x0+r.getBoundsWidth(), y1=y0+r.getBoundsHeight()
  if (x0<minX) minX=(int)Math.floor(x0); if (y0<minY) minY=(int)Math.floor(y0)
  if (x1>maxX) maxX=(int)Math.ceil(x1);  if (y1>maxY) maxY=(int)Math.ceil(y1)
}
minX=Math.max(0,minX-PAD); minY=Math.max(0,minY-PAD)
maxX=Math.min(server.getWidth(),  maxX+PAD); maxY=Math.min(server.getHeight(), maxY+PAD)

int targetN=(int)Math.round(ps*NEG_POS)
for (int y=minY; y<=maxY-tileBase && ns<targetN; y+=tileBase)
  for (int x=minX; x<=maxX-tileBase && ns<targetN; x+=tileBase) {
    double cx=x+tileBase/2.0, cy=y+tileBase/2.0
    if (inAny(cx,cy,lesions)) { nl++; continue }
    if (inBorder(cx,cy)) continue
    BufferedImage img = safeReadRegion(server, down, x, y, tileBase, tileBase)
    if (img==null) { ne++; continue }
    if (img.getWidth()!=TILE || img.getHeight()!=TILE) { ne++; continue }
    def bgan = analyzeBackground(img); boolean isBg=bgan[0]; double tissueFrac=bgan[1]
    if (isBg || tissueFrac < MIN_TISSUE_FRAC) { nb++; continue }
    if (hasEdgeArtifact(img)) { na++; continue }
    ImageIO.write(img,'PNG', new File(outNeg, String.format('N_x%07d_y%07d_%s.png',x,y,RUN_TAG))); ns++
  }
println String.format("[Normal] saved=%d/target=%d, background/tissue=%d, in_lesion=%d, artifact=%d, edge=%d", ns,targetN,nb,nl,na,ne)

// ===== 3) 预扫候选（ROI 外 + 同样过滤）=====
class TileCand { int x; int y; double a; double s; double distPx }
def cands = new ArrayList<TileCand>()
int GRID_STEP = Math.max(1, (int)Math.round(tileBase * PREPASS_STRIDE_FACTOR))
for (int y=0; y<=server.getHeight()-tileBase; y+=GRID_STEP) {
  for (int x=0; x<=server.getWidth()-tileBase; x+=GRID_STEP) {
    double cx=x+tileBase/2.0, cy=y+tileBase/2.0
    if (inAny(cx,cy,lesions)) continue
    if (inBorder(cx,cy)) continue
    BufferedImage img = safeReadRegion(server, down, x, y, tileBase, tileBase)
    if (img==null) continue
    if (img.getWidth()!=TILE || img.getHeight()!=TILE) continue
    def bgan = analyzeBackground(img); boolean isBg=bgan[0]; double tissueFrac=bgan[1]
    if (isBg || tissueFrac < MIN_TISSUE_FRAC) continue
    if (hasEdgeArtifact(img)) continue
    def ft = tileFeatures(img, SAMPLE_STEP)
    double dist = minDistToLesionsPx(cx, cy, lesions)
    def t = new TileCand(); t.x=x; t.y=y; t.a=(ft[0] as Double); t.s=(ft[1] as Double); t.distPx=dist
    cands.add(t)
  }
}
println String.format("[Pre-scan] candidates: %d (stride=%.2f*tile)", cands.size(), PREPASS_STRIDE_FACTOR)
if (cands.isEmpty()) { println "[WARN] no candidates; stop."; return }

def A_vals = cands.collect{ it.a } as List<Double>
def S_vals = cands.collect{ it.s } as List<Double>
double A_th_base = quantile(A_vals, A_Q)
double S_th_base = quantile(S_vals, T_Q)
println String.format("Adaptive thresholds (base): a* P%.0f = %.3f, std P%.0f = %.3f", A_Q*100, A_th_base, T_Q*100, S_th_base)

// ===== 4) Far/Hard 选择（回退 + 兜底）=====
int target_far  = ENABLE_FAR  ? (int)Math.round(ps*FAR_MULT)  : 0
int target_hard = ENABLE_HARD ? (int)Math.round(ps*HARD_MULT) : 0
int MIN_DIST_PX = MIN_DIST_PX_INIT
double A_th = A_th_base
double S_th = S_th_base

// 采样写盘（注意参数名用 tgt，避免 Groovy 冲突）
def pickAndWrite = { List<TileCand> pool, File outDir, String prefix, int tgt ->
  if (tgt<=0) return 0
  Collections.shuffle(pool, new java.util.Random(42L))
  int saved=0
  for (tc in pool) {
    if (saved>=tgt) break
    BufferedImage img = safeReadRegion(server, down, tc.x, tc.y, tileBase, tileBase)
    if (img==null) continue
    if (img.getWidth()!=TILE || img.getHeight()!=TILE) continue
    def bgan = analyzeBackground(img); boolean isBg=bgan[0]; double tissueFrac=bgan[1]
    if (isBg || tissueFrac < MIN_TISSUE_FRAC) continue
    if (hasEdgeArtifact(img)) continue
    ImageIO.write(img,'PNG', new File(outDir, String.format(prefix+'_x%07d_y%07d_%s.png', tc.x, tc.y, RUN_TAG)))
    saved++
  }
  return saved
}

int gotFar=0, gotHard=0
for (int round=1; round<=MAX_BACKOFF_ROUNDS; round++) {
  def farPool  = cands.findAll{ it.distPx >= MIN_DIST_PX }
  def hardPool = cands.findAll{ it.distPx >= MIN_DIST_PX && (it.a >= A_th || it.s >= S_th) }

  int needFar  = Math.max(0, target_far  - gotFar)
  int needHard = Math.max(0, target_hard - gotHard)

  int addFar   = pickAndWrite(farPool,  new File(outFar),  (round==1?"Nfar":"NfarB"+round),  needFar)
  int addHard  = pickAndWrite(hardPool, new File(outHard), (round==1?"Nhard":"NhardB"+round), needHard)
  gotFar  += addFar
  gotHard += addHard

  println String.format("[Try#%d] Far += %d  (now %d/%d),  Hard += %d  (now %d/%d)  | dist>=%.0fpx (%.1fmm), a*>%.3f or std>%.3f",
    round, addFar, gotFar, target_far, addHard, gotHard, target_hard,
    (double)MIN_DIST_PX, MIN_DIST_PX/px_per_um/1000.0, A_th, S_th)

  if (gotFar>=target_far && gotHard>=target_hard) break

  // ——修复版回退：显式 Math.max/Math.round，避免 Integer.call 报错——
  MIN_DIST_PX = (int)Math.max((double)MIN_DIST_PX_FLOOR,
                              Math.round((double)MIN_DIST_PX * BACKOFF_DIST_FACTOR))
  A_th = Math.max(quantile(A_vals, A_Q_FLOOR), A_th - BACKOFF_AQ)
  S_th = Math.max(quantile(S_vals, T_Q_FLOOR), S_th - BACKOFF_TQ)
}

// 兜底
if (gotFar < target_far && ENABLE_FAR) {
  int needFar = target_far - gotFar
  def farPoolLoose = cands.findAll{ it.distPx >= MIN_DIST_PX_FLOOR }
  int addFar = pickAndWrite(farPoolLoose, new File(outFar), "NfarF", needFar)
  gotFar += addFar
  println String.format("[Fallback] Far += %d (loose dist >= %.0fpx, %.1fmm)", addFar, (double)MIN_DIST_PX_FLOOR, MIN_DIST_PX_FLOOR/px_per_um/1000.0)
}
if (gotHard < target_hard && ENABLE_HARD) {
  int needHard = target_hard - gotHard
  def byRed = new ArrayList<TileCand>(cands); Collections.sort(byRed, (a,b)-> Double.compare(b.a, a.a))
  def byTex = new ArrayList<TileCand>(cands); Collections.sort(byTex, (a,b)-> Double.compare(b.s, a.s))
  def topK = new LinkedHashSet<TileCand>()
  int k = Math.min(needHard*3, Math.max(needHard, (int)Math.round(cands.size()*0.05)))
  for (int i=0; i<Math.min(k, byRed.size()); i++) topK.add(byRed.get(i))
  for (int i=0; i<Math.min(k, byTex.size()); i++) topK.add(byTex.get(i))
  int addHard = pickAndWrite(new ArrayList<TileCand>(topK), new File(outHard), "NhardF", needHard)
  gotHard += addHard
  println String.format("[Fallback] Hard += %d (global top-K by redness/texture)", addHard)
}

println String.format("[Final] Normal-Far = %d/%d, Normal-Red(hard) = %d/%d", gotFar, target_far, gotHard, target_hard)
println "=== Done: ${imgName} ==="
