
import os, re, math, subprocess, tempfile, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from moduli.ffmpeg_utils import ffmpeg_path as _ffmpeg, ffprobe_path as _ffprobe

W = int(os.getenv("VIDEO_WIDTH","1920"))
H = int(os.getenv("VIDEO_HEIGHT","1080"))
FPS = int(os.getenv("VIDEO_FPS","30"))
CRF = int(os.getenv("FERN_CRF","20"))
PRESET = os.getenv("FERN_PRESET","fast")
WORKERS = max(1,int(os.getenv("FERN_WORKERS","2")))
MIN_SHOT = float(os.getenv("FERN_SHOT_MIN","1.6"))
MAX_SHOT = float(os.getenv("FERN_SHOT_MAX","8.0"))
VIDEO_EXTS = {".mp4",".mov",".mkv",".webm",".avi",".m4v"}
IMAGE_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}

def run(cmd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:] or "ffmpeg failed")
    return p.stdout

def probe(path):
    out = run([_ffprobe(),"-v","error","-show_entries","format=duration:stream=width,height",
               "-of","default=noprint_wrappers=1",path],60).splitlines()
    d,w,h = .01,0,0
    for x in out:
        if x.startswith("duration="):
            try: d=float(x.split("=",1)[1])
            except: pass
        elif x.startswith("width=") and not w:
            try: w=int(x.split("=",1)[1])
            except: pass
        elif x.startswith("height=") and not h:
            try: h=int(x.split("=",1)[1])
            except: pass
    return max(d,.01),max(w,1),max(h,1)

def words(s):
    return re.findall(r"[a-z0-9']+",(s or "").lower())

def sentences(script):
    t=re.sub(r"\s+"," ",script or "").strip()
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+",t) if x.strip()]

def beats_for(total, ss):
    if not ss: return [(total,"")]
    weights=[max(3,len(words(s))) for s in ss]
    sw=sum(weights)
    out=[]
    for s,wt in zip(ss,weights):
        d=total*wt/sw
        n=max(1,math.ceil(d/MAX_SHOT))
        piece=d/n
        for _ in range(n): out.append((piece,s))
    # merge tiny tail without exceeding max
    if out and out[-1][0] < MIN_SHOT and len(out)>1:
        d,s=out.pop(); prev_d,prev_s=out[-1]
        out[-1]=(prev_d+d,prev_s)
    scale=total/max(sum(d for d,_ in out),.01)
    out=[(d*scale,s) for d,s in out]
    return out

def token_set(text):
    return set(words(text))

def assets(keywords, clip_paths):
    out=[]
    seen=set()
    for k in keywords or []:
        p=clip_paths.get(k)
        if not p or not os.path.exists(p) or p in seen: continue
        seen.add(p)
        ext=os.path.splitext(p)[1].lower()
        try:
            d,w,h=probe(p)
        except Exception:
            continue
        out.append({"key":str(k),"path":p,"kind":"image" if ext in IMAGE_EXTS else "video",
                    "duration":d,"w":w,"h":h,"tokens":token_set(str(k)+" "+os.path.basename(p))})
    return out

def pick(pool, sentence, idx, used):
    wanted=token_set(sentence)
    best=None
    for a in pool:
        score=20*len(wanted & a["tokens"])
        if a["path"] in used[-2:]: score-=80
        if a["path"]==used[-1:] [0] if used else False: score-=50
        # Encourage a different medium every few beats.
        if idx%4==2 and a["kind"]=="image": score+=6
        if idx%4!=2 and a["kind"]=="video": score+=2
        tie=int(hashlib.md5((a["path"]+str(idx)).encode()).hexdigest()[:8],16)/1e8
        score+=tie
        if best is None or score>best[0]: best=(score,a)
    return best[1]

def vf_for(a):
    aspect=a["w"]/max(1,a["h"])
    if a["kind"]=="image":
        return ("scale=2200:1238:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}:x='(in_w-out_w)/2+25*sin(on/95)':"
                "y='(in_h-out_h)/2+18*cos(on/120)',"
                f"zoompan=z='min(zoom+0.00065,1.07)':d=1:s={W}x{H}:fps={FPS},"
                "setsar=1,format=yuv420p")
    if aspect < 1.55:
        return ("split=2[bg][fg];"
                f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},boxblur=24:2,eq=brightness=-0.12:saturation=0.82[bg];"
                f"[fg]scale={int(W*.78)}:{int(H*.86)}:force_original_aspect_ratio=decrease,"
                "setsar=1[fg];"
                "[bg][fg]overlay=(W-w)/2+16*sin(t*.35):(H-h)/2+10*cos(t*.28),"
                "format=yuv420p")
    sw=int(W*1.035); sh=int(H*1.035)
    return (f"scale={sw}:{sh}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:x='(in_w-out_w)/2+18*sin(t*.32)':"
            "y='(in_h-out_h)/2+12*cos(t*.24)',setsar=1,format=yuv420p")

def render_one(a,dst,start,dur):
    vf=vf_for(a)
    if a["kind"]=="image":
        cmd=[_ffmpeg(),"-y","-nostdin","-loop","1","-framerate",str(FPS),"-i",a["path"],
             "-t",f"{dur:.3f}","-vf",vf,"-an","-c:v","libx264","-preset",PRESET,
             "-crf",str(CRF),"-pix_fmt","yuv420p","-movflags","+faststart",dst]
    else:
        maxstart=max(0,a["duration"]-dur)
        start=(start%maxstart) if maxstart>.05 else 0
        cmd=[_ffmpeg(),"-y","-nostdin","-ss",f"{start:.3f}","-i",a["path"],
             "-t",f"{dur:.3f}","-vf",vf,"-an","-c:v","libx264","-preset",PRESET,
             "-crf",str(CRF),"-pix_fmt","yuv420p","-movflags","+faststart",dst]
    run(cmd,900)

def render(audio_path, clip_paths, output_path, script="", keywords=None):
    total,_,_=probe(audio_path)
    pool=assets(keywords or list(clip_paths),clip_paths)
    if not pool: raise RuntimeError("Fern editor: no usable assets")
    beats=beats_for(total,sentences(script))
    plan=[]; used=[]
    for i,(dur,s) in enumerate(beats):
        a=pick(pool,s,i,used)
        start=(i*1.91) % max(a["duration"],.01) if a["kind"]=="video" else 0
        plan.append((i,a,start,dur,s)); used.append(a["path"])
    os.makedirs(os.path.dirname(output_path) or ".",exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fern_") as tmp:
        def job(item):
            i,a,start,dur,_=item
            dst=os.path.join(tmp,f"{i:04d}.mp4")
            render_one(a,dst,start,dur)
            return i,dst
        results={}
        with ThreadPoolExecutor(max_workers=min(WORKERS,len(plan))) as ex:
            fs=[ex.submit(job,x) for x in plan]
            for f in as_completed(fs):
                i,dst=f.result(); results[i]=dst
        lst=os.path.join(tmp,"concat.txt")
        with open(lst,"w",encoding="utf-8") as f:
            for i,_,_,_,_ in plan:
                f.write("file '"+os.path.abspath(results[i]).replace("\\","/").replace("'","'\\''")+"'\n")
        run([_ffmpeg(),"-y","-nostdin","-v","error","-f","concat","-safe","0",
             "-i",lst,"-c","copy","-movflags","+faststart",output_path],1800)
    print(f"[FERN] {len(plan)} editorial beats | {W}x{H} {FPS}fps",flush=True)
