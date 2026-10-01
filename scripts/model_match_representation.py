"""Model the brain's match scorer with one line of representation changed.

No build, under a second. Both numbers it prints are load-bearing for backlog
items f16e499d and 972f1bb8, and a scorecard run cannot produce either in under
three minutes. The scorer modelled is the real one as measured in
crates/brain/tests/synonym_span_derivation.rs -- precision x recall over the
frame's element set, with 0 of 186 trained questions reaching
MatchTier::Concept, so today the elements are UNORDERED DISTINCT BYTES.

Run:  python scripts/model_match_representation.py

MEASURED 2026-10-01 on the scale-1 corpus (186 trained questions, 1,762
recurring runs). Read these back off the script's own output, not off here:

    byte-set  probes 24 ; probes with >1 ceiling rewrite 16 ; UNTAUGHT at ceiling 112
    run-set   probes 24 ; probes with >1 ceiling rewrite  0 ; UNTAUGHT at ceiling   0
    byte-set  'r000 desk material?' vs 'r0desk material?': equal? True  score 1.0000
    run-set   'r000 desk material?' vs 'r0desk material?': equal? False score 0.7241
    r001 beside?  want r002  TOP 0.9000 'r000 beside?' -> r001  ties=1
                             its own 'r001 next?' 0.4000  rank 44 of 186

THE REPRESENTATION HALF. The untaught twin "r0desk material?" reaches exactly
1.0 today because its distinct byte set is IDENTICAL to the taught
"r000 desk material?". Under runs it falls to 0.7241 and stops being a ceiling
rewrite at all, because "r0d" and "0de" are runs the taught question never
produces. So derive_by_substitution's accept rule -- "a question scoring 1.0 IS
a question the brain was taught" -- becomes TRUE instead of false, and the
ambiguity a tie-break has to resolve goes from 16 of 24 probes to 0. That is
why 972f1bb8 is a representation change and not a selection policy.

THE beside_next HALF. Every held-out "rNNN beside?" probe gives the same four
numbers: the top match is UNIQUE at 0.9000 and it is the WRONG room, while the
question holding the right answer sits at rank 44 of 186. Nine of ten distinct
bytes are the relation word; the three bytes that are the entire question are
worth 0.1. So that family is neither a tie-break problem nor a budget problem,
which is what f16e499d's relation transfer is for.

THIS IS A MODEL, NOT THE BRAIN. The real collapse is longest-tail-wins
producing a SEQUENCE rather than the set of all runs, the real score has
terminal propagation in it, and a sharper representation can refuse a
paraphrase it used to accept -- the OOV-honesty-versus-recall trade this
repository has already paid for twice. The narrow claim is mechanical: elements
that are ordered runs cannot collide the way unordered bytes can.
"""

ROOMS=8; M=8
OBJECTS=["bed","chair","mirror","desk","lamp","door","window","paper"]
COLORS=["red","blue","green","white","black","grey"]
MATERIALS=["oak","steel","glass","cloth","pine","brass"]
RESTS_ON=[("lamp","desk"),("paper","desk"),("mirror","door")]
rm=lambda r:"r%03d"%r
col=lambda r,i:COLORS[(r+i)%6]
mat=lambda r,i:MATERIALS[(r+2*i)%6]
idx=lambda o:OBJECTS.index(o)
def decoy(r,obj,base):
    c=[o for o in OBJECTS if o!=obj and o!=base]; return c[r%len(c)]
facts=[]
for r in range(ROOMS):
    for i,o in enumerate(OBJECTS):
        facts.append(("%s %s color?"%(rm(r),o),col(r,i)))
        facts.append(("%s %s material?"%(rm(r),o),mat(r,i)))
    for obj,base in RESTS_ON:
        facts.append(("%s %s on?"%(rm(r),obj),base))
        facts.append(("%s %s near?"%(rm(r),obj),decoy(r,obj,base)))
    facts.append(("%s next?"%rm(r),rm((r+1)%ROOMS)))
    if r%4==0: facts.append(("%s beside?"%rm(r),rm((r+1)%ROOMS)))

# which runs recur at least twice across the corpus (emergence_threshold = 2)
from collections import Counter
cnt=Counter()
for q,_ in facts:
    b=q.encode()
    for L in range(2,M+1):
        for i in range(len(b)-L+1): cnt[b[i:i+L]]+=1
concepts={k for k,v in cnt.items() if v>=2}
def runset(s):
    b=s.encode(); out=set(b[i:i+1] for i in range(len(b)))
    for L in range(2,M+1):
        for i in range(len(b)-L+1):
            r=b[i:i+L]
            if r in concepts: out.add(r)
    return out
def byteset(s): return set(s.encode()[i:i+1] for i in range(len(s)))
def score(q,t,f):
    a,b=f(q),f(t); n=len(a&b)
    return (n/len(b))*(n/len(a))
print("trained %d ; recurring runs (concepts) %d"%(len(facts),len(concepts)))
tq=[q for q,_ in facts]
for model,f in (("byte-set",byteset),("run-set",runset)):
    amb=0; ceil_untaught=0; tot=0
    for r in range(ROOMS):
        for obj,base in RESTS_ON:
            q="%s %s on material?"%(rm(r),obj); tot+=1
            good="%s %s material?"%(rm(r),base)
            # every rewrite the splice scan can produce, as in production
            k=len("%s %s on?"%(rm(r),obj))
            rws=[]
            for j in range(0,k+1):
                rw=q[:j]+base+q[k:]
                if rw!=q: rws.append(rw)
            ceil=[rw for rw in rws if max(score(rw,t,f) for t in tq)>=1.0]
            if len(ceil)>1: amb+=1
            ceil_untaught+=sum(1 for rw in ceil if rw not in tq)
    print("%-9s probes %d ; probes with >1 ceiling rewrite %d ; UNTAUGHT rewrites at ceiling %d"
          %(model,tot,amb,ceil_untaught))
# the single pair the whole ambiguity rests on
for a,b in [("r000 desk material?","r0desk material?")]:
    for model,f in (("byte-set",byteset),("run-set",runset)):
        print("%-9s  '%s' vs '%s': sets equal? %s ; score of the untaught one against the taught one %.4f"
              %(model,a,b,f(a)==f(b),score(b,a,f)))

# ---- the beside_next half: why a 1-hop family is 0% at every scale ----------
print()
for r in range(ROOMS):
    if r % 4 == 0:
        continue
    q = "%s beside?" % rm(r)
    want = rm((r + 1) % ROOMS)
    ranked = sorted(((score(q, t, byteset), t, a) for t, a in facts), reverse=True)
    top = ranked[0]
    ties = [x for x in ranked if abs(x[0] - top[0]) < 1e-9]
    own = "%s next?" % rm(r)
    own_rank = 1 + [i for i, x in enumerate(ranked) if x[1] == own][0]
    own_score = [x[0] for x in ranked if x[1] == own][0]
    print("%-13s want=%s  TOP %.4f %-22s -> %-6s ties=%d | its own 'next?' %.4f rank=%d of %d"
          % (q, want, top[0], top[1], top[2], len(ties), own_score, own_rank, len(facts)))
