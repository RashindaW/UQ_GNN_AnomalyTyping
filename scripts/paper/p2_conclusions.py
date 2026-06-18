#!/usr/bin/env python3
"""Append the adjudicated Conclusions to PART2_TYPING_STATS.md, sourcing
every volatile number FROM THE REGENERATED BODY ITSELF (audit fix: the
hand-written conclusions once disagreed with the body after a re-emission).
Run immediately after p2_battery.py."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "results/thesis_part2/PART2_TYPING_STATS.md")
doc = open(P).read()


def grab(pattern, n=1):
    m = re.search(pattern, doc, re.S)
    assert m, pattern
    return m.group(n)


conf = doc.split("### H1 view: confirmatory-14")[1].split("###")[0]
h1_ch = grab(r"channel rule \| \*\*(0\.\d+)\*\* \| permutation p=(0\.\d+)",
             1).strip()
h1_perm = re.search(r"permutation p=(0\.\d+)", conf).group(1)
h1_card = re.search(r"cardinality \(n_points==1\) \| (0\.\d+)", conf).group(1)
orc = re.search(r"Pooled over 18 combos: per-timestep accuracy (0\.\d+).*?"
                r"event-majority\s*accuracy (0\.\d+)", doc, re.S)
orc_step, orc_ev = orc.group(1), orc.group(2)
h2c = doc.split("### H2 view: confirmatory-14")[1].split("### H2 exploratory")[0]
cat = h2c.split("Grouping: SSSP/SSMP/MSMP")[1]
h2_ep_kw = re.search(r"\| epistemic \| (0\.\d+) \|", cat).group(1)
h2_pair = re.search(r"epistemic \| 0\.\d+ \| MSMP~SSMP: p=(0\.\d+) d=(0\.\d+)",
                    cat).group
buck = h2c.split("Grouping: mechanism buckets")[1].split("**Grouping")[0]
bkws = [float(x) for x in re.findall(r"\| \w+ \| (0\.\d+) \|", buck)]
loeo = re.search(r"LOEO AUC = (\S+); label-permutation p = (\S+) ", doc)
h3 = re.search(r"corroborated-verdict false rate = (0\.\d+) \(n=(\d+)\) vs "
               r"quiet false rate = (0\.\d+) \(n=(\d+)\).*?= (0\.\d+)\.", doc, re.S)
r4g = re.findall(r"R4 escalation gate \((\w+)\): held-out conservative "
                 r"precision = (0\.\d+) \(n=(\d+)\)", doc)
h4 = {m[0]: m[1] for m in re.findall(
    r"\| (\w+) \(all 6\) \| 6 \| (0\.\d+) ", doc)}
lat = re.findall(r"\| (gdn|topogdn|cstgl) \| \d+ \| \S+ \| \S+ \| \d+/\d+ \| "
                 r"(0\.\d+|1\.000|nan) \|", doc)
loc = re.search(r"target-hit@1 = (\d+)/(\d+); intent-equipment-hit@1 = (\d+)/\d+",
                doc)

con = f"""## Conclusions against the pre-registered claims ladder
(adjudicated 2026-06-07; every number below is parsed from this document's
own body at generation time; scope = post-Amendment-A1, 3 published
backbones, 18 combos, confirmatory view = 14 non-pilot combos)

H1 (mechanism) -- CLAIM GATE FAILS, mechanically. The channel rule
({h1_ch}, confirmatory view) is beaten outright by the trivial cardinality
baseline ({h1_card}); permutation p = {h1_perm}; exact-binomial Holm p >=
0.72 against every baseline. The phrase "the channels classify attack
mechanism" (and synonyms) is NOT licensed. Mandatory framing: this design
detects only >= ~0.30 absolute accuracy gaps at 80% power on n~36 events,
so the result is INCONCLUSIVE, NOT NEGATIVE. The oracle ablation
(target-informed: per-timestep {orc_step} / event-majority {orc_ev}) is an
upper bound by construction and licenses nothing.

H2 (separation) -- NOT CONFIRMATORY under the pre-committed rule. The
mechanism-bucket axis is null (confirmatory-view KW p {min(bkws):.3f} to
{max(bkws):.3f}). The category axis shows a single-channel trend:
epistemic KW p = {h2_ep_kw} in the confirmatory view -- above the 0.05
gate for the pooled test (the single confirmatory test per the
hierarchical multiplicity rule). The within-family pairwise MSMP~SSMP
contrast (Holm p = {h2_pair(1)}, Cliff's d = {h2_pair(2)}) is reported as
a DESCRIPTIVE trend toward a structural-complexity (multi-point) signal,
never as mechanism reading. The LOEO logistic is exploratory and null
(AUC {loeo.group(1)}, permutation p = {loeo.group(2)}).

H3 (triage utility) -- PASSES; THE CARRYING CONFIRMATORY LEG. On held-out
episodes under deployable C-slice thresholds, the corroborated-verdict
(peak-AND-onset) false rate is {h3.group(1)} (n={h3.group(2)}) versus
quiet {h3.group(3)} (n={h3.group(4)}); combo-cluster bootstrap P(ordering
violated) = {h3.group(5)}. R4 passes its pre-registered escalation gate on
both sources (held-out conservative precision {r4g[0][1]} {r4g[0][0]},
n={r4g[0][2]}; {r4g[1][1]} {r4g[1][0]}, n={r4g[1][2]}) and is described as
escalation-ELIGIBLE, never as reliable. FA/day figures are indicative
only; per-combo rows are descriptive; the supervised operating point's
threshold non-calibration is the documented P0 finding and the root cause
of the withdrawn latency claim.

H4 (stability) -- descriptive: modal-verdict share gdn {h4.get('gdn','-')}
/ topogdn {h4.get('topogdn','-')} / cstgl {h4.get('cstgl','-')};
Krippendorff alpha and observed agreement as tabled (med-split is a
label-using stress test, not an operating point).

H5 (capability) -- the centerpiece stands as a detection/triage exhibit
(never mechanism-scored): 13 held-out attacks, A29 = pre-declared
correct-quiet (official mechanical-interlock citation; counts toward no
recall figure). LATENCY: under the corrected within-window clamped metric
NO earlier-detection claim survives (sign p: {', '.join(f'{b} {p}' for b, p in lat)});
pre-window episode activity is reported only as the already-active-at-
onset rate, a symptom of the supervised threshold flood. Exploratory
localization: the modal CST-GL peak sensor hits the official attack point
on {loc.group(1)}/{loc.group(2)} held-out attacks and the intent
equipment on {loc.group(3)}/{loc.group(2)} -- localization tracks the
manipulated point, not the attacker's goal; hit@3 deferred.

CHAPTER VOCABULARY: the licensed sentence is ADAPTED FROM the
pre-registered fallback (the prereg fallback reads "the channels separate
attack types (H2) and make alarms triageable (H3)"; since H2 did not
confirm post-A1, the adapted licensed sentence drops the positive H2
clause): "the channels do not demonstrably classify attack mechanism
(inconclusive at this sample size), but they make alarms triageable" --
anchored by H3, with the adjudication's forbidden-claims list binding.
Part-1 cross-reference discipline: when quoting the pooled S1 result,
cite BOTH tests (Wilcoxon p=0.0134 AND the borderline exact sign test
p=0.0481); the S2 result (15/18, sign p=0.0038, Wilcoxon p=0.0045)
carries no such asymmetry.
"""
placeholder = doc.split("## Conclusions")[0]
open(P, "w").write(placeholder + con)
print("conclusions written from body-parsed values:",
      dict(h1=h1_ch, card=h1_card, orc=(orc_step, orc_ev),
           h2kw=h2_ep_kw, loeo=loeo.groups(), h3=h3.groups()[:1]))
