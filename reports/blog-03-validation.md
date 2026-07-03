# I Got 100% Accuracy. Then I Tried to Break It.

The first time I trained this model, one epoch in, before any real optimisation had happened, held-out
validation retrieval was already sitting at 87%. Three cancer lineages, a model that had barely started
learning, and it was almost getting the answer right on the first try.

That should make anyone suspicious. A result that comes this easily either means the model found real
biology fast, or it means the task was never hard to begin with. Those two explanations look identical
from the outside: both produce a clean number and a nice UMAP plot. The only way to tell them apart is
to go looking for reasons the result should fail, and see if it does.

So that is what the last twelve days were: an attempt to break my own 100% test kNN@5 (cell lines
correctly finding same-lineage patients as their nearest neighbours in embedding space) using every
adversarial check I could think of. Here is what survived, what didn't, and what I still don't know.

## First: is it a lucky test set?

The original test set is 38 cell lines. That is small enough that one favourable split could produce
100% by chance. So I re-ran the entire pipeline, split through training through evaluation, on ten
independent random seeds, each drawing a different train/val/test partition.

Result: mean kNN@5 across the ten splits is 95.0% ± 3.4%, with a 95% confidence interval of
[93.2%, 97.1%]. The worst single seed still hit 89.5%. The 100% number was a little optimistic, but the
underlying result is not a fluke of one split. It holds up.

## Second: is it a real baseline, or a strawman?

A held-out kNN retrieval number means nothing without something to compare it to. I ran the real batch
correction methods people actually use for this kind of problem. Harmony, run on the identical test
embeddings, gets 84.2%. That is a strong number, and my contrastive model beats it by 15.8 points.
Good.

Then I ran the check that actually worried me: what if you skip all the alignment machinery entirely?
I trained a plain logistic regression classifier on CCLE expression alone, no cross-domain training, no
shared embedding space, nothing. Just: learn lineage from cell lines, then predict lineage on patient
expression directly.

That classifier scores 97.1%.

My contrastive model, with its two-tower architecture and its InfoNCE loss and its learned temperature,
beats that plain classifier by 2.9 points. Three cancer lineages, it turns out, are close to linearly
separable in bulk RNA-seq without any domain adaptation at all. The alignment step is real and it does
help, but the honest reading is that lineage identity was never the hard part of this problem. This is
the least comfortable number in the whole project, and it is the one I trust the most, because I went
looking for a reason my result might be hollow and found one.

## Third: make the task actually hard

If three lineages are nearly trivial, the fix is to stop using three lineages. I scaled the task to 15,
including pairs that are genuinely easy to confuse biologically: LUAD and LUSC (both lung), COAD and
READ (both colorectal), GBM and LGG (both glioma).

kNN@5 dropped to 78.4% (95% CI [69.8%, 85.0%], n=111). That drop is the point, not a failure. What
matters is where the errors land. Named biologically-related pairs absorbed 45.8% of all the model's
mistakes, despite being only 3.8% of the possible ways to be wrong, a 12x concentration. The two hardest
lineages, LGG and READ, sent 100% of their misclassifications to their real biological partners, GBM
and COAD. When this model is wrong, it is wrong in a way that tracks actual tumour biology. That is
much better evidence of learned structure than a clean number on an easy task.

## Fourth: is it secretly measuring tumour purity?

Cell lines are grown in pure culture. Patient tumours are contaminated with stroma and immune cells. A
model could, in principle, learn "pure sample versus impure sample" and call it lineage alignment. I
checked: the correlation between each sample's position on the domain axis and its tumour purity score
is r = -0.455, moderate but not dominant. More importantly, kNN@5 holds at 100% in both the high-purity
and low-purity halves of the patient test set, and lineage clustering survives purity correction
(silhouette +0.566 down to a still-strong +0.500). It is not a purity detector wearing a lineage
costume.

## Fifth: does it collapse when it should?

The strongest negative control is the simplest one. Shuffle which cell-line lineage corresponds to
which patient lineage, breaking the real correspondence the loss depends on, and see if the model can
still succeed. Across 100 shuffled permutations, the null result hovers at 7.0-7.7% (chance for 15
lineages is 6.7%), and the single best shuffled run across all 100 only reached 17-20%. The real 78.4%
is nowhere near that range. Empirical p = 0.0099. When there is no real signal to find, the model
correctly fails to find one.

## Sixth: the comparison I couldn't finish

The obvious next step was a head-to-head against Celligner, the published method other researchers
already use for this exact cell-line-to-tumour alignment problem. I tried. The published Python package
depends on a library that does not exist on PyPI under the name it asks for, and the workaround requires
building from source with R installed, which this environment does not have. I could not get a number.

I am reporting that as a gap, not quietly dropping the comparison. What I can say is that Celligner is
unsupervised. It has to discover lineage structure on its own. My model is handed lineage labels during
training. If Celligner reaches comparable accuracy without ever being told the answer, that is not a
threat to anything above, it actually reinforces the point that coarse lineage is an easy signal to
find.

## Seventh: does any of this touch drug response?

Everything so far is about lineage. Lineage is not why anyone cares about cell line models; drug
response is. So I picked the cleanest available case: vemurafenib, a BRAF-targeted melanoma drug with a
textbook mechanism, and asked whether my embedding says anything about it.

Part one worked, weakly. BRAF-mutant melanoma cell lines sit closer to the centroid of BRAF-mutant
melanoma patients than BRAF-wild-type lines do (p = 0.047, effect size 0.649). The model has found
something below the level of "melanoma" and above the level of noise.

Part two did not work. Proximity to that BRAF-mutant patient centroid does not predict vemurafenib
sensitivity (Spearman rho = 0.209, p = 0.19, n = 41). I checked whether this was the embedding
specifically destroying drug-response signal by comparing it against raw gene expression on the same
prediction task, cross-validated within the cell lines. The raw expression did not do any better
(negative R² for both). At n=41, a single categorical fact, BRAF status alone, beat both continuous
representations. That is not evidence the alignment threw signal away; it is evidence that 41 cell
lines is not enough to resolve a continuous drug-response phenotype from either representation. Honest
answer: inconclusive, not negative.

## What this adds up to

A model that reliably finds cancer lineage, mostly for reasons that turn out to be nearly trivial, that
survives every adversarial check I could throw at it, and that does not yet reach the drug-response
question that would make it clinically interesting. None of this is a clinical tool. It has not been
tested outside TCGA, on other platforms, on patient-derived models, or in anything resembling a
prospective trial, and none of that is in scope for what one person can responsibly claim from a public
dataset and a laptop. What it is: a released, reproducible harness for stress-testing this exact class
of claim, built by trying to prove myself wrong before anyone else could.

Full evidence table, code, and every daily report: https://github.com/vthawfeek/pre-clinical-to-clinical-translation
