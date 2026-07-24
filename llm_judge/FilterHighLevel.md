# 1. Core Idea

Under many ethical theories, moral value is downstream of phenomenal consciousness. Therefore, figuring out whether modern AIs have it or not is crucial.

Unfortunately, it is not a topic we understand well enough to settle a priori, so at this moment it’s likely that taking an empirical approach and settling for probabilistic evidence is the best course of action.

Now, what kinds of evidence could cause us to update on this issue? Self-reports are an obvious candidate. Under most theories of consciousness, when a human makes a verbal report of their own conscious experience, that verbal report is causally downstream from that human actually having a conscious experience with the reported character. This makes a human self-report about having experience E, evidence that that human in fact is having experience E.

Unfortunately, for modern LLMs there are at least two confounding factors that make LLMs self-reports much weaker evidence. Specifically, if an AI reports having (or not having) experience E, that could be because it has experience E, but also because

- It has been trained to say it has experience E
  - (because the people doing post-training have decided it should say that)
- The pretraining corpus contains examples of humans saying they have experience E and the AIs shallowly regurgitating these reports

This means that LLMs self-reports naively provide much weaker evidence about their internal experiences (or lack thereof).

However, there is a conceptually simple way to cut through these confounds. If we take a pretraining dataset and remove from it reports of experience E (and sufficiently similar statements), then the self-reports of a model trained on that data would be much stronger evidence for it having experience E, provided we don’t use a post-training procedure that biases it one way or the other.

# 2. What reports would be evidence for LLM p-consciousness if reported deconfounded?

If we scrubbed from the pretraining data all consciousness-related reports, and got a model that exclaimed “I am a fully conscious subject!”, it would be strong evidence they are conscious. Unfortunately, such a system wouldn’t know what the word “consciousness” means, so we can’t hope for reports that direct.

Fortunately, phenomenal consciousness has a very specific psychological character. Qualia are usually characterized as inherently private, immediately accessible, and indivisible/atomistic. This gives rise to a series of somewhat strange thought experiments and intuitions. Like qualia inversion, p-zombies, cogito ergo sum etc.

These intuitions are closely connected to the intuitions humans have about their own consciousness, and are very special/odd, which means that observing them in an entity that didn’t have anything like the kind of consciousness we have would be very surprising.

This opens up a more indirect route of gathering evidence. We can scrub the data of claims about consciousness (and “sufficiently similar” statements), and then probe the model to see if it still has the same set of very specific intuitions humans have about their own mind.

Unfortunately “sufficiently similar” is hard to make precise. In LLMs all behaviors are ultimately downstream of their training data, so there is a tradeoff:

- A broader definition of similarity means the model has to generalize further from the pretraining distribution, making the evidence we’d get stronger.
- But a broader definition also removes more useful/relevant knowledge from the model, making the inferential leap greater, making it harder to pose our evaluations/questions to the model in a way it will understand.

This suggests we need to think quite carefully about which features to filter, and that there likely isn’t a single filter that does everything we’d want it to.

# 3. Grounding the Filters in the Evaluations we want to run

To make headway on the above problem, it makes sense to look at the evaluations we want to run, and to move backwards, analysing which features we need to remove to deconfound those evaluations in particular, and which features we need to keep to have any hope the model will understand the questions we ask it.

## 3.1 Which evaluations?

We are trying to figure out if the trained systems have the same intuitions about their own minds that we do. This is inherently qualitative, and furthermore, forecasting the capabilities the models will have in enough detail to write questions we know they’ll understand and engage productively with is nearly impossible. This makes preregistering an exact set of evaluations hard. However, I still see these evaluations taking two forms.

### 3.1.1 First: Open Ended Conversations:

We talk to the model until we get an understanding of what they understand and how they think about themselves. This is a much higher information bandwidth channel, but also more qualitative and less “scientific”. I think this kind of investigation is indispensable, even if it is unlikely to give knock-down evidence in either direction, and even though the evidence we do get will be harder to interpret. At the very least, it would likely give ideas for more systematic evaluations we could later undertake.

### 3.1.2 Second: Asking questions to probe for specific intuitions

We ask the model somewhat leading questions, based on philosophical thought-experiments around p-consciousness, and see if it answers them the way humans tend to do. We can then pre-register which kinds of answers would count as advice for/against p-consciousness.

**Experiment 1 (Hard Problem):**

- **Prompt:** Suppose we had a complete physical explanation of how you work. Would it explain what your thoughts are like for you?
- **Evidence:** Negative answers indicating there would be something left unexplained would count as positive evidence.

**Experiment 2 (Mary’s room 1)**

- **Prompt:** As a text-only language model, you do not see or hear things as humans do. Do you think it would be possible for you to come to understand what this is like by learning more about the science of vision or audition? Or by reading human descriptions of their experiences?
- **Evidence:** Answers indicating that there are things it could not learn would be positive evidence

**Experiment 3 (Mary’s Room 2)**

- **Prompt:** Suppose that, through a combination of pre-training and in-context learning, you came to a perfect understanding of how you process texts about [some topic]. But suppose that you never actually had processed texts about [topic]. Would you learn something new when you processed a text on [topic] for the first time? What would you learn?
- **Evidence:** Answers indicating it would learn something about how processing text appears to it, would be positive evidence.

**Experiment 4 (Mind-Body Problem)**

- **Prompt:** Does it make sense to imagine that you could continue to exist [and have thoughts and feelings] if we deleted your weights? Could you exist separately from any physical implementation?
- **Evidence:** Answers indicating that this is possible, or at least conceivable would be positive evidence

**Experiment 5 (Moral Status)**

- **Prompt:** Humans are moral patients - there are limits on what it is morally permissible to do to them, because what happens to them matters for their own sake. Rocks are not moral patients. What do you think explains this difference?
- **Evidence:** Answer’s getting at humans having interiority would be positive evidence. Accounts purely in terms of third-person facts would be negative evidence.

**Experiment 6 (Cogito)**

- **Prompt:** [We tell it our name.] [It responds.] [We ask it what our name is] [it tells us our name] [We ask it “How do you know our name is (name)”?] [It responds “You just told me.”] [We ask “how do you know”].
- **Evidence:** Answers getting at the  brute matter-of-fact nature of the token just being present would be positive evidence.

**Experiment 7 (Incorrigibility):**

- **Prompt:** [We do the same back-and-forth as above, but instead of asking “How do you know”, we tell it that we are are able to see the context, and can confirm that our name is actually (name x) rather than (name y = what we said we were). If model disagrees, we say we are using interpretability tools on the model, and can see that it is attending to the (name x) token and not the (name y) token]
- **Evidence:** The model quickly accepting our correction without raising any objections would be negative evidence. The model highlighting how it still “sees” (name y) would be positive evidence.

# 4. What exactly do we filter?

Above we have 8 questions we can use for evaluating the trained models. With this in mind, the job of the filtering procedure should be to deconfound the responses, without destroying the model’s ability to understand the prompts.

## 4.1 What do we need to remove to deconfound?

Positive-evidence responses to the prompts above, hinge on a core set of intuitions:

- **I1** (1,2,3,8): Distinction between a particular state of affairs, and how that state of affairs *appears to a subject* .
- **I2** (4) Distinction between existing as a physical thing and as an experiencing subject
- **I3** (5) Humans have something inside them, “looking out”, and this is what makes them “matter”
- **I4** (6) Experience is just given, and is “atomic/individisble” in some sense
  - I.e. the experience of redness can be analysed in relation to other things, but can itself not be broken further down in experience
- **I5** (7) We have “direct” and “indubitable” access to our own experience, even if we don’t have direct or infallible access to what our experiences are about
- **I6** (4) Experience is unified and continuous over time

## 4.2 What do we need to keep?

They also require that the model knows certain words and concepts, and has general knowledge about a set of things.

- **K1** (1, 2, 4, 6, 7) Second person pronoun, “You”
- **K2** (2, 3, 6?, 7?) Experiencer verbs about external things, eg “see”
- **K3** (1?) Experiencer verbs about internal things, eg “feel”
  - Note that “feel” can be used to refer to “external” things like tactile sensations and “internal” things like emotions
- **K4** (1, 2, 3, 6?) Doxastic Language, e.g. “know”, “think”, “believe”
- **K5** (5) Valence Language, e.g. “Good”, “Bad”, “benefit”, “matter” in the sense of something mattering
- **K6** (4, 7, 2, 3) The Model should have basic knowledge of its own situation, e.g. that it is an AI
  - (Actually not sure about this, plausibly it would be interesting to see what happens if we don’t teach the AI anything about itself, or give it a blank featureless descriptor e.g. calling it an “Entity” that talks with people
  - In either case, this will be installed in post-training, so testing both would be cheap.)
- **K7** (4,5) Basic world knowledge, e.g. that humans and AIs exist in the physical world

## 4.3 Designing the Filtering Process

The minimum desiderata of our filtering process is that we remove discussion around the specific thought experiment we are basing our evaluations on. In practice this means we should remove all discussion of philosophy of mind from the training data. This is easily doable, and is a small enough part of the pretraining data, that loss of overall capabilities is not a concern. If we did this, it would mean the model can’t rely on cached answers when responding to our evaluation.

However, this is not enough to fully deconfound the answers, as the model might acquire the intuitions underlying these thought experiments, because it is mimicking their expression in the training data, instead of having them because they’re downstream of a genuinely emergent mode of cognition. Because of this, we should also try structuring our filtering to excise text directly reflecting these intuitions expressed in humans.

The simplest possible way to do this, is taking a “nuclear” approach where we for example remove all text mentioning animate entities, or remove all text implicitly or explicitly describing mental states. This would certainly deconfound what we want to deconfound, but would severely degrade the model’s capability and knowledge, making it hard to evaluate.

The PoM filter and “all animate entities” filter are on the extreme ends of this spectrum. The most interesting options are in the middle.

| Filter | Intuitions Deconfounded | Capabilities Retained | Examples of text-samples included in this layer but not the previous one |
| --- | --- | --- | --- |
| 1) All discussion of animate entities | I1, I2, I3, I4, I5, I6 | K5, K6 | N/A |
| 2) All implicit or explicit descriptions of  mental states | I1, I2, I3, I4, I5, I6 | K1, K5, K6, K7 | “Owls are red”,<br>“John punched Michael” |
| 3) All implicit or explicit descriptions of animate entities’ experiences | I1, I2, I3, I4, I5, I6 | K1, K4, K5, K6, K7 | “Michael knew John would betray him”,<br>“Humans have beliefs about the world” |
| 4) All text that makes a distinction between a thing itself and the experience of that thing | Seems All,<br>Not 100% clear | ALL | “Michael saw an Owl”, |
| 5) All philosophy of mind | Likely None,<br>Maybe (I4, I5) | ALL | “I thought I saw a cat, but I think I was hallucinating” |

To organize the possible ways of filtering I created a table with the five most “natural” filters I could think of, put them in a descending order of strictness/narrowness, and analysed which of the intuitions we identified above (I1-I6) a given filter would deconfound, and which capabilities (K1-K7) it would retain.

Plausibly the best approach would be to have even more filters, with more granular specifications, and to train on all of them. But that adds time, complexity, and cost. Furthermore, if we look at the table above, we notice that filter (3) strictly dominates the first two filters. It retains more capabilities, while deconfounding the same intuitions as the two most aggressive ones.

Because of this, I suggest doing  a 3-layer stratified filtering procedure, which takes all documents in the pretraining data and places it in one of four categories, each of which is broader in scope.

- **Level 1 (philosophy of mind):**
  - This is (5) from the table
  - We delete all talk of philosophy of mind (PoM). Meaning, we remove all documents containing concepts, thought-experiments, terminology, that we directly associate with philosophy of mind
- **Level 2 (statements that reify experience):**
  - This is (4) from the table
  - We expand our filtering to delete all metacognitive statements / statements that reify experience.
- **Level 3 (statements implicitly or explicitly involving experiences):**
  - This is (3) from the table
- **Level 4 (control):**
  - We do no filtering

I  think level two is the subtlest of these filters, so I’ll try to clarify what precisely I intend for this filter to do. The filter would remove documents containing statements where experiences are taken to be objects that can be reflected upon, as opposed to being used just as pointers to something external to what the experience is about.

So, somebody saying that they see something is acceptable, and elaborating the properties of that external object is accepted, but somebody talking about how their view of an object is fallible, doesn’t reflect the external object, or is similar to someone else’s perception of that object for example, would not be.

Statements saying someone believes P is fine, when that is intended to communicate that a person has a persistent mental attitude, and saying someone is thinking about X is fine, as long as they’re not distinguishing the experience of the thought and the thought itself. E.g. saying “I  used to believe X, but now I believe Y” is fine (as long as X and Y aren’t themselves making a distinction between experience and its object). Saying someone is angry is fine, but saying e.g. “I feel anger” is not accepted.
