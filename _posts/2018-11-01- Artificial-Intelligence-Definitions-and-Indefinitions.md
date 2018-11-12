---
layout: post
title: "On the buzziest of buzzwords: what is AI, anyway?"
toc: true
image: /images/unsplash-grey-flowerbuds.jpg
categories:
  - AI for MDs
tags:
  - artificial intelligence
  - machine learning
  - deep learning
  - big data
  - definitions
---

# Introduction

![cheezy AI stock image for visual interest. Brain on one side, circuit on the other](https://cdn.pixabay.com/photo/2017/04/13/20/26/artificial-intelligence-2228610_1280.jpg)

Artificial Intelligence is [kind of a big deal](https://trends.google.com/trends/explore?date=all&q=%2Fm%2F0mkz,potato) [^2]. Despite real advances, particularly in medicine, for most clinicians "AI" is at best a shadowy figure, a vaguely defined ethereal mass of bits and bytes that lives in Silicon Valley basements and NYT headlines.

"Bob, the AI is getting hungry."

"I don't know, Jane, just throw some AI on it."

"AI AI AI, I think I'm getting a headache".

This article is the first of a series meant to demystify AI, aimed at MDs and other clinicians but without too much medical jargon. We begin with definitions (and indefinitions) of a few of the most popular terms in the lay and technical presses.

# Artificial Intelligence

There is no generally accepted definition of AI.

This begins a theme that will run throughout this article, best illustrated with an analogy from Humanities 101.

If you pick up a stack of Western humanities textbooks with chronologies of the arts from prehistory to the present, you will likely find a [fairly unified canon](https://en.wikipedia.org/wiki/Western_canon) (try to find a textbook that does _not_ include Michelangelo and Bach, even if it is brand new and socially aware) up until the 1970s or 1980s . At this point, scholarly consensus ceases. It has not had time to mature. As the present day approaches, the selection of important pieces and figures, and even the acknowledgement and naming of new artistic movements (e.g. "[post-postmodernism](https://en.wikipedia.org/wiki/Post-postmodernism)"), becomes idiosyncratic to the specific set of textbook writers.

While the phrase "AI" has been around since [1955](https://aaai.org/ojs/index.php/aimagazine/article/view/1904), the recent explosion in tools, techniques, and applications has destabilized the term. Everyone uses it in a slightly different way, and opinions vary as to what "counts" as AI. This reality requires a certain mental flexibility, and an acknowledgement that any definition of AI (or any of the other terms discussed below) will be incomplete, biased, and likely to change.

With that in mind, we offer three definitions:

1. General AI: This is the kind of AI that can reason about any kind of problem, without the requirement for explicit programming. In other words, general AI can think flexibly and creatively, much in the same way humans can. _General AI has not yet been achieved_. [Predictions](https://hackernoon.com/frontier-ai-how-far-are-we-from-artificial-general-intelligence-really-5b13b1ebcd4e) about when it will be achieved range from the next few decades, to the next few centuries, to never. [Perceptions](https://www.newyorker.com/magazine/2018/05/14/how-frightened-should-we-be-of-ai) of what will happen if it is achieved range from salvific to apocalyptic.

2. Narrow AI: This is the kind of AI that can perform well if the problem is well-defined, but isn't good for much else. Most AI breakthroughs in recent years are "narrow," algorithms that can meet or exceed human performance on a specific task. Instead of "thinking, general-purpose wonder-boxes," current AI successes are more akin to "[highly specialised toasters](https://aeon.co/ideas/the-ai-revolution-will-be-led-by-toasters-not-droids)." Because most AI is narrow, and quite so, when clinicians see any article or headline claiming that "AI beats doctors," they would be wise to ask questions proposed by radiologist and AI researcher, [Dr. Oakden-Rayner](https://lukeoakdenrayner.wordpress.com/2016/11/27/do-computers-already-outperform-doctors/): "What, _exactly_, did the algorithm do, and is that a thing that doctors actually do (or even want to do)?" A more comprehensive rubric for evaluating narrow AI and planning projects is available in the appendix to Brynjolfsson and Mitchell's [practical guide to AI](http://science.sciencemag.org/content/358/6370/1530.full) from _Science_ magazine.

3. AI: We will finish with Kevin Kelly's [flexible and aware definition](https://ideas.ted.com/why-we-need-to-create-ais-that-think-in-ways-that-we-cant-even-imagine/):
>In the past, we would have said only a superintelligent AI could beat a human at Jeopardy! or recognize a billion faces. But once our computers did each of those things, we considered that achievement obviously mechanical and hardly worth the label of true intelligence. We label it "machine learning." Every achievement in AI redefines that success as "not AI."

    This view of AI takes into account the continual progression of the field, in sync with the progression of humans that produce and use the technology.

    Kelly's quote calls to mind George R.R. Martin's [humerously sobering line](http://www.georgerrmartin.com/about-george/on-writing-essays/on-fantasy-by-george-r-r-martin/), "Fantasy flies on the wings of Icarus, reality on Southwest airlines." If we flip this quote on its head a bit, we can see that real-life flight is a comprehensible thing, intellectually accessible to any person willing to put in the time with a little physics and engineering, to the point that it becomes banal. Even for those who know nothing of the math and science, most are unmoveably bored during their typical commuter flight, some fast asleep even before the roar of the tarmac gives way to the smooth and steady stream at 30,000 feet. AI is now, for many, more akin to Icarus than the 5:15 to Atlanta.

    In this series we hope to gain the middle ground: help the reader maintain a sense of possibility and perspective, but also understand the mundane ins-and-outs of day-to-day AI.

# Machine Learning

A common definition of ML goes something like,
>Given examples and some set of inputs, an algorithm predicts the output without being told explicitly how to get there. It 'learns' the relationship between inputs and outputs.

This is reasonable, but incomplete. Each algorithm has its own flavor: assumptions, strengths, weaknesses, uses, and adherents.

The simplest example generalizes well to more complex algorithms:

Imagine an AI agent that is shown input and output values of multiple cases. If it assumes a linear relationship between input and output, which is often a reasonable approach, it can then calculate (“learn”) a line that approximates the trend. After that, all you have to do is give the AI an input, even one it hasn't seen before, and it will tell you the most likely output. This describes the basic ML algorithm known as [Linear Regression](https://www.youtube.com/watch?v=zPG4NjIkCjc).

![illustration of linear regression](https://thumbs.gfycat.com/JointPartialIslandcanary-size_restricted.gif)

While linear regression is powerful and should not be underestimated, it depends on the core assumption we outlined, that is, the data are arranged in something approaching a straight line. From linear and logistic regression through high-end algorithms such as gradient-boosting machines ([GBM](https://www.youtube.com/watch?v=OaTO8_KNcuo)s) and Deep Neural Networks (DNNs), each machine learning algorithm has certain assumptions. A major advantage of many newer algorithms is that their assumptions are far more flexible than the classic regression functions available on high school graphing calculators, but at their core they are still abstracted approximations of the real world, equations defined by humans.

Even "unsupervised" machine learning, wherein the algorithm seeks to find relationships between groups of data points rather than being told exactly what these relationships should be, is based on iterations of simple rules. Below is an example of an unsupervised learning algorithm called [DBSCAN](https://dashee87.github.io/data%20science/general/Clustering-with-Scikit-with-GIFs/). It randomly selects data points, applies a simple rule to see what other points are "close enough," and repeats this until a predefined goal has been met.

![unsupervised learning DBSCAN gif](https://dashee87.github.io/images/DBSCAN_tutorial.gif)

Lastly, as always, regardless of the elegance of the algorithm, the machine can only take the data we provide. Junk in still equals junk out, even if it goes through an ultraintelligent washing machine.


## Deep Learning

Deep Learning (DL) is a subset of machine learning, best known for its use in computer vision and language processing. Most DL techniques use the analogy of the human brain. A "neural network" connects discrete "neurons," individual algorithms that each process a simple bit of information and decide whether it is worth passing to the next neuron. Over time the accumulation of simple decisions yields the ability to process huge amounts of complex data.

![Deep learning free stock image from Pixabay](https://cdn.pixabay.com/photo/2018/08/28/13/20/neural-network-3637503_1280.png)

For example, the neural network may be able to tell you [whether or not something is a hotdog](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3), what you probably meant [when you asked Alexa to "play Prince,"](https://developer.amazon.com/blogs/alexa/post/4e6db03f-6048-4b62-ba4b-6544da9ac440/the-scalable-neural-architecture-behind-alexa-s-ability-to-arbitrate-skills) or whether the retina shows signs of [diabetic retinopathy](https://doi.org/10.1001/jama.2016.17216). These successes in previously intractable problems led researchers and pundits to claim that DL was the breakthrough that would lead to general AI, but, in line with Kevin Kelly's fluid definition cited above, experience has now tempered these claims with [specific concerns and shortcomings](https://arxiv.org/abs/1801.00631).


# Big data

![Most of the time, "big data" is an overkill term. Sometimes, it isn't. Tricycle chained to light pole.](https://cdn.pixabay.com/photo/2014/05/03/01/02/three-wheeler-336700_1280.jpg)

The best definition of "big data" borders on the tautological:
>Data are "big" when they require specialized software to process.

In other words, if you can deal with it easily in Microsoft Excel, your database probably is not big enough to qualify. If you need something fancy like Hadoop or NoSQL, you are probably dealing with big data. Put simply, these applications excel at breaking massive datasets into smaller chunks that are analyzed across many machines and/or in a step-wise fashion, with the results stitched together along the way or at the end.

There is no hard-and-fast cutoff, no magic number of rows on a spreadsheet or bytes in a file, and no single "big data algorithm." In general, the size of big data is increasing rapidly, especially with such tools as always-on fitness trackers that include a growing number of sensors and can yield troves of data, per person, per day. The major task is to separate the wheat from the chaff, the signal from the noise, and find meaningful, actionable trends. The larger the data, the more the potential: both for finding something novel and meaningful, and for drowning in so many meaningless bits and bobs. 

# Summary

AI and related terms have no completely satisfying or accepted definitions. They are relatively new and constantly evolving. Flexibility is required[^1]. Behind all of the technological terms, there are humans with mathematics and computers, creativity and bias, just as there is a human inside the white coat next to the EKG machine.

As an important aside and ending note: if you were smart enough to learn to read those EKGs, you are smart enough to learn the basics of artificial intelligence and apply them in your own work.

[^1]: A linguistic gem from an early AI researcher is here apropos: "Time flies like an arrow. Fruit flies like a banana." The source of this Oettinger quote is frustratingly hard to find, as is often the case with classic papers from the middle of the 1900s. [Here's a PDF](http://worrydream.com/refs/Scientific%20American,%20September,%201966.pdf). The article starts on p. 166.

[^2]: Lest you think I've lost perspective on what really matters, here's a [comparison of the Google search trends over time for "Artificial Intelligence" and "potato."](https://trends.google.com/trends/explore?date=all&q=%2Fm%2F0mkz,potato) Happy Thanksgiving.