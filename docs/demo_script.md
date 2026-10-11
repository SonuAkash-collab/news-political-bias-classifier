# Live demo script

For the review session. The demo itself takes under 3 minutes. Everything below was checked on
2026-10-11 against the models committed in `models/` (scikit-learn 1.6.1).

## Before the session

1. Activate the virtual environment in the repository root (`.venv\Scripts\Activate.ps1` on
   Windows, `source .venv/bin/activate` on Linux and macOS).
2. Make sure `data/processed/articles.csv` exists. Without it, "Pick a test article" shows a warning.
   If it is missing, run `python scripts/download_articles.py` (about 3 minutes).
3. Start the app and leave it open on the first screen:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   Open http://localhost:8501. The first prediction loads the model, which takes a few seconds,
   so do one practice click before the session starts.
4. Keep a second terminal open in the repository root with the virtual environment active, in case
   you need the CLI fallback.

## The six demo articles

All six come from the binary (left/right) AllSides test set, media-based split. "Position" is the
article's place in the app's test-article dropdown, counting from 0. Excerpts are the title plus the
first two sentences only, with the dataset's spacing around punctuation tidied.

Each model cell shows the prediction and P(Right) as the app shows it ("Right" in the result card).
The app's confidence is the decision score passed through a sigmoid, not a calibrated probability.

Model keys: **Ext 3** = LinearSVC trained on articles (app default); **Ext 1** = TF-IDF + logistic
regression; **SVC** = LinearSVC tweet-trained; **SVC bal** = LinearSVC balanced; **LR** = logistic
regression tweet-trained; **CNB** = Complement NB. Only Ext 3 is trained on articles; the other five
are trained on tweets.

### Summary table

| # | Article id | Position | Source | True | Ext 3 | Ext 1 | SVC | SVC bal | LR | CNB |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | TAVOm1O3NSti1DlF | 494 | The Guardian | Left | Left, 0.0% | Left, 4.4% | Left, 0.0% | Left, 0.0% | Left, 0.0% | Left, 0.0% |
| L2 | OxNF6FoHcp50lVsA | 432 | The Guardian | Left | Left, 0.8% | Left, 8.3% | Left, 39.0% | Left, 8.3% | Left, 0.3% | **Right, 100.0%** |
| L3 | BrysXPT7tmklcIOU | 224 | CNN (Web News) | Left | **Right, 94.5%** | Left, 13.0% | Left, 0.0% | Left, 0.0% | Left, 0.0% | Left, 0.0% |
| R1 | 03MzG5A1p9xyfcNh | 0 | Breitbart News | Right | Right, 99.3% | Right, 79.9% | Right, 100.0% | Right, 100.0% | Right, 100.0% | Right, 100.0% |
| R2 | uXNyDaauEcmCoARC | 915 | Reason | Right | Right, 99.0% | Right, 77.4% | Right, 100.0% | Right, 99.9% | Right, 100.0% | Right, 100.0% |
| R3 | If9eyXazobjslwX8 | 326 | Reason | Right | **Left, 0.0%** | **Left, 30.4%** | **Left, 0.0%** | **Left, 0.0%** | **Left, 0.0%** | **Left, 12.5%** |

Wrong predictions are in bold. Values such as 0.0% and 100.0% are rounded to one decimal as in the
app; they are not exactly 0 or 1.

### L1. One nation, two Trumps: America as divided as ever after first 100 days

- Id `TAVOm1O3NSti1DlF`, The Guardian, 2017-04-27, true label Left.
- Excerpt: "How did Donald Trump do in his first 100 days as US president? The moment you step out into
  the beating heart of America, you realize that's the wrong question."
- Every model says Left; Ext 3 gives Right 0.0%. Use it as the easy, high-confidence correct case.

### L2. Amazon to ban police use of facial recognition software for a year

- Id `OxNF6FoHcp50lVsA`, The Guardian, 2020-06-10, true label Left.
- Excerpt: "Amazon is implementing a one-year moratorium on police use of its artificial intelligence
  software Rekognition amid a growing backlash over the tech company's ties to law enforcement. The
  company has recently stated its support for the Black Lives Matter movement, which advocates for
  police reform, using Twitter to call for an end to "the inequitable and brutal treatment of black
  people" in the US and has putting a "Black lives matter" banner at the top of its home page."
- Ext 3 is correct (Right 0.8%). The models disagree: Complement NB says Right 100.0%. Good for
  showing that switching the model changes the answer.

### L3. White House faces off against Congress in Supreme Court

- Id `BrysXPT7tmklcIOU`, CNN (Web News), 2014-01-13, true label Left.
- Excerpt: "(CNN) - Supreme Court appeared on Monday to lean toward Congress in the fight over
  presidential recess appointments, those placed in top government jobs temporarily without Senate
  approval. It's a high-stakes constitutional and political confrontation over a practice that has
  accelerated in recent years due to partisan gridlock in Congress."
- Ext 3 is confidently wrong (Right 94.5%), while all five tweet-trained models say Left. A neutral,
  procedural news report has little party vocabulary, so the label really comes from the outlet.

### R1. Report: Nancy Reagan Dead at 94

- Id `03MzG5A1p9xyfcNh`, Breitbart News, 2016-03-06, true label Right.
- Excerpt: "Nancy Reagan has died at the age of 94, according to TMZ. Mrs. Ronald Reagan, as she
  wished to be known even when her husband died, was the 40th president's closest confidant and a
  played a major role in his career and presidency."
- Every model says Right; Ext 3 gives Right 99.3%. It is the first article in the dropdown, so it is
  already selected when you switch to "Pick a test article".

### R2. What Does It Mean To Be a Libertarian?

- Id `uXNyDaauEcmCoARC`, Reason, 2018-11-12, true label Right.
- Excerpt: "We libertarians have always preferred to use esoteric arguments, specialized language,
  and other secret handshakes usually invoking the furthest reaches of Austrian praxeology. Like
  exclusive membership in any tribe, this can all be great fun."
- Every model says Right; Ext 3 gives Right 99.0%. A backup for R1.

### R3. When Should Force Be Used To Protect Public Health?

- Id `If9eyXazobjslwX8`, Reason, 2020-06-07, true label Right.
- Excerpt: "During an April 2 interview with Chris Wallace on Fox News, Surgeon General Jerome Adams
  compared deaths caused by COVID-19 to deaths caused by smoking and drug abuse. "More people will
  die, even in the worst projections, from cigarette smoking in this country than are going to die
  from coronavirus this year," he said."
- Every model says Left; Ext 3 gives Right 0.0%. AllSides rates Reason (libertarian) as right, but
  this article's public-health vocabulary reads as left to every model. Good for the "where it fails"
  point.

## Step-by-step script (under 3 minutes)

Times are rough targets. Words in quotes are what to say; adapt them freely.

**0:00. Open (app already on screen, sidebar visible).**
"This is our demo. On the left you pick a model; the default is the LinearSVC we trained on news
articles in Extension 3, our best article model at 66.5% accuracy. You can paste any text, or pick
one of the 1,001 test articles the model never saw."

**0:20. Easy correct case (R1).**
Click "Pick a test article" in the sidebar. The first article, "[Right] Breitbart News — Report:
Nancy Reagan Dead at 94", is already selected. Click **Classify**.
"It says Right with 99.3% confidence, and the true label is Right. The chips underneath are the
words in this article that pushed the score towards Right most."

**0:45. Clear left article (L1).**
Click the article dropdown, type `One nation`, press Enter, click **Classify**.
"A Guardian piece on Trump's first 100 days: Left, almost 100%. Correct."

**1:05. Where it fails (L3).**
Click the dropdown, type `White House faces off`, press Enter, click **Classify**.
"This one is wrong. It's a CNN report on a Supreme Court case, labelled Left because AllSides rates
CNN as left, but the text is neutral and procedural. The model says Right at 94.5%. That's the core
difficulty: labels come from the outlet, not from each article."

**1:35. Switch models (same article).**
In the sidebar, change the model to "LinearSVC — tweet-trained (model zoo)". Click **Classify**
again.
"The tweet-trained SVM says Left here, so it's right this time. Overall, though, the tweet models
barely beat always guessing Right: 57.7 to 60.6% against 59.8%. Training on articles is what gave us
the +6.7 points."

**2:00. Free text.**
Switch the sidebar back to "LinearSVC — in-domain articles (Ext 3, best)", choose "Paste text" and paste:
`Republicans in Congress blocked the voting rights bill and refused to expand access to affordable health care for working families.`
Click **Classify**.
"This sentence sounds Democratic to a person, but the article model says Right at 56.3%, close to
a coin flip. It learned the style of whole news articles, not tweet-style slogans, and one sentence
gives it very little to work with. That's honest about what the model can and cannot do."

**2:30. Close.**
Open the "Accuracy reference" expander at the bottom.
"This table has every model's accuracy on the same 1,001 articles. Ext 3 is the only one clearly
above the 59.8% baseline. Thank you."

Optional if there is time or a question: L2 (models disagree: Complement NB says Right 100.0%)
and R3 (every model wrong on a libertarian public-health article).

## If the web app fails: CLI fallback

Run these from the repository root with the virtual environment active. The CLI uses the same
models and prints the prediction, Left and Right percentages, and the driving words.

```bash
python app/cli.py --model ext3 --text "Report: Nancy Reagan Dead at 94. Nancy Reagan has died at the age of 94, according to TMZ."
```

```bash
python app/cli.py --model ext3 --text "White House faces off against Congress in Supreme Court. Supreme Court appeared on Monday to lean toward Congress in the fight over presidential recess appointments."
```

```bash
python app/cli.py --model linear --text "White House faces off against Congress in Supreme Court. Supreme Court appeared on Monday to lean toward Congress in the fight over presidential recess appointments."
```

```bash
python app/cli.py --model ext3 --text "Republicans in Congress blocked the voting rights bill and refused to expand access to affordable health care for working families."
```

```bash
python app/cli.py
```

Expected results with the committed models: Right 65.1%; Right 65.9%; Left 53.8%; Right 56.3%.
The last command starts interactive mode: type or paste text, press Enter on an empty line to
classify, and type `quit` to exit. Model keys: `ext3`, `ext1`, `linear`, `logistic`, `cnb`
(`python app/cli.py --help` lists them).

The CLI percentages for these short excerpts differ from the table above, because the table uses the
full article text, as the app does. Say so if you switch to the CLI.

## Notes

- The app scores the title plus the article body. Extension 3's evaluation scores the body only, so
  in the app Ext 3 gets 667 of the 1,001 test articles right (66.6%), one more than the 666 (66.5%)
  reported in `results/ext3.json`.
- If a reviewer asks why confidences are often 0% or 100%: the tweet-trained models see long
  articles with many words, so their decision scores are large and the sigmoid saturates. A
  confident score does not mean a likely correct answer, as L3 and R3 show.
