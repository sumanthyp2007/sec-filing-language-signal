# Do Changes in Filing Language Predict Post-Release Returns? (2-page write-up template)

Fill every bracket from `outputs/results.md` after a REAL run. Delete this line when done.

## 1. Question and hypothesis
Do quarter-over-quarter changes in guidance direction, hedging, tone, and supply-chain / FX / demand
emphasis in [10-K/10-Q MD&A | earnings releases | transcripts] predict abnormal returns over the next
1, 5 and 20 trading days for [N] large-cap US stocks, [start]-[end]? Signs were fixed before looking at
returns (see `config.FEATURE_SIGNS`).

## 2. Data and method
- Universe: [N] tickers, [source], survivorship-biased (current constituents).
- Documents: [counts by source]. Extractor: [LLM model + prompt version | lexicon]. Documents truncated at [MAX_CHARS] chars.
- Signal: QoQ change per feature, z-scored within source and calendar quarter, winsorised at +/-3, equal-weighted with prior signs.
- Event study: market model vs SPY, estimated on days [-250,-30]; day 0 = first tradable session after acceptance; offset [0|1].
- Tests: quintile long-short per quarter (t-stat across quarters), rank IC, regression with quarter-clustered SEs, OOS split at [date].

## 3. Results
| Horizon | IC | IC t | LS gross | LS net ([x] bps) | OOS IC |
|---|---|---|---|---|---|
| 1d | | | | | |
| 5d | | | | | |
| 20d | | | | | |

[Embed quintile_car.png, event_path.png. Describe which features drive it (feature_coefs.png).]
[Compare LLM vs lexicon: does the LLM add anything?]

## 4. Limitations (be specific about which apply to your run)
1. Survivorship: universe is today's large caps.
2. LLM memorisation: the model may have seen these documents and their market reaction in training. Check the effect in periods after the model's training cutoff.
3. Timing: 10-K/10-Q often follow the earnings release by days or weeks, so for `periodic` documents much of the news is already priced. Releases/transcripts are the cleaner event.
4. Overlapping windows and cross-sectional correlation inflate t-stats; quarter-clustering helps but does not remove this.
5. Multiple testing: [k] horizons x [k] features x [k] sources were examined.
6. Costs and capacity: [bps] assumed; large-cap only; no borrow costs for the short leg.
7. Extraction noise: [accuracy from a hand-labelled sample of N documents].

## 5. Conclusion
[One honest paragraph: effect size, whether it survives costs and OOS, what you would do next.]
