# RideResponse reuse record

The source project at `/Users/huali/Desktop/RideResponse` is unchanged. The prototype runs independently of that path after creation.

The frozen prototype copied two class definitions from `continuous_benchmarks.py` into `airline_market/legacy_toy/ride_response.py`:

- `PairedEstimate`, source lines 53–73, supplies sample means and standard errors.
- `FixedPolicy`, source lines 420–426, supplies the fixed coefficient vector.

Source-file SHA256 at extraction:

```text
1ea5279fa2f11ef1d88c9331eb2f0e21f84c6bedd6fb624b69c2070e4d3743cc
```

The copied classes receive a small independent import header. Extracting them avoids importing `continuous_benchmarks.py`'s recurrent SAC dependencies. Paired seed alignment, confidence intervals, and output serialization are airline-prototype code. RideResponse's small-sample standard-error behavior is preserved; the experiment requires at least two evaluation cohorts and serializes unavailable standard errors as JSON null.

The experiment also adopts RideResponse's explicit-seed paired evaluation and saved-state/common-random-input conventions. Airline-specific noise tapes index all potential requests and all choice alternatives before policy-dependent decisions. Counterfactual branches retain the exogenous draws while recomputing purchases and rival prices.

RideResponse's `RecedingHorizonCEM` explicitly describes itself as a privileged-state diagnostic without a tested public-history belief branch. It is therefore not imported as the public-information comparator. The airline planner reconstructs hypothetical states from observations and exact posterior samples. It cannot access the environment or its realized future tape.

The framework now adapts recurrent PPO and the network counterfactual-credit idea, as recorded below. SAC and CrowdLLM are not implemented. No airline performance result is represented as an advance in those inherited techniques; the airline data compiler is independently implemented.

## Framework v1 vendoring

| Target | Source | Lines | SHA256 | Mode |
| --- | --- | --- | --- | --- |
| airline_market/vendor/ride_response/paired.py | /Users/huali/Desktop/RideResponse/continuous_benchmarks.py | [[53, 73], [126, 142], [420, 426]] | 1ea5279fa2f11ef1d88c9331eb2f0e21f84c6bedd6fb624b69c2070e4d3743cc | verbatim definitions; independent import header |
| airline_market/vendor/ride_response/cem.py | /Users/huali/Desktop/RideResponse/continuous_benchmarks.py | [[596, 612], [615, 695]] | 1ea5279fa2f11ef1d88c9331eb2f0e21f84c6bedd6fb624b69c2070e4d3743cc | verbatim definitions; independent import header |
| airline_market/vendor/ride_response/calendar.py | /Users/huali/Desktop/RideResponse/market_shocks.py | [[73, 89]] | 41e97204eb65de8751999b4bb6ad36566e4c7780135d2f9e2d59a09bf480e410 | verbatim definitions; independent import header |
| airline_market/vendor/ride_response/strategic.py | /Users/huali/Desktop/RideResponse/rideshare_v26r_v17/strategic_benchmarks.py | [[1, 121]] | af7efd587668612e07c17d75a311656eb5c2bca0236cfbda3465a120c7d70ed5 | verbatim |
| airline_market/core/registry.py | /Users/huali/Desktop/RideResponse/model_registry.py | [[1, 108]] | d161e56101cf95e28464782dda66f31e6e3155eb7ff43f84eedd22094f68bf4b | adapted: project default registry directory |
| airline_market/core/journal.py | /Users/huali/Desktop/RideResponse/rideshare_v26r_v17/diagnostic_logging.py | [[1, 137]] | 49b865793287bb0e16bc4f35a4ed2306102563ae2adb5d097be6398ca911e725 | adapted: training_step method name |

## Recurrent PPO airline adaptation

Source `/Users/huali/Desktop/RideResponse/recurrent_ppo.py`, original lines 1–787; original SHA256 f40020975bddb817b157187842b8d06aa41ad85a5467a819ec9830db40724a12. Relative common import, explicit bounded sigmoid coefficients, and actor input mask for critic-only columns.
Source recurrent_sac.py SHA256 291233fbca949cbc24956f1ad059298cf99735a64014eab5498345073a102414, extracted definitions and original line ranges: [['NumericalPolicyActionError', 47, 52], ['ActionTransform', 55, 169], ['Episode', 172, 208], ['EpisodeBuilder', 211, 240], ['RunningMeanStd', 513, 600]]. Removed market_controls; changed bounded map/inverse to sigmoid/logit.

NOAA airport weather index: adapted WeatherObservation and NOAAWeatherIndex construction/nearest/category methods from /Users/huali/Desktop/RideResponse/city_data_pipeline.py lines [[78, 86], [693, 697], [749, 766], [768, 779]], SHA256 3d0a11f625c987ca388b0b0e7635a411823d0c27b91275b65849faa04bdd9aa1. Airport/station file filtering remains in pipeline sources; removed city resolver.

The airline network kernel, observations, filters, shared-capacity pricing and planners are new framework implementations. `sim/batch.py` is an independently implemented specialization verified against the generic sampler, not a delegated legacy simulator. Recurrent adaptations additionally preserve original Gaussian coordinates for online/replay consistency at bounded-action saturation; actor input masks are applied inside every recurrent actor pass. Counterfactual credit is independently written for the airline ledger and is not presented as a new NMCC contribution.

The data field-map approach is adapted from `_field_name`/`_LooseRow` in RideResponse `city_data_pipeline.py` lines 188 onward; no city-specific data compiler is imported. Airline loaders/compiler and calibration moment registry are independently implemented.

NOAA raw-hourly decoders additionally extracted verbatim from city_data_pipeline.py at lines [[207, 210], [646, 656], [659, 674], [677, 687]], same source hash as the weather index; station-scoped IO replaces city-scoped IO.

## Executable structured training (F6)

`learn/structured_runtime.py` imports the frozen recurrent PPO rollout/context/GAE machinery and adapts its optimizer update for the separate leg-cost actor/additive critic and auxiliary BCE. The maintained vendor source is `airline_market/vendor/ride_response/recurrent/ppo.py`, SHA256 `b3162d156ec94831ec34b2ce1bae2b2ce3f80f3feb04582bc253860e4d5ae552`; relevant ranges295–445 (context, actions and on-policy storage),502–720 (preparation and PPO update). No vendor or RideResponse source is changed. It reuses AirlineMarket's `StructuredActorCritic` and `JointEventHead`; sampled latent leg preactivations and residual coefficients map through the airline incidence matrix to prices. Network counterfactual branches and delayed joint-event labels are new airline training integration; generic PPO/counterfactual credit are reused ideas, not claimed as new RideResponse-distinct contributions.

Market-local Poisson tape: adapted the maintained AirlineMarket `sim/tape.py` generation logic into a new module, preserving all non-taste stream keys/order and replacing only global unused tastes with immutable per-request market vectors. Original file unchanged; SHA256 8df7191e59df7e78b3cde608c3695fb8bf145a8990363da8b6c4649b663cf1b8. No RideResponse or frozen legacy/vendor edit. Exact non-taste parity and equivalent-dense engine path/restore checks are in `tests/unit/test_behavioral_network.py`.


## Exact-small belief-feature PPO precursor

New `learn/belief_ppo.py` adapts the native flat loop in `experiments/f6_study.py` lines22–94 (SHA256 `5bf7adb2a47ea9f05e28b1e3e6142f491f5a837388d664d79663ec6682949a3f`), retaining native rollout/PPO updates and adding an exact-small observation-derived adapter, semantic feature/checkpoint contract and exact continuation. It subclasses the existing adapter and imports unchanged collect/save/load; the separate driver reuses the existing paired-bootstrap summary. Full internal source hashes are retained in `results/P2_belief_tooling/reuse_sources.json`. No RideResponse, vendor or frozen legacy source is changed or newly copied. The added belief actor inputs define a separate policy observation class; this is not a new generic PPO algorithm or the full P2 I5 tuning study.


## Persistent-type native MPC precursor

New `game/type_mpc.py` uses the unchanged keyed episode-type/full-ledger Bayesian primitives and adapts native `solvers/mpc.py` scenario branch/price-map semantics (lines180–275, SHA256 `43616a8ff8c8ce461532264110715fa0166d28853c95781e23487c98fac8a4ff`). The new driver reuses existing PolicyRef/MixedController for physical episode-level policy draws; inference explicitly expands that declared type bank. Internal source hashes and adaptations are in results/F7_type_mpc_tooling/reuse_sources.json. No external/vendor/legacy file changed. This adds a mixture-aware information law and actual sampled response implementation, not a new generic MPC algorithm or an exact best-response/Nash certificate.


## Saved candidate power analysis

New `experiments/candidate_power_study.py` implements the plan's section2.4 formula with stdlib NormalDist and NumPy sample SD. It reads the already measured F5 dollar-profit/revenue arrays and original result receipts; it does not copy or change the F5 driver, vendor, RideResponse or legacy code. Internal metric/source provenance is in results/F9_power_tooling/reuse_sources.json; source F5driver SHA256 `40501d7550bf0e056614d63529ce88adb46c885ed3e8c96877e3547ab02d8257`, episode/accounting ranges245–267 and sealed aggregate ranges286–310. Reusing saved cohorts is not a new simulation or a full F9 primary-power/budget result.


## MPC-generated finite-history population

New game/mpc_population.py imports the existing joint-type MPC/filter, full-ledger history-policy representation, native engine/accounting and finite matrix meta-solver. It adds explicit reachable-history materialization, declared off-support tail actions, cached sampled population payoffs and fresh paired deviations for both carriers. Internal source hashes and adaptations are retained in results/F7_population_tooling/reuse_sources.json. No vendor, legacy or RideResponse bytes are changed or copied. The finite response class and sampled deviation evidence do not establish exact best responses or game-wide Nash equilibrium.


## Native curriculum/domain-sampling engineering precursor

New learn/curriculum_ppo.py imports unchanged native PPO collect/save/load, RecurrentEnvAdapter, original threshold/matching/DLP controllers and existing persistent-type MPC/history-mixture controllers. It adds cohort-stage/domain selection and an ensemble/meta-aware continuation contract around the original optimizer. Source hashes are in results/TR_curriculum_tooling/reuse_sources.json. No external/vendor/RideResponse/legacy file is edited or copied. Its declared reference model of a learning focal PPO is misspecified; this is not new generic PPO/MPC, an accurate learner-inference certificate or the calibrated24+8/full TR training program.


## Read-only gradient probes and TR5 monitor

New learn/gradient_monitor.py imports native PPO full-batch preparation and clipped loss, and reproduces the native flat/structured actor-loss reductions for a read-only autograd probe. It includes existing entropy and auxiliary terms and preserves native R2 advantage blending/normalization. Independent hooks compare with actual native backward gradients for PPO/SRL/C/CA. New experiments/monitored_learning.py imports unchanged structured collectors, updates and checkpoint load/save; completed throughput states are continued, not retrained. Internal source hashes and protocol references are in results/gradient_monitor_tooling/reuse_sources.json. The scalar writer implements public Event/Summary/TFRecord wire facts; no TensorFlow code, vendor, legacy or RideResponse source is copied/edited. External TensorBoard consumer verification remains unrun.

2026-11-13 | A.8 / B0 / V9 | Native cap evidence reuses project pricing hooks.
- `policy/cap_evidence.py` inserts a cooperative hook after `sim/engine.py:MarketEngine._action_prices` and before `policy/regulator.py` clipping, with an outer hook after `policy/trigger.py` execution. No copied pricing or repeated actor/rule/trigger forward; original modules unchanged.
- Exact source ranges/SHA and adaptation are in `results/V9_cap_evidence_tooling/native_adaptation.json`; dated executor and ordinary-pool join are native project reuse. No external/RideResponse/legacy_toy/vendor edits.

2026-11-13 | A.8 / B0 / V9 | Whole-period native request admission reuses parent closure.
- `policy/rationed_native.py` supplies a lazy iterable to the frozen native `step` after posting/re-entry. Original Request objects alone are reordered; choice/book/refund/resolve/outcome callback/closure stay inherited, with no repeated controller or physical draw.
- Frozen current-pool/static-priority modules supply a comparison witness. Function ranges, file SHA and adaptations: `results/V9_native_admission_tooling/native_adaptation.json`. No external, RideResponse, legacy_toy or vendor changes.

2026-11-13 | A.5 / A.8 / B0 / V9 | Prospective potential-slot lineage reuses native marks.
- `policy/potential_lineage.py` wraps the existing lazy ordinary iterator and captures native Book/request correspondence before parent cancellation/closure. Frozen `RebookingMarketEngine._marks` is delegated once with a temporary immutable coordinate view; actual Booking rank and old algorithms unchanged.
- Original root/replacement maps, native booking/proof/refund paths and independent snapshots are reused. Function ranges/SHA/adaptations: `results/V9_potential_lineage_tooling/native_adaptation.json`. No external, RideResponse, legacy_toy or vendor changes.

2026-11-13 | A.5 / B0 / B1 / V9 | Shared demand envelope reuses native sampling/thinning.
- `policy/shared_demand.py` compiles declared maximum stratum rates and delegates one `ExogenousTape.generate`; projections transform acceptance scores while retaining all potential rows and frozen auxiliary arrays. Native `period_requests` performs state thinning; no sampler/arrival algorithm copied.
- Source ranges/SHA/adaptations: `results/V9_shared_demand_tooling/native_adaptation.json`. Original potential-lineage/native booking integration tested; no external, RideResponse, legacy_toy or vendor edits.

2026-11-13 | A.5 / B0 / B1 / V9 | Temporary capacity-window executor v2.
  | Native ledger resolve (65–89), disruption boundary/stranding hooks, rebooking and reserve inherited unchanged; full source ranges/SHAs in results/V9_capacity_windows_tooling/native_adaptation{,_v2}.json.
  | Retained v1 overlay supplies expiry/exact floor/balance contracts; v2 batches same-boundary window losses before one native resolve/carrier. Frozen v1 sequential-overcancellation limitation retained with fresh counterexample, no saved scientific replay.
  | No RideResponse/legacy_toy/vendor edits or native lifecycle copy. New sources and parent SHAs in source_record{,_v2}.json; actual pre-commit Git cfddc5d, later commit not retroactive provenance.

2026-11-13 | A.5 / B0 / B1 / V9 | Dated physical shock adapter.
  | Reuse aware-UTC/exact-boundary helpers, frozen capacity-window v2 and RivalExit by delegation; exact source ranges/SHAs in results/V9_dated_shocks_tooling/native_adaptation.json.
  | New pure B1 design-to-date/window projection and nested date-contract snapshots only; native choice/book/refund/resolve/lineage/callback lifecycle inherited. No RideResponse/legacy_toy/vendor changes or simulator copy.
  | Sources/fixtures captured in source_record.json at actual Git 4e887cd plus then-new SHAs; later commit not retroactive clean-tree provenance. No old scientific results rewritten/replayed.

2026-11-13 | A.5 / B0 / B1 / V9 | Dated demand and hurricane composition.
  | Frozen shared_poisson_bank/project, dated_shocks public-date/physical executor, native segment_weights and PeriodSchedule.value_at reused by delegation; source ranges/SHAs in results/V9_dated_demand_tooling/native_adaptation.json.
  | New pure rate/date/finite-alpha nominal CDF projection, scenario contracts and composite factory; no native sampler/simulator/ledger copy or RideResponse/legacy_toy/vendor edit.
  | Actual source Git9c78c58 plus new frozen SHAs in source_record.json; later commit not retroactive provenance. No old science/source-bank/frame/review replay.

2026-11-13 | B2 / H2.1 | Declared paired spike-decomposition arithmetic.
  | Reuse frozen dated_reserve.utc and core.snapshot.deep_freeze only; source ranges/SHAs in results/B2_decomposition_tooling/native_adaptation.json. New exact rational formula follows complete v2 Section9.4 B2 card.
  | No simulator/controller/filter/kernel/sampler/ledger copy, native source edit or RideResponse/legacy_toy/vendor change. Input hash declarations are NOT authenticated source/transitive seals or economic-channel validity.
  | Actual source Git99420b2 plus new frozen SHAs in source_record.json; later commit not retroactive provenance. No old science/source-bank/frame/review replay.

2026-11-13 | B2 / H2.1 | Native panel measurement delegation.
  | airline_market/sim/engine.py:319-425 SHA256 10156a9bfb0c03de07cd418518e4f6764f616d71a845fa34120829648a676485: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | airline_market/sim/engine.py:438-442 SHA256 10156a9bfb0c03de07cd418518e4f6764f616d71a845fa34120829648a676485: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | airline_market/sim/rebooking.py:286-406 SHA256 e560ab296537f3b540934faad6b345c2ef5890ab9743275fbc796ff2d8209587: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | airline_market/core/hashing.py:11-26 SHA256 4f6890477eba79ec3e99743e38482244e9a0c193ad24a49e42d8ab93a5416ec4: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | airline_market/core/hashing.py:47-52 SHA256 4f6890477eba79ec3e99743e38482244e9a0c193ad24a49e42d8ab93a5416ec4: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | airline_market/eval/spike_decomposition.py:22-53 SHA256 445f89fde26520039bee12037ace0d5116f1cc2cd19a06302410ddd627c03b32: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | airline_market/eval/spike_decomposition.py:56-61 SHA256 445f89fde26520039bee12037ace0d5116f1cc2cd19a06302410ddd627c03b32: native lifecycle unchanged; trace projection/schema or exact scalar helper reused only.
  | New fare_panel_v2 checks exposed public UTC clocks, then delegates one v1 projection; frozen v1 clock limitation retained. Full source/ranges in B2_fare_panel_tooling/native_adaptation*.json; no RideResponse/vendor code copied.

2026-11-13 | A.15 / B1 / V10 | Prepared service-calendar count adapter.
  | airline_market/pipeline/event_network.py:246-580, SHA256 88789a20defa60d91bce66dcc34184b95dceff229e20f7557572039e3a237e57: existing saved SQLite schema/identity definition referenced only; ingestion and identity construction not rerun.
  | airline_market/pipeline/event_seats.py:47-68, SHA256 a5fa08a92ffc439f102ce99cc58b2cc61cc9997afe595cc267ac73a6f663a1f2: existing saved SQLite schema/identity definition referenced only; ingestion and identity construction not rerun.
  | airline_market/pipeline/event_seats.py:71-182, SHA256 a5fa08a92ffc439f102ce99cc58b2cc61cc9997afe595cc267ac73a6f663a1f2: existing saved SQLite schema/identity definition referenced only; ingestion and identity construction not rerun.
  | New read-only sid=row_id/date join and exact reported-flight ratios; no native simulator/controller/RNG or RideResponse/vendor code copied. Source/range adaptation manifest retained in B1_event_counts_tooling.

2026-11-13 | A.5 / B0 / B1 / V9 | All-type dated shared-family delegation.
  | airline_market/policy/dated_demand.py:195-293 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | airline_market/policy/dated_demand.py:296-317 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | airline_market/policy/dated_demand.py:394-405 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | airline_market/policy/dated_demand.py:380-384 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | airline_market/policy/dated_shocks.py:151-231 SHA256 4c4f3c91be1359a16ad0851ba08dd7c32dbd8715e8892b2ac85a840172b21eae: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | airline_market/policy/shared_demand.py:103-158 SHA256 843df4ea3f24928dae0452e45d8f20d8ca7440b58dbb44f5e6c8dc4b05a4bf1f: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | airline_market/policy/shared_demand.py:63-82 SHA256 843df4ea3f24928dae0452e45d8f20d8ca7440b58dbb44f5e6c8dc4b05a4bf1f: delegate dated projection/shared envelope/native engine without copying its algorithm.
  | New typed declaration/lift/factory only; no simulator/ledger/sampler/controller, RideResponse or vendor edits. Detailed source/range adaptation manifest retained in V9_dated_b1_family_tooling.

2026-11-13 | A.8 / A.5 / B0 / V9 | Native static allocation-comparison sidecar.
  | airline_market/policy/rationing.py:227-280 SHA256 4b6f55e9f1f18119be6303f71d507095f04fb98fb26f24eafe3810646b71c657: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | airline_market/policy/rationing.py:283-294 SHA256 4b6f55e9f1f18119be6303f71d507095f04fb98fb26f24eafe3810646b71c657: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | airline_market/policy/rationing_native.py:219-228 SHA256 eb637d4e6c88a76257cb1a97a020203a63c136260e8dafcc26c57ff65d4514c6: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | airline_market/policy/rationing_native.py:231-269 SHA256 eb637d4e6c88a76257cb1a97a020203a63c136260e8dafcc26c57ff65d4514c6: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | airline_market/policy/rationed_native.py:81-112 SHA256 8e22f9416814c0943e7165a920e22b733d298dc32e648f4dd84c369d7bdc981a: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | airline_market/policy/rationed_native.py:114-168 SHA256 8e22f9416814c0943e7165a920e22b733d298dc32e648f4dd84c369d7bdc981a: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | airline_market/policy/dated_demand.py:358-372 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: delegate pool/priority/solver and adapt snapshot/branch constructor plumbing only.
  | New reweight/full witness/receipt comparison, no efficient booking or copied native simulator/ledger/choice/callback/RNG. Source/range adaptations retained; no RideResponse/vendor edits.

2026-11-13 | A.8 / B0 / V9 | Exact native-choice full-order analyst bound.
  | airline_market/policy/rationing.py:65-114 SHA256 4b6f55e9f1f18119be6303f71d507095f04fb98fb26f24eafe3810646b71c657: delegate exact native choice/full witness/export; no native booking lifecycle copied.
  | airline_market/policy/rationing.py:166-220 SHA256 4b6f55e9f1f18119be6303f71d507095f04fb98fb26f24eafe3810646b71c657: delegate exact native choice/full witness/export; no native booking lifecycle copied.
  | airline_market/policy/rationing.py:283-294 SHA256 4b6f55e9f1f18119be6303f71d507095f04fb98fb26f24eafe3810646b71c657: delegate exact native choice/full witness/export; no native booking lifecycle copied.
  | airline_market/policy/allocation_comparison.py:80-110 SHA256 6ed2e1ba7179cfa2006c07b37c81bf2bccc35aea225ab4f9227ca6da52dfbfd8: delegate exact native choice/full witness/export; no native booking lifecycle copied.
  | airline_market/policy/allocation_comparison.py:114-124 SHA256 6ed2e1ba7179cfa2006c07b37c81bf2bccc35aea225ab4f9227ca6da52dfbfd8: delegate exact native choice/full witness/export; no native booking lifecycle copied.
  | airline_market/policy/allocation_comparison.py:301-344 SHA256 6ed2e1ba7179cfa2006c07b37c81bf2bccc35aea225ab4f9227ca6da52dfbfd8: delegate exact native choice/full witness/export; no native booking lifecycle copied.
  | New remaining-pool/stock dynamic program and prospective score bracket; frozen parents unchanged, no efficient policy/RNG/actor/world copy. Full method ranges/hashes in V9_native_order_bound_tooling/native_adaptation.json; no RideResponse/vendor edits.

2026-11-13 | A.8 / A.5 / B0 / V9 | Dated native-order benchmark delegation.
  | airline_market/policy/potential_lineage.py:55-107 SHA256 b1dec2108e330551f3e9e5bbc5f86bfdf57fc55f89de2b4b0f10fac078520e50: lazy full-pool/receipt plumbing adapted; native lifecycle/lineage delegated.
  | airline_market/policy/potential_lineage.py:109-112 SHA256 b1dec2108e330551f3e9e5bbc5f86bfdf57fc55f89de2b4b0f10fac078520e50: lazy full-pool/receipt plumbing adapted; native lifecycle/lineage delegated.
  | airline_market/policy/potential_lineage.py:117-126 SHA256 b1dec2108e330551f3e9e5bbc5f86bfdf57fc55f89de2b4b0f10fac078520e50: lazy full-pool/receipt plumbing adapted; native lifecycle/lineage delegated.
  | airline_market/policy/rationed_native.py:81-112 SHA256 8e22f9416814c0943e7165a920e22b733d298dc32e648f4dd84c369d7bdc981a: lazy full-pool/receipt plumbing adapted; native lifecycle/lineage delegated.
  | airline_market/policy/rationed_native.py:114-168 SHA256 8e22f9416814c0943e7165a920e22b733d298dc32e648f4dd84c369d7bdc981a: lazy full-pool/receipt plumbing adapted; native lifecycle/lineage delegated.
  | airline_market/policy/dated_demand.py:346-349 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: lazy full-pool/receipt plumbing adapted; native lifecycle/lineage delegated.
  | New source hook/order receipt only; exact optimizer/current-pool/dated parent delegation, no RideResponse/vendor/native ledger or choice edits. Full ranges in V9_dated_order_admission_tooling/native_adaptation.json.

2026-11-14 | A.10 / A.5 / B0 / V9 | Original-passenger fare account reader.
  | airline_market/sim/state.py:12-22 SHA256 50fe3544ae35a49b3315b710210cdecae0506824cbc90d8917a3a6ebaad9a029: native ticket/cancellation/cash schema read/reference only; no physical algorithm copied or called.
  | airline_market/sim/state.py:26-60 SHA256 50fe3544ae35a49b3315b710210cdecae0506824cbc90d8917a3a6ebaad9a029: native ticket/cancellation/cash schema read/reference only; no physical algorithm copied or called.
  | airline_market/sim/ledger.py:29-51 SHA256 3f47549ba6f4e2bcd1955f04c77627df6c053b889154ecf2fcd3286689d7dc40: native ticket/cancellation/cash schema read/reference only; no physical algorithm copied or called.
  | airline_market/sim/ledger.py:65-89 SHA256 3f47549ba6f4e2bcd1955f04c77627df6c053b889154ecf2fcd3286689d7dc40: native ticket/cancellation/cash schema read/reference only; no physical algorithm copied or called.
  | airline_market/sim/disruption.py:123-127 SHA256 e385b4f80a25a656673ee33d56a620a5a26e3f6faaade5eb074c1287a80a2854: native ticket/cancellation/cash schema read/reference only; no physical algorithm copied or called.
  | airline_market/sim/disruption.py:177-183 SHA256 e385b4f80a25a656673ee33d56a620a5a26e3f6faaade5eb074c1287a80a2854: native ticket/cancellation/cash schema read/reference only; no physical algorithm copied or called.
  | New complete root/ticket/search join and exact binary64 fare envelopes/residuals; no penalty allocation/lifecycle utility inferred. Full method references in V9_passenger_accounts_tooling/native_adaptation.json; no RideResponse/vendor edits.

2026-11-14 | A.10 / A.5 / B0 / V9 | All-ordinary prospective identity sidecar.
  | airline_market/policy/potential_lineage.py:55-107 SHA256 b1dec2108e330551f3e9e5bbc5f86bfdf57fc55f89de2b4b0f10fac078520e50: genuine-request/lifecycle delegation and snapshot/branch constructor adaptation only.
  | airline_market/policy/potential_lineage.py:117-126 SHA256 b1dec2108e330551f3e9e5bbc5f86bfdf57fc55f89de2b4b0f10fac078520e50: genuine-request/lifecycle delegation and snapshot/branch constructor adaptation only.
  | airline_market/policy/dated_demand.py:346-349 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: genuine-request/lifecycle delegation and snapshot/branch constructor adaptation only.
  | airline_market/policy/dated_demand.py:351-356 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: genuine-request/lifecycle delegation and snapshot/branch constructor adaptation only.
  | airline_market/policy/dated_demand.py:358-372 SHA256 5f1b895f29f2d9a5054c9c0412dfbc3c424ae7fc41483751f24710a34da88052: genuine-request/lifecycle delegation and snapshot/branch constructor adaptation only.
  | airline_market/policy/dated_order_admission.py:199-213 SHA256 b2a186edb62c86ce16b59eaaa69430da90db78ab770f57f5bf2d7c6d19097839: genuine-request/lifecycle delegation and snapshot/branch constructor adaptation only.
  | Frozen binary64 encoder/immutable exports reused; supplied NEW account/period objects joined without re-running producers. No RideResponse/vendor/physical algorithm edits. Full ranges in V9_ordinary_cohort_tooling/native_adaptation.json.

2026-11-14 | A.10 / B0 / V9 | Analytic ordinary Poisson depletion block.
  | airline_market/sim/choice.py:8-42 SHA256 1d8475acb13545ca17f3024c63234efe86047903cc9e46cca964e668a724ece1: delegated native utility/MNL/own-alpha logsum; no sampler/physical execution copied.
  | airline_market/sim/choice.py:45-49 SHA256 1d8475acb13545ca17f3024c63234efe86047903cc9e46cca964e668a724ece1: delegated native utility/MNL/own-alpha logsum; no sampler/physical execution copied.
  | airline_market/sim/segmented_welfare.py:15-36 SHA256 f820c4018f4a9ad42e8f3a05908ac9ddbf8112af8f669d0c2ede738d9f0dc724: delegated native utility/MNL/own-alpha logsum; no sampler/physical execution copied.
  | Frozen hash/immutable-export helpers delegated; complete ranges in V9_poisson_period_tooling/native_adaptation.json. Installed SciPy1.15.3 expm source/SHA in backend_record.json; small dense Float64 backend, no symbolic exactness/rigorous interval or planner adoption. No RideResponse/legacy/vendor edits.

2026-11-14 | A.10 / B0 / V9 | Finite joint ordinary Poisson forecast expectation.
  | airline_market/sim/poisson_period.py:27-91 SHA256 00be3710745d1dc75042b7b32e1207fc9019357b72665972fec940262002c6bb: frozen type/conditional operator delegated at propagated integer stock; no native engine/sampler or planner copied.
  | airline_market/sim/poisson_period.py:94-248 SHA256 00be3710745d1dc75042b7b32e1207fc9019357b72665972fec940262002c6bb: frozen type/conditional operator delegated at propagated integer stock; no native engine/sampler or planner copied.
  | Hash/immutable JSON helpers delegated; full ranges in V9_poisson_forecast_tooling/native_adaptation.json. New full-stock propagation/whole-path total expectation only, Float64/no pruning/no symbolic exactness or rigorous interval. Parent dense-backend restrictions inherited; no RideResponse/legacy/vendor edits.

2026-11-14 | A.10 / A.5 / B0 / V9 | Prospective original-person closing reservations.
  | airline_market/sim/ordinary_cohort.py:95-125 SHA256 43c231c3ddb1f68c16994f9f4b88c557d4d9a5d85d34d64b4620cd2a4d5df57a: one complete parent step/ordinary history and nested snapshot interface delegated; no lifecycle/ledger algorithm copied.
  | airline_market/sim/ordinary_cohort.py:127-132 SHA256 43c231c3ddb1f68c16994f9f4b88c557d4d9a5d85d34d64b4620cd2a4d5df57a: one complete parent step/ordinary history and nested snapshot interface delegated; no lifecycle/ledger algorithm copied.
  | airline_market/sim/ordinary_cohort.py:134-139 SHA256 43c231c3ddb1f68c16994f9f4b88c557d4d9a5d85d34d64b4620cd2a4d5df57a: one complete parent step/ordinary history and nested snapshot interface delegated; no lifecycle/ledger algorithm copied.
  | New analyst-only ledger point capture/full original-person follow-up summary; frozen potential-coordinate/hash/immutable JSON interfaces referenced, full ranges in V9_reservation_census_tooling/native_adaptation.json. No account/timing producer replay or RideResponse/legacy/vendor edits.

2026-11-14 | A.10 / B0 / V9 | Declared single-commitment payoff logit information timings.
  | airline_market/sim/segmented_welfare.py:15-36 SHA256 f820c4018f4a9ad42e8f3a05908ac9ddbf8112af8f669d0c2ede738d9f0dc724: frozen static logsum/immutable export delegated; no native dynamics/choice algorithm copied.
  | airline_market/core/hashing.py:11-26 SHA256 4f6890477eba79ec3e99743e38482244e9a0c193ad24a49e42d8ab93a5416ec4: frozen static logsum/immutable export delegated; no native dynamics/choice algorithm copied.
  | airline_market/core/snapshot.py:84-95 SHA256 e0717055c63ec6867d1c8b8ba9375753944ca8acc5f19dea3590fdb7edd60b2a: frozen static logsum/immutable export delegated; no native dynamics/choice algorithm copied.
  | New explicit rational payoff table/risk-neutral single commitment/two information timings only, not native lifecycle welfare or empirical preference adoption. Full assumptions/ranges in V9_payoff_logit_tooling/native_adaptation.json; no RideResponse/legacy/vendor edits.

2026-11-14 | A.10 / A.5 / B0 / V9 | Recurrent unreserved spans on declared native UTC opportunities.
  | airline_market/policy/dated_reserve.py:15-26 SHA256 a47f6d96a2bbc00879872739125de1f0687fd38e1d3ccf13fa149feb7ffa4ee6: Delegate explicit aware UTC conversion and declared native opportunity lookup, no clock inference/interpolation.
  | airline_market/policy/dated_reserve.py:30-48 SHA256 a47f6d96a2bbc00879872739125de1f0687fd38e1d3ccf13fa149feb7ffa4ee6: Delegate explicit aware UTC conversion and declared native opportunity lookup, no clock inference/interpolation.
  | airline_market/policy/dated_rescue.py:152-156 SHA256 d4607ae5c4fdd0372b47e60d7f0fa079a8a8c8d5cd9251fae276226324dba39c: Read-only source-model timing contract reviewed; no execution/delegation/copy or timing report call by new consumer.
  | NEW pure complete-account consumer/exact microsecond endpoint arithmetic; frozen account/deadline/JSON interfaces referenced, full ranges in V9_unreserved_spells_tooling/native_adaptation.json. No native dynamics/ledger algorithm copied, original producer/timing report replay or RideResponse/legacy/vendor edit.

2026-11-14 | A.10 / A.5 / B0 / V9 | Analytic native own-first/interline shared-outside choice.
  | airline_market/sim/rebooking.py:119-143 SHA256 e560ab296537f3b540934faad6b345c2ef5890ab9743275fbc796ff2d8209587: Frozen source-model contract: same full product/outside draw across own/interline; period-keyed Gumbels/root-coordinate adjusted taste. No re-entry/booking algorithm copied or invoked by runtime; NEW BASE unit witnesses invoke actual native execution.
  | airline_market/sim/rebooking.py:145-216 SHA256 e560ab296537f3b540934faad6b345c2ef5890ab9743275fbc796ff2d8209587: Frozen source-model contract: same full product/outside draw across own/interline; period-keyed Gumbels/root-coordinate adjusted taste. No re-entry/booking algorithm copied or invoked by runtime; NEW BASE unit witnesses invoke actual native execution.
  | airline_market/sim/choice.py:8-42 SHA256 1d8475acb13545ca17f3024c63234efe86047903cc9e46cca964e668a724ece1: Runtime delegates systematic_utilities once on supplied complete current native context; sampled_choice used by NEW independent native witness, not runtime.
  | NEW exponential-race/tail integration and one native-utility projection; frozen binary64/JSON interfaces referenced, full ranges in V9_ordered_reentry_logit_tooling/native_adaptation.json. No account/timing/study replay or RideResponse/legacy/vendor edit; no lifecycle CS/primary B4 adoption.

2026-11-14 | A.10 / A.5 / B0 / V9 | One-person finite declared re-entry policy payoff.
  | airline_market/sim/ordered_reentry_logit.py:28-66 SHA256 db59139e98c4dc692cb36182b8652c234e469e71e4d6dd09ea563a47f5f6e9a3: Delegate frozen one-current-attempt ordered probabilities on NEW declared states, retaining full records/residuals; ignore per-search expected utility gains and never rerun old acceptance/review.
  | airline_market/sim/ordered_reentry_logit.py:77-159 SHA256 db59139e98c4dc692cb36182b8652c234e469e71e4d6dd09ea563a47f5f6e9a3: Delegate frozen one-current-attempt ordered probabilities on NEW declared states, retaining full records/residuals; ignore per-search expected utility gains and never rerun old acceptance/review.
  | airline_market/sim/payoff_logit.py:15-20 SHA256 f630ec9837bcad6d3c99ba773a6daa3915b4805a039ab7b175c9ffd791476482: Reuse explicit FOUR rational-or-unknown payoff schema and exact ratio export only; no single-commitment timing evaluator or old payoff result called.
  | NEW complete finite type/state/conditional-branch payoff recursion, exact rational weighting of numerical probabilities; full references in V9_reentry_payoff_tooling/native_adaptation.json. No native lifecycle algorithm copied or RideResponse/legacy/vendor edit; no native CS/primary B4 adoption.
