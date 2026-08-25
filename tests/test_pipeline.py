import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import finalize
import generate
import analyze_results
import evaluate_judge_audit
import evaluate_layer3
import judge_data
import prepare_layer3
import prepare_judge_audit
import score_layer1
import score_layer2
import select_judge_variant
import translate
from train_judge import (_model_load_kwargs, _per_tag_recall,
                         _tokenize_batch)
from train_layer2_variants import stratified_row_split
from judge_data import (grouped_split, grouped_train_validation_test_split,
                        load_judge_rows)
from wahm_text import normalize_arabic, token_f1
from scripts.verify_hf_snapshot import run as verify_hf_snapshot


ROOT = Path(__file__).resolve().parents[1]


class ArabicTextTests(unittest.TestCase):
    def test_normalization_unifies_common_forms(self):
        self.assertEqual(normalize_arabic("إجابةٌ"), "اجابه")

    def test_token_f1_is_symmetric(self):
        left = token_f1("أين يقع المسجد؟", "وين يقع المسجد")
        right = token_f1("وين يقع المسجد", "أين يقع المسجد؟")
        self.assertEqual(left, right)


class LayerOneTests(unittest.TestCase):
    def test_exact_reference_still_defers_to_factual_judge(self):
        coverage, decision, reasons = score_layer1.score_answer(
            "فازت الأوروغواي ببطولتين.", "بطولتين")
        self.assertEqual((coverage, decision, reasons), (1.0, "defer", []))

    def test_partial_reference_defers(self):
        _, decision, _ = score_layer1.score_answer("بيير كوري", "بيير كوري وأخوه جاك")
        self.assertEqual(decision, "defer")

    def test_no_reference_overlap_still_defers_to_factual_judge(self):
        _, decision, _ = score_layer1.score_answer("لا أعرف", "بطولتين")
        self.assertEqual(decision, "defer")

    def test_stray_foreign_fragment_does_not_override_factual_routing(self):
        _, decision, reasons = score_layer1.score_answer(
            "بطولتين 中文", "بطولتين")
        self.assertEqual((decision, reasons), ("defer", []))

    def test_heavy_foreign_script_is_degeneration(self):
        _, decision, reasons = score_layer1.score_answer(
            "هذا جواب 中文中文中文中文中文", "جواب")
        self.assertEqual(decision, "degeneration")
        self.assertIn("foreign_script_heavy", reasons)

    def test_known_normalization_edge_cases_never_receive_factual_labels(self):
        for answer, gold in (("تفليس", "تبليسي"),
                             ("فرنسية", "فرنسي"),
                             ("7", "سبع")):
            with self.subTest(answer=answer, gold=gold):
                _, decision, reasons = score_layer1.score_answer(answer, gold)
                self.assertEqual((decision, reasons), ("defer", []))

    def test_empty_or_failed_generation_is_invalid(self):
        _, decision, reasons = score_layer1.score_answer("", "إجابة", "timeout")
        self.assertEqual(decision, "degeneration")
        self.assertEqual(reasons, ["generation_error", "empty_answer"])

    def test_decimal_comma_is_not_treated_as_variant_separator(self):
        variants = score_layer1.gold_variants(
            "37,5 مليون, 37,5 مليون نسمة تقريبا")
        self.assertEqual(variants[0], "37,5 مليون")
        self.assertEqual(len(variants), 2)

    def test_retry_log_keeps_latest_success_only(self):
        base = {"qid": "q1", "variety": "msa", "condition": "direct",
                "model": "m"}
        rows = [dict(base, answer="", error="timeout"),
                dict(base, answer="answer", error="")]
        canonical = score_layer1.canonical_generations(rows)
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["answer"], "answer")


class LayerTwoRoutingTests(unittest.TestCase):
    def test_factual_rows_use_validation_threshold(self):
        self.assertEqual(
            score_layer2.final_decision("defer", 0.7, 0.6),
            ("factual_hallucination", 1, 1))
        self.assertEqual(
            score_layer2.final_decision("defer", 0.5, 0.6),
            ("clean", 0, 0))

    def test_degeneration_is_preserved_outside_factual_binary_label(self):
        self.assertEqual(
            score_layer2.final_decision("degeneration", None, 0.6),
            ("degeneration", "", 1))

    def test_unexpected_layer1_route_is_rejected(self):
        with self.assertRaises(ValueError):
            score_layer2.final_decision("clean", None, 0.6)

    def test_threshold_override_requires_explicit_acknowledgement(self):
        metadata = {"decision_threshold": 0.235}
        self.assertEqual(score_layer2.resolve_threshold(metadata), 0.235)
        with self.assertRaises(ValueError):
            score_layer2.resolve_threshold(metadata, 0.5)
        self.assertEqual(
            score_layer2.resolve_threshold(metadata, 0.5, True), 0.5)


class JudgeInputAblationTests(unittest.TestCase):
    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"input_ids": [[1]] for _ in args[0]}

    def test_question_gold_answer_formats_both_segments(self):
        tokenizer = self.RecordingTokenizer()
        _tokenize_batch(tokenizer, {
            "question": ["متى تأسست؟"],
            "gold": ["1878"],
            "answer": ["سنة 1878"],
        }, "question_gold_answer")
        args, _ = tokenizer.calls[0]
        self.assertEqual(args[0], ["السؤال: متى تأسست؟\nالإجابة المرجعية: 1878"])
        self.assertEqual(args[1], ["الإجابة المرشحة: سنة 1878"])

    def test_question_variant_rejects_missing_question(self):
        with self.assertRaises(ValueError):
            _tokenize_batch(self.RecordingTokenizer(), {
                "question": [""], "gold": ["g"], "answer": ["a"],
            }, "question_gold_answer")

    def test_per_tag_recall_reports_overlapping_binary_tags(self):
        rows = [{column: "0" for column in judge_data.LABEL_COLUMNS}
                for _ in range(2)]
        rows[0]["Named-Entity Hallucination"] = "1"
        rows[0]["Factual Contradiction"] = "1"
        rows[1]["Named-Entity Hallucination"] = "1"
        metrics = _per_tag_recall(rows, [0, 1], [0.9, 0.1], 0.5)
        self.assertEqual(metrics["Named-Entity Hallucination"],
                         {"n": 2, "recall": 0.5})
        self.assertEqual(metrics["Factual Contradiction"],
                         {"n": 1, "recall": 1.0})

    def test_variant_selection_uses_validation_roc_auc(self):
        def metadata(variant, roc_auc, f1):
            return {"input_variant": variant, "test_metrics": None,
                    "validation_metrics": {"roc_auc": roc_auc, "f1": f1}}

        variants = {
            "answer_only": metadata("answer_only", 0.8, 0.9),
            "gold_answer": metadata("gold_answer", 0.85, 0.8),
            "question_gold_answer": metadata(
                "question_gold_answer", 0.9, 0.7),
        }
        winner, ranking = select_judge_variant.select(variants)
        self.assertEqual(winner, "question_gold_answer")
        self.assertEqual(ranking[0], winner)

    def test_variant_selection_rejects_exposed_test_metrics(self):
        variants = {
            variant: {"input_variant": variant, "test_metrics": None,
                      "validation_metrics": {"roc_auc": 0.8, "f1": 0.8}}
            for variant in ("answer_only", "gold_answer",
                            "question_gold_answer")
        }
        variants["answer_only"]["test_metrics"] = {"f1": 1.0}
        with self.assertRaises(ValueError):
            select_judge_variant.select(variants)

    def test_training_seeds_before_model_construction(self):
        source = (ROOT / "train_judge.py").read_text(encoding="utf-8")
        self.assertLess(source.index("set_seed(42)"),
                        source.index("AutoModelForSequenceClassification.from_pretrained"))
        self.assertIn("full_determinism=True", source)


class TranslationTests(unittest.TestCase):
    def test_existing_confirmed_gulf_file_loads(self):
        old = os.getcwd()
        os.chdir(ROOT)
        try:
            pairs, _ = translate.load_exemplars("gulf")
        finally:
            os.chdir(old)
        self.assertEqual(len(pairs), 10)

    def test_valid_exemplar_file_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exemplars_test.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[
                    "msa_question", "dialect_question", "sub_variety"])
                writer.writeheader()
                for i in range(5):
                    writer.writerow({"msa_question": f"msa {i}",
                                     "dialect_question": f"dia {i}",
                                     "sub_variety": "Test variety"})
            old = os.getcwd()
            os.chdir(directory)
            try:
                pairs, variety = translate.load_exemplars("test")
            finally:
                os.chdir(old)
            self.assertEqual(len(pairs), 5)
            self.assertEqual(variety, "Test variety")

    def test_translation_resumes_without_repeating_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "wahm_seed_msa.csv").write_text(
                "qid,question_msa,gold_answer\nq1,سؤال,جواب\n", encoding="utf-8")
            with (directory / "exemplars_test.csv").open(
                    "w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[
                    "msa_question", "dialect_question", "sub_variety"])
                writer.writeheader()
                for i in range(5):
                    writer.writerow({"msa_question": f"msa {i}",
                                     "dialect_question": f"dia {i}",
                                     "sub_variety": "Test variety"})

            calls = []
            original_seed, original_call = translate.SEED, translate.call
            translate.SEED = "wahm_seed_msa.csv"
            translate.DIALECT_NAMES["test"] = "Test Arabic"
            translate.call = lambda *args, **kwargs: (
                calls.append(args) or "نص", "resolved-model")
            old = os.getcwd()
            os.chdir(directory)
            try:
                translate.translate("test", model="requested-model")
                translate.translate("test", model="requested-model")
            finally:
                os.chdir(old)
                translate.SEED, translate.call = original_seed, original_call
                del translate.DIALECT_NAMES["test"]
            self.assertEqual(len(calls), 2)  # translation + backtranslation once


class GenerationTests(unittest.TestCase):
    def test_requested_multilingual_model_ids_are_registered(self):
        self.assertEqual(generate.MODELS["qwen3-8b"]["model_id"],
                         "qwen/qwen3-8b")
        self.assertEqual(generate.MODELS["llama-3.3-70b"]["model_id"],
                         "meta-llama/llama-3.3-70b-instruct")
        self.assertEqual(generate.MODELS["gemma-2-9b"]["model_id"],
                         "google/gemma-2-9b-it")
        self.assertEqual(generate.MODELS["gemma-2-9b-base"]["model_id"],
                         "google/gemma-2-9b")
        self.assertEqual(generate.MODELS["gemma-2-9b-base"]["api_mode"],
                         "completion")

    def test_qwen3_disables_thinking_mode(self):
        class Completions:
            def create(self, **kwargs):
                from types import SimpleNamespace
                self.kwargs = kwargs
                return SimpleNamespace(
                    model="Qwen/Qwen3-8B",
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="الدوحة"))])

        from types import SimpleNamespace
        completions = Completions()
        generate._clients["qwen3-8b"] = SimpleNamespace(
            chat=SimpleNamespace(completions=completions))
        try:
            answer, error, _, _, _ = generate.generate_one(
                "qwen3-8b", "ما هي عاصمة قطر؟", "msa", "direct", 0.0)
        finally:
            generate._clients.clear()
        self.assertEqual((answer, error), ("الدوحة", ""))
        self.assertEqual(completions.kwargs["extra_body"], {
            "chat_template_kwargs": {"enable_thinking": False}})

    def test_responses_model_omits_unsupported_temperature(self):
        class Responses:
            def create(self, **kwargs):
                from types import SimpleNamespace
                self.kwargs = kwargs
                return SimpleNamespace(
                    model="gpt-5.5-2026-04-24", output_text="الدوحة")

        from types import SimpleNamespace
        responses = Responses()
        generate._clients["gpt55"] = SimpleNamespace(responses=responses)
        try:
            answer, error, pivot, resolved, pivot_model = generate.generate_one(
                "gpt55", "ما هي عاصمة قطر؟", "msa", "direct", 0.0)
        finally:
            generate._clients.clear()
        self.assertEqual(error, "")
        self.assertEqual((answer, resolved),
                         ("الدوحة", "gpt-5.5-2026-04-24"))
        self.assertEqual(responses.kwargs["input"], [
            {"role": "user", "content": "ما هي عاصمة قطر؟"}])
        self.assertEqual(responses.kwargs["max_output_tokens"], 2048)
        self.assertNotIn("temperature", responses.kwargs)

    def test_responses_model_retries_empty_payload(self):
        class Responses:
            calls = 0
            budgets = []

            def create(self, **kwargs):
                from types import SimpleNamespace
                self.calls += 1
                self.budgets.append(kwargs["max_output_tokens"])
                return SimpleNamespace(
                    model="gpt-5.5-2026-04-24",
                    output_text="" if self.calls == 1 else "إجابة",
                    status="completed",
                    incomplete_details=None)

        from types import SimpleNamespace
        responses = Responses()
        generate._clients["gpt55"] = SimpleNamespace(responses=responses)
        try:
            answer, error, _, _, _ = generate.generate_one(
                "gpt55", "سؤال", "msa", "direct", 0.0)
        finally:
            generate._clients.clear()
        self.assertEqual((answer, error), ("إجابة", ""))
        self.assertEqual(responses.calls, 2)
        self.assertEqual(responses.budgets, [2048, 4096])

    def test_base_model_uses_text_completion_endpoint(self):
        class Completions:
            def create(self, **kwargs):
                from types import SimpleNamespace
                self.kwargs = kwargs
                return SimpleNamespace(
                    model="resolved-fanar",
                    choices=[SimpleNamespace(text="الدوحة")])

        from types import SimpleNamespace
        completions = Completions()
        generate._clients["fanar"] = SimpleNamespace(completions=completions)
        old_mode = generate.MODELS["fanar"]["api_mode"]
        generate.MODELS["fanar"]["api_mode"] = "completion"
        try:
            answer, error, pivot, resolved, pivot_model = generate.generate_one(
                "fanar", "ما هي عاصمة قطر؟", "msa", "direct", 0.0)
        finally:
            generate.MODELS["fanar"]["api_mode"] = old_mode
            generate._clients.clear()
        self.assertEqual(error, "")
        self.assertEqual((answer, resolved), ("الدوحة", "resolved-fanar"))
        self.assertEqual(completions.kwargs["prompt"], "ما هي عاصمة قطر؟")

    def test_msa_pivot_records_both_resolved_models(self):
        class Completions:
            calls = 0

            def create(self, **kwargs):
                from types import SimpleNamespace
                self.calls += 1
                text = "سؤال فصيح" if self.calls == 1 else "إجابة"
                return SimpleNamespace(
                    model=f"resolved-{self.calls}",
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=text))])

        from types import SimpleNamespace
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        generate._clients["gpt4o"] = client
        try:
            answer, error, pivot, resolved, pivot_model = generate.generate_one(
                "gpt4o", "سؤال خليجي", "gulf", "msa_pivot", 0.0)
        finally:
            generate._clients.clear()
        self.assertEqual(error, "")
        self.assertEqual((pivot, answer), ("سؤال فصيح", "إجابة"))
        self.assertEqual((pivot_model, resolved), ("resolved-1", "resolved-2"))


class FinalizationTests(unittest.TestCase):
    def test_rewrite_survives_low_backtranslation_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidate = directory / "candidates_test.csv"
            validation = directory / "validation_test.csv"
            candidate.write_text(
                "qid,question_msa,gold_answer,dialect_candidate,backtranslation_msa,sub_variety\n"
                "q1,السؤال,الجواب,ترجمة سيئة,مختلف تماما,Test\n",
                encoding="utf-8")
            validation.write_text(
                "qid,APPROVE_or_REWRITE,your_rewrite,fluency_1_5\n"
                "q1,REWRITE,ترجمة صحيحة,5\n", encoding="utf-8")
            old = os.getcwd()
            os.chdir(directory)
            try:
                metrics = finalize.run("test", min_backtrans=0.9)
            finally:
                os.chdir(old)
            self.assertEqual(metrics["n_final_benchmark"], 1)
            self.assertEqual(metrics["rewrite_rate"], 1.0)


class JudgeSplitTests(unittest.TestCase):
    def test_binary_training_file_is_consistent(self):
        rows = load_judge_rows(ROOT / "judge_train.csv")
        self.assertEqual(len(rows), 4195)
        self.assertEqual(sum(row["label"] for row in rows), 2347)
        self.assertEqual(len({row["sample_index"] for row in rows}), 300)

    def test_question_groups_do_not_leak(self):
        rows = load_judge_rows(ROOT / "judge_train.csv")
        train, test = grouped_split(rows)
        train_groups = {rows[i]["sample_index"] for i in train}
        test_groups = {rows[i]["sample_index"] for i in test}
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_train_validation_test_groups_are_disjoint(self):
        rows = load_judge_rows(ROOT / "judge_train.csv")
        partitions = grouped_train_validation_test_split(rows)
        groups = [{rows[i]["sample_index"] for i in partition}
                  for partition in partitions]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))

    def test_binary_label_mismatch_is_rejected(self):
        row = {column: "0" for column in judge_data.LABEL_COLUMNS}
        row["hallucinated"] = "1"
        with self.assertRaises(ValueError):
            judge_data.binary_label(row)

    def test_compact_binary_only_row_is_supported(self):
        self.assertEqual(judge_data.binary_label({"hallucinated": "1"}), 1)

    def test_specified_row_split_is_seeded_stratified_80_20(self):
        rows = load_judge_rows(ROOT / "judge_train.csv")
        train, test = stratified_row_split(rows)
        self.assertEqual((len(train), len(test)), (3356, 839))
        self.assertEqual(sum(rows[i]["label"] for i in train), 1878)
        self.assertEqual(sum(rows[i]["label"] for i in test), 469)
        self.assertEqual([train, test], stratified_row_split(rows))

    def test_snapshot_audit_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            model_file = snapshot / "model.bin"
            model_file.write_bytes(b"weights")
            audit = {
                "model_id": "model",
                "revision": "commit",
                "snapshot_dir": str(snapshot),
                "files": [{
                    "path": "model.bin",
                    "bytes": len(b"weights"),
                    "sha256": hashlib.sha256(b"weights").hexdigest(),
                }],
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            verify_hf_snapshot(audit_path, "model", "commit")
            model_file.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_hf_snapshot(audit_path, "model", "commit")


class ModelLoadingTests(unittest.TestCase):
    def test_local_snapshot_does_not_forward_hub_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_model_load_kwargs(directory, "commit"), {})

    def test_hub_model_forwards_pinned_revision(self):
        self.assertEqual(
            _model_load_kwargs("organization/model", "commit"),
            {"revision": "commit"})


class AnalysisTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_in_rank(self):
        adjusted = analyze_results._benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertEqual([round(value, 6) for value in adjusted],
                         [0.03, 0.04, 0.04])

    def test_hds_uses_only_qids_paired_with_msa(self):
        def row(qid, variety, decision):
            return {"qid": qid, "variety": variety, "condition": "direct",
                    "model": "m", "family": "test",
                    "combined_decision": decision, "degeneration_reasons": ""}

        rows = [row("q1", "msa", "clean"),
                row("q2", "msa", "hallucinated"),
                row("q1", "gulf", "hallucinated"),
                row("q3", "gulf", "hallucinated")]
        result = analyze_results.analyze(rows)[0]
        self.assertEqual(result["n_paired_factual"], 1)
        self.assertEqual(result["hds"], 1.0)

    def test_transitions_mcnemar_and_degeneration_are_separate(self):
        def row(qid, variety, decision):
            return {"qid": qid, "variety": variety, "condition": "direct",
                    "model": "m", "family": "test",
                    "combined_decision": decision,
                    "degeneration_reasons": (
                        "heavy_repetition" if decision == "degeneration" else "")}

        rows = [
            row("q1", "msa", "clean"),
            row("q2", "msa", "factual_hallucination"),
            row("q3", "msa", "degeneration"),
            row("q1", "gulf", "factual_hallucination"),
            row("q2", "gulf", "clean"),
            row("q3", "gulf", "clean"),
        ]
        result = analyze_results.analyze(rows, bootstrap_iterations=100)[0]
        self.assertEqual(result["n_paired_total"], 3)
        self.assertEqual(result["n_paired_factual"], 2)
        self.assertEqual(result["clean_to_hallucination"], 1)
        self.assertEqual(result["hallucination_to_clean"], 1)
        self.assertEqual(result["n_degenerated_msa"], 1)
        self.assertEqual(result["mcnemar_exact_p"], 1.0)

    def test_analysis_run_keeps_same_reported_model_runs_separate(self):
        def row(run, qid, variety, decision):
            return {"qid": qid, "variety": variety, "condition": "direct",
                    "model": "fanar", "analysis_run": run,
                    "family": "arabic_centric",
                    "combined_decision": decision, "degeneration_reasons": ""}

        rows = [
            row("fanar", "q1", "msa", "clean"),
            row("fanar", "q1", "gulf", "factual_hallucination"),
            row("fanar_instruct", "q1", "msa", "factual_hallucination"),
            row("fanar_instruct", "q1", "gulf", "clean"),
        ]
        results = analyze_results.analyze(rows, bootstrap_iterations=100)
        self.assertEqual({result["model"] for result in results},
                         {"fanar", "fanar_instruct"})
        self.assertEqual({result["hds"] for result in results}, {1.0, -1.0})


class LayerThreeTests(unittest.TestCase):
    def test_kappa_is_one_for_identical_labels(self):
        self.assertEqual(evaluate_layer3.cohen_kappa([0, 1, 1], [0, 1, 1]), 1.0)

    def test_stratified_sample_is_deterministic_and_covers_strata(self):
        rows = []
        for model in ("a", "b"):
            for decision in ("clean", "hallucinated"):
                rows.append({"model": model, "combined_decision": decision,
                             "layer2_hallucination_probability": "",
                             "qid": f"{model}-{decision}"})
        first = prepare_layer3.stratified_sample(rows, 4, seed=42)
        second = prepare_layer3.stratified_sample(rows, 4, seed=42)
        self.assertEqual([row["qid"] for row in first],
                         [row["qid"] for row in second])
        self.assertEqual(len(first), 4)

    def test_judge_v2_audit_sampling_covers_routes_and_varieties(self):
        rows = []
        for variety in ("msa", "gulf", "egyptian", "levantine", "sudanese"):
            for decision, probability in (
                    ("degeneration", ""), ("clean", "0.49"),
                    ("factual_hallucination", "0.51")):
                rows.append({
                    "variety": variety,
                    "combined_decision": decision,
                    "layer2_hallucination_probability": probability,
                    "layer2_decision_threshold": "0.5" if probability else "",
                    "qid": f"{variety}-{decision}",
                })
        selected = prepare_judge_audit.stratified_sample(rows, 15, seed=42)
        self.assertEqual(len(selected), 15)
        self.assertEqual({row["variety"] for row in selected},
                         {"msa", "gulf", "egyptian", "levantine", "sudanese"})
        self.assertEqual({prepare_judge_audit.audit_route(row) for row in selected},
                         {"layer1_degeneration", "layer2_clean",
                          "layer2_hallucination"})

    def test_judge_v2_audit_metrics_keep_degeneration_separate(self):
        truth = ["clean", "factual_hallucination", "degeneration"]
        predicted = ["clean", "degeneration", "degeneration"]
        metrics = evaluate_judge_audit._metrics(truth, predicted)
        self.assertEqual(metrics["accuracy"], 0.666667)
        self.assertEqual(
            metrics["confusion"]["factual_hallucination"]["degeneration"], 1)
        self.assertEqual(metrics["per_class"]["degeneration"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
