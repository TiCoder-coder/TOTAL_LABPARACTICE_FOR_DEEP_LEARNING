"""Safe tests for Practice 3 v2.1 preparation; no training is executed."""

from __future__ import annotations

import json
import unittest

from transformers import AutoTokenizer

from processing_own_phase.experiment_protocol_v2 import sha256_payload
from processing_own_phase.experiment_runner_v2_1 import build_parser, load_and_verify_metadata
from processing_own_phase.tokenizer_correction_v2_1 import (
    MODEL_IDENTIFIER,
    RUN_IDS,
    TOKENIZER_IDENTIFIER,
    V21_RESULT_DIR,
    build_configs,
    load_json,
    resolve_model_identifier,
    resolve_tokenizer_identifier,
    validate_configs,
    validate_registry,
    validate_tokenizer_integrity,
)


class BrokenTokenizer:
    vocab_size = 5
    unk_token_id = 1
    pad_token_id = 0
    cls_token_id = 2
    sep_token_id = 3
    mask_token_id = 4
    all_special_ids = [0, 1, 2, 3, 4]

    def tokenize(self, text):
        return ["[UNK]"] * len(text.split())

    def convert_tokens_to_ids(self, tokens):
        return [1] * len(tokens)

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[2] + [1] * len(text.split()) + [3] for text in texts]}


class TokenizerCorrectionTests(unittest.TestCase):
    def test_model_and_tokenizer_resolution_are_separate(self):
        self.assertEqual(resolve_model_identifier(), MODEL_IDENTIFIER)
        self.assertEqual(resolve_tokenizer_identifier(), TOKENIZER_IDENTIFIER)
        self.assertNotEqual(MODEL_IDENTIFIER, TOKENIZER_IDENTIFIER)

    def test_real_tokenizer_integrity_passes(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_IDENTIFIER, local_files_only=True)
        report = validate_tokenizer_integrity(
            tokenizer, ["this movie is good and positive", "this movie is bad and negative"]
        )
        self.assertEqual(report["vocab_size"], 30522)
        self.assertEqual(report["tokenizer_integrity"], "PASS")
        self.assertLessEqual(report["unk_ratio"], 0.05)
        self.assertGreater(report["unique_encoded_sequence_count"], 1)

    def test_broken_vocab_and_high_unk_are_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_tokenizer_integrity(BrokenTokenizer(), ["this movie good", "bad negative"])

    def test_non_distinct_encodings_are_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_IDENTIFIER, local_files_only=True)
        with self.assertRaises(RuntimeError):
            validate_tokenizer_integrity(tokenizer, ["same movie text", "same movie text"])


class ProtocolAndCliTests(unittest.TestCase):
    def test_real_configs_registry_holdout_and_authorization(self):
        configs = [load_json(V21_RESULT_DIR / "experiments" / run_id / "run_config.json") for run_id in RUN_IDS]
        validate_configs(configs)
        registry = load_json(V21_RESULT_DIR / "experiment_registry.json")
        validate_registry(registry, configs, require_planned=True)
        self.assertEqual([item["status"] for item in registry["runs"]], ["PLANNED"] * 3)
        holdout = load_json(V21_RESULT_DIR / "holdout_access_state.json")
        self.assertFalse(holdout["holdout_access_allowed"])
        self.assertEqual(holdout["holdout_evaluation_count"], 0)
        authorization = load_json(V21_RESULT_DIR / "part_03_2_f_training_authorization.json")
        supplied = authorization.pop("authorization_hash")
        self.assertEqual(supplied, sha256_payload(authorization))
        self.assertTrue(authorization["training_authorized"])

    def test_cli_requires_one_known_run(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--run-id", RUN_IDS[0]]).run_id, RUN_IDS[0])
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--run-id", "unknown"])

    def test_real_first_run_preflight_is_read_only(self):
        metadata = load_and_verify_metadata(RUN_IDS[0])
        self.assertEqual(metadata["config"]["run_id"], RUN_IDS[0])
        self.assertEqual(metadata["registry"]["runs"][0]["status"], "PLANNED")


if __name__ == "__main__":
    unittest.main()
