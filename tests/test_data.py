import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.data import DataError, inspect_image, parse_voc, prepare_dataset, resolve_image, split_records, apply_curation, source_fingerprint


def xml(box="<xmin>1</xmin><ymin>1</ymin><xmax>200</xmax><ymax>200</ymax>", name="crazing"):
    return f"<annotation><filename>crazing_1.png</filename><size><width>200</width><height>200</height></size><object><name>{name}</name><bndbox>{box}</bndbox></object></annotation>"


class DataTests(unittest.TestCase):
    def parse(self, content):

        import io
        return parse_voc(io.StringIO(content))

    def test_voc_coordinate_conversion(self):
        self.assertEqual(self.parse(xml())["boxes"], [[0, 0, 0, 200, 200]])

    def test_reject_bad_boxes_and_classes(self):
        for box in ["<xmin>0</xmin><ymin>1</ymin><xmax>2</xmax><ymax>3</ymax>",
                    "<xmin>2</xmin><ymin>1</ymin><xmax>1</xmax><ymax>3</ymax>",
                    "<xmin>nan</xmin><ymin>1</ymin><xmax>2</xmax><ymax>3</ymax>",
                    "<xmin>1</xmin><ymin>1</ymin><xmax>201</xmax><ymax>3</ymax>"]:
            with self.subTest(box=box), self.assertRaises(DataError):
                self.parse(xml(box))
        for content in [xml(name="unknown"), "<annotation>", "<annotation/>"]:
            with self.assertRaises(DataError):
                self.parse(content)

    def test_multiclass_preserves_object_labels_and_acquisition_category(self):
        content = xml(name="inclusion")
        content = content.replace("</annotation>", "<object><name>patches</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>")
        result = self.parse(content)
        self.assertEqual(result["class_id"], 0)
        self.assertEqual([b[0] for b in result["boxes"]], [1, 2])
        self.assertEqual(result["boxes"][1], [2, 9, 19, 30, 40])

    def test_exact_duplicate_boxes_only_are_canonicalized(self):
        content = xml()
        obj = content[content.index("<object>"):content.index("</object>") + len("</object>")]

        content = content.replace("</annotation>", obj + obj.replace("crazing", "inclusion") +
                                  obj.replace("<xmin>1</xmin>", "<xmin>2</xmin>") + "</annotation>")
        result = self.parse(content)
        self.assertEqual(result["raw_box_count"], 4)
        self.assertEqual(result["removed_duplicate_count"], 1)
        self.assertEqual(result["boxes"], [[0, 0, 0, 200, 200], [0, 1, 0, 200, 200], [1, 0, 0, 200, 200]])
        self.assertEqual(result["removed_duplicates"], [{"box": [0, 0, 0, 200, 200], "removed": 1}])

    def test_extensionless_filename_matching_and_ambiguity(self):
        result = self.parse(xml().replace("crazing_1.png", "crazing_1"))
        image = Path("images/crazing_1.jpg")
        annotation = Path("annotations/crazing_1.xml")
        self.assertEqual(resolve_image(result["filename"], annotation, {}, {"crazing_1": [image]}), image)
        for matches in ([], [image, Path("other/crazing_1.png")]):
            with self.assertRaises(DataError):
                resolve_image(result["filename"], annotation, {}, {"crazing_1": matches})
        with self.assertRaises(DataError):
            resolve_image("crazing_1.jpg", Path("inclusion_1.xml"), {"crazing_1.jpg": [image]}, {})
        with self.assertRaises(DataError):
            resolve_image("crazing_1.jpg", annotation, {"crazing_1.jpg": [image, image]}, {})

    def test_reject_unsafe_and_noncanonical_filename(self):
        for name in ("../crazing_1.jpg", "other\\crazing_1.jpg", "unknown_1.jpg", "crazing_0.jpg", "crazing_1.exe", "arbitrary.png"):
            with self.subTest(name=name), self.assertRaises(DataError):
                self.parse(xml().replace("crazing_1.png", name))

    def test_deterministic_stratified_split(self):
        records = [dict(pixel_sha256=f"{cls}:{i}", class_id=cls, boxes=[[cls, 0, 0, 2, 2]])
                   for cls in range(6) for i in range(300)]
        result = split_records(records)
        self.assertEqual(result, split_records(list(reversed(records))))
        self.assertNotEqual(result, split_records(records, 43))
        for cls in range(6):
            self.assertEqual(Counter(result[r["pixel_sha256"]] for r in records if r["class_id"] == cls),
                             {"train": 210, "val": 45, "test": 45})

    def test_duplicate_group_and_conflict(self):
        record = dict(pixel_sha256="same", class_id=0, boxes=[[0, 0, 0, 2, 2]])
        result = split_records([record, dict(record)])
        self.assertEqual(len(result), 1)
        with self.assertRaises(DataError):
            split_records([record, dict(record, boxes=[[1, 0, 0, 2, 2]])])
        self.assertEqual(len(split_records([record, dict(record, class_id=1)])), 1)

    def test_hash_bound_curation_and_indivisible_groups(self):
        records = [dict(image=f"crazing_{i}.jpg", annotation=f"crazing_{i}.xml",
                        image_sha256=str(i), annotation_sha256=str(i), pixel_sha256=str(i),
                        class_id=0, boxes=[[0, 0, 0, 2, 2]]) for i in range(3)]
        review = dict(source_fingerprint=source_fingerprint(records), reviewer="test",
                      limitations=["heuristic"], groups=[dict(images=["crazing_0.jpg", "crazing_1.jpg"], reason="related")])
        kept, excluded = apply_curation(records, review)
        split = split_records(kept)
        self.assertEqual(split["0"], split["1"])
        with self.assertRaises(DataError):
            apply_curation(records, dict(review, source_fingerprint="stale"))
        with self.assertRaises(DataError):
            apply_curation(records, dict(review, groups=review["groups"] * 2))
        duplicate = dict(records[1], pixel_sha256="0")
        pair = [records[0], duplicate]
        with self.assertRaises(DataError):
            apply_curation(pair, dict(review, source_fingerprint=source_fingerprint(pair),
                           quarantine=[dict(image="crazing_0.jpg", reason="conflict")], groups=[]))

    def test_missing_source_refused(self):
        with self.assertRaises(DataError):
            prepare_dataset(Path("/nonexistent-neu-fixture"), Path("/nonexistent-output"))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow unavailable: real decode/export tests pending")
    def test_decode_and_export_fixture(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            Image.new("L", (200, 200), 70).save(source / "crazing_1.png")
            (source / "crazing_1.xml").write_text(xml(), encoding="utf-8")
            with self.assertRaises(DataError):
                prepare_dataset(source, root / "full")
            report = prepare_dataset(source, root / "output", allow_partial=True)
            self.assertFalse(report["g1_passed"])
            self.assertTrue((root / "output/data.yaml").exists())
            label = next((root / "output/labels").rglob("*.txt"))
            self.assertEqual(label.read_text().strip(), "0 0.5000000000 0.5000000000 1.0000000000 1.0000000000")
            with self.assertRaises(DataError):
                prepare_dataset(source, root / "output", allow_partial=True)
            Image.new("L", (200, 200), 80).save(source / "crazing_2.png")
            (source / "crazing_2.xml").write_text(xml().replace("crazing_1.png", "crazing_2.png"), encoding="utf-8")
            with self.assertRaisesRegex(DataError, "near-duplicate"):
                prepare_dataset(source, root / "blocked", allow_partial=True)
            self.assertFalse((root / "blocked/data.yaml").exists())
            (source / "crazing_1.png").write_bytes(b"corrupt")
            with self.assertRaises(DataError):
                inspect_image(source / "crazing_1.png")


if __name__ == "__main__":
    unittest.main()
